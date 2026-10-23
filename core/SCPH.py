import torch
from torch import nn
import torch.nn.functional as F
from .PGMD import PriorGuidedDisentanglement, SemanticSubspacePredictor

class ContextAnchorModule(nn.Module):

    def __init__(self, feature_dim, num_anchors, beta):
        super().__init__()
        self.beta = beta
        self.anchors = nn.Parameter(torch.empty(num_anchors, feature_dim))
        nn.init.xavier_uniform_(self.anchors)

    def forward(self, context):
        assigned = self.anchors[torch.cdist(context, self.anchors).argmin(dim=1)]
        anchor = (context.detach() - assigned).square().sum(dim=-1).mean()
        commitment = (context - assigned.detach()).square().sum(dim=-1).mean()
        return (anchor + self.beta * commitment, assigned)

class SplitHashingModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, semantic_bits, context_bits):
        super().__init__()

        def head(bits):
            return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, bits))
        self.phi_s, self.phi_c = (head(semantic_bits), head(context_bits))

    def forward(self, semantic, context):
        semantic_code, context_code = (self.phi_s(semantic).tanh(), self.phi_c(context).tanh())
        return (semantic_code, context_code, torch.cat([semantic_code, context_code], dim=-1))

class SCPH(nn.Module):

    def __init__(self, *, input_dim, hidden_dim, hash_bits, num_classes, semantic_bits, context_bits, num_basis, kappa_max, lambda_sem, lambda_context, lambda_distill, lambda_quant, static_neg_K, neg_margin_zeta, gate_scale, gate_shift, context_num_anchors, context_beta):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hash_bits = hash_bits
        self.num_classes = num_classes
        self.semantic_bits = semantic_bits
        self.context_bits = context_bits
        self.num_basis = num_basis
        self.kappa_max = kappa_max
        self.lambda_sem = lambda_sem
        self.lambda_context = lambda_context
        self.lambda_distill = lambda_distill
        self.lambda_quant = lambda_quant
        self.static_neg_K = static_neg_K
        self.neg_margin_zeta = neg_margin_zeta
        self.gate_scale = gate_scale
        self.gate_shift = gate_shift
        self.context_num_anchors = context_num_anchors
        self.context_beta = context_beta
        if self.semantic_bits is None or self.context_bits is None or min(self.semantic_bits, self.context_bits) < 1 or (self.semantic_bits + self.context_bits != self.hash_bits):
            raise ValueError('Semantic and context bits must be positive and sum to hash_bits')
        if not 1 <= self.num_basis <= self.input_dim or self.kappa_max <= 0 or self.static_neg_K < 1:
            raise ValueError('Invalid basis count, kappa cap, or negative-class count')
        self.semantic_bits, self.context_bits = (self.semantic_bits, self.context_bits)
        self.prior_guided_disentanglement = PriorGuidedDisentanglement(self.input_dim, self.hidden_dim, self.num_basis)
        self.ssp = SemanticSubspacePredictor(self.input_dim, self.hidden_dim, self.num_basis)
        self.split_hash = SplitHashingModule(self.input_dim, self.hidden_dim, self.semantic_bits, self.context_bits)
        self.proxy_generator = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(self.hidden_dim, self.semantic_bits))
        self.global_proxies = nn.Parameter(torch.empty(self.num_classes, self.semantic_bits))
        nn.init.orthogonal_(self.global_proxies)
        self.context_anchor = ContextAnchorModule(self.input_dim, self.context_num_anchors, self.context_beta)

    def _estimate_kappa(self, point_cloud):
        points = F.normalize(point_cloud, dim=-1)
        resultant = points.mean(dim=1).norm(dim=-1).clamp(max=1.0 - 1e-06)
        denominator = (1.0 - resultant.square()).clamp_min(1e-06)
        return ((resultant * points.shape[-1] - resultant.pow(3)) / denominator).clamp(min=0.0, max=self.kappa_max)

    def _gate_from_kappa(self, kappa):
        return torch.sigmoid(self.gate_scale * (kappa / self.kappa_max - self.gate_shift))

    def generate_dynamic_proxy(self, semantic, point_cloud):
        weights = (semantic.unsqueeze(1) @ point_cloud.transpose(1, 2)).squeeze(1).softmax(dim=-1)
        mean = F.normalize((weights.unsqueeze(-1) * point_cloud).sum(dim=1), dim=-1)
        return self.proxy_generator(mean).tanh()

    def _anchor_proxy_loss(self, semantic_code, labels):
        similarity = F.normalize(semantic_code, dim=-1) @ F.normalize(self.global_proxies, dim=-1).T
        positive = -(similarity * labels).sum() / labels.sum().clamp_min(1)
        negative_similarity = similarity.masked_fill(labels.bool(), float('-inf'))
        hardest = negative_similarity.topk(min(self.static_neg_K, labels.shape[-1]), dim=-1).values
        valid = torch.isfinite(hardest)
        negative = F.relu(torch.where(valid, hardest, torch.zeros_like(hardest)) - self.neg_margin_zeta)
        negative = (negative * valid).sum() / valid.sum().clamp_min(1)
        return 0.5 * (positive + negative)

    def _sem_loss(self, semantic_code, proxy, kappa, labels, lambda_adaptive):
        adaptive = (self._gate_from_kappa(kappa) * (1.0 - F.cosine_similarity(semantic_code, proxy, dim=-1))).mean()
        return self._anchor_proxy_loss(semantic_code, labels) + lambda_adaptive * adaptive

    @staticmethod
    def _distill_loss(teacher, student):
        return (1.0 - F.cosine_similarity(teacher, student, dim=-1)).mean()

    def forward(self, feature, point_cloud=None, return_all=False, training=None):
        use_teacher = self.training if training is None else training
        proxy = kappa = teacher_basis = student_basis = None
        if use_teacher:
            if point_cloud is None:
                raise ValueError('Teacher training requires the semantic point cloud')
            point_cloud = F.normalize(point_cloud, dim=-1)
            semantic, context, teacher_basis = self.prior_guided_disentanglement(feature, point_cloud)
            student_basis = self.ssp(feature)
            kappa = self._estimate_kappa(point_cloud)
            proxy = self.generate_dynamic_proxy(semantic, point_cloud)
        else:
            student_basis = self.ssp(feature)
            semantic, context = self.ssp.project_and_disentangle(feature, student_basis)
        semantic_code, context_code, continuous = self.split_hash(semantic, context)
        binary = torch.where(continuous >= 0, torch.ones_like(continuous), -torch.ones_like(continuous))
        if use_teacher:
            binary = binary + (continuous - continuous.detach())
        if not return_all:
            return binary
        return {'b_i': binary, 'h_s': semantic_code, 'h_c': context_code, 'h_visual': continuous, 'f_s': semantic, 'f_c': context, 'p_i': proxy, 'kappa_i': kappa, 'basis_teacher': teacher_basis, 'basis_student': student_basis}

    def compute_loss(self, feature, point_cloud, labels, *, lambda_adaptive, return_outputs=False):
        out = self(feature, point_cloud, return_all=True, training=True)
        context_loss, _ = self.context_anchor(out['f_c'])
        parts = {'loss_sem': self._sem_loss(out['h_s'], out['p_i'], out['kappa_i'], labels, lambda_adaptive), 'loss_context': context_loss, 'loss_distill': self._distill_loss(out['basis_teacher'], out['basis_student']), 'loss_quant': (out['h_visual'].abs() - 1).square().mean()}
        total = sum((getattr(self, key) * parts[loss] for key, loss in (('lambda_sem', 'loss_sem'), ('lambda_context', 'loss_context'), ('lambda_distill', 'loss_distill'), ('lambda_quant', 'loss_quant'))))
        if return_outputs:
            return (total, parts, out)
        return (total, parts)

    @torch.no_grad()
    def get_hash_codes(self, feature):
        self.eval()
        return self(feature, training=False)
