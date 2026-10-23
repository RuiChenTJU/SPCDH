import math
import torch
from torch import nn
import torch.nn.functional as F

def orthogonalize(vectors, eps=1e-06):
    basis = []
    for direction in vectors.unbind(dim=1):
        for _ in range(2):
            for previous in basis:
                direction = direction - (direction * previous).sum(-1, keepdim=True) * previous
        basis.append(F.normalize(direction, dim=-1, eps=eps))
    return torch.stack(basis, dim=1)

def project_and_disentangle(feature, basis):
    projection = torch.einsum('bhd,bh->bd', basis, torch.einsum('bd,bhd->bh', feature, basis))
    return (F.normalize(projection, dim=-1), feature - projection)

class GuidedMultimodalFusion(nn.Module):

    def __init__(self, visual_dim, audio_dim, motion_dim, unified_dim, hidden_dim, num_classes, tau_assign):
        super().__init__()
        if num_classes < 2 or tau_assign <= 0:
            raise ValueError('At least two classes and a positive alignment temperature are required')
        self.unified_dim, self.num_classes, self.tau_assign = (unified_dim, num_classes, tau_assign)

        def projector(input_dim):
            return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, unified_dim))
        self.proj_visual = projector(visual_dim)
        self.proj_audio = projector(audio_dim)
        self.proj_motion = projector(motion_dim)
        self.shared_classifier = nn.Linear(unified_dim, num_classes, bias=False)
        self.lambda_guide = nn.Parameter(torch.tensor(1.0))

    def compute_confidence(self, logits):
        probabilities = logits.softmax(dim=-1)
        entropy = -(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)
        return 1.0 - entropy / math.log(self.num_classes)

    def build_correlation_matrix(self, logits_v, logits_a, logits_m):
        confidence_v = self.compute_confidence(logits_v)
        return torch.stack([F.cosine_similarity(logits_v, auxiliary, dim=-1) * confidence_v * self.compute_confidence(auxiliary) for auxiliary in (logits_a, logits_m)], dim=-1)

    def guided_cross_attention(self, visual, audio, motion, guidance):
        auxiliary = torch.stack([audio, motion], dim=1)
        scores = (visual.unsqueeze(1) * auxiliary).sum(-1) / math.sqrt(self.unified_dim)
        weights = (scores + self.lambda_guide * guidance).softmax(dim=-1)
        return visual + (weights.unsqueeze(-1) * auxiliary).sum(dim=1)

    def forward(self, visual, audio, motion, labels=None, return_loss=False):
        visual, audio, motion = [x.mean(dim=1) if x.ndim == 3 else x for x in (visual, audio, motion)]
        visual, audio, motion = (self.proj_visual(visual), self.proj_audio(audio), self.proj_motion(motion))
        logits_v, logits_a, logits_m = [self.shared_classifier(x) for x in (visual, audio, motion)]
        guidance = self.build_correlation_matrix(logits_v, logits_a, logits_m)
        fused = self.guided_cross_attention(visual, audio, motion, guidance)
        if not return_loss:
            return fused
        targets = torch.arange(len(visual), device=visual.device)
        visual_unit = F.normalize(visual, dim=-1)
        loss_align = sum((F.cross_entropy(visual_unit @ F.normalize(x, dim=-1).T / self.tau_assign, targets) for x in (audio, motion)))
        losses = {'loss_align': loss_align}
        if labels is not None:
            losses['loss_cls'] = F.binary_cross_entropy_with_logits(self.shared_classifier(fused), labels, reduction='sum') / len(fused)
        return (fused, losses)

class PriorGuidedDisentanglement(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_basis):
        super().__init__()
        if not 1 <= num_basis <= input_dim:
            raise ValueError('Invalid basis count')
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.direction_queries = nn.Parameter(torch.randn(num_basis, input_dim))
        self.mlp_refine = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.LayerNorm(input_dim))

    def forward(self, feature, point_cloud):
        point_cloud = F.normalize(point_cloud, dim=-1)
        queries = self.query_projection(feature).unsqueeze(1) + self.direction_queries
        weights = (queries @ point_cloud.transpose(1, 2)).softmax(dim=-1)
        basis = orthogonalize(self.mlp_refine(weights @ point_cloud))
        semantic, context = project_and_disentangle(feature, basis)
        return (semantic, context, basis)

class SemanticSubspacePredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_basis):
        super().__init__()
        self.num_basis, self.input_dim = (num_basis, input_dim)
        self.basis_predictor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, num_basis * input_dim))

    def forward(self, feature):
        return orthogonalize(self.basis_predictor(feature).reshape(-1, self.num_basis, self.input_dim))
    project_and_disentangle = staticmethod(project_and_disentangle)
CAMA = GuidedMultimodalFusion
