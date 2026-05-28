import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GuidedMultimodalFusion(nn.Module):
    def __init__(
        self,
        visual_dim=512,
        audio_dim=768,
        motion_dim=768,
        unified_dim=512,
        num_classes=20,
        tau_assign=0.07,
        lambda_align=1.0,
        lambda_cls=0.5
    ):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_classes = num_classes
        self.tau_assign = tau_assign
        self.lambda_align = lambda_align
        self.lambda_cls = lambda_cls

        self.proj_visual = nn.Sequential(
            nn.Linear(visual_dim, unified_dim),
            nn.BatchNorm1d(unified_dim),
            nn.ReLU(),
            nn.Linear(unified_dim, unified_dim)
        )
        self.proj_audio = nn.Sequential(
            nn.Linear(audio_dim, unified_dim),
            nn.BatchNorm1d(unified_dim),
            nn.ReLU(),
            nn.Linear(unified_dim, unified_dim)
        )
        self.proj_motion = nn.Sequential(
            nn.Linear(motion_dim, unified_dim),
            nn.BatchNorm1d(unified_dim),
            nn.ReLU(),
            nn.Linear(unified_dim, unified_dim)
        )

        self.shared_classifier = nn.Sequential(
            nn.Linear(unified_dim, unified_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(unified_dim // 2, num_classes)
        )

        self.query_proj = nn.Linear(unified_dim, unified_dim)
        self.key_proj = nn.Linear(unified_dim, unified_dim)
        self.value_proj = nn.Linear(unified_dim, unified_dim)
        self.lambda_guide = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Sequential(
            nn.Linear(unified_dim, unified_dim),
            nn.LayerNorm(unified_dim)
        )

    def compute_confidence(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=logits.device))
        return 1.0 - (entropy / max_entropy)

    def build_correlation_matrix(self, logits_v, logits_a, logits_m, conf_v, conf_a, conf_m):
        logits_v_norm = F.normalize(logits_v, dim=-1)
        logits_a_norm = F.normalize(logits_a, dim=-1)
        logits_m_norm = F.normalize(logits_m, dim=-1)

        cos_v_a = torch.sum(logits_v_norm * logits_a_norm, dim=-1)
        cos_v_m = torch.sum(logits_v_norm * logits_m_norm, dim=-1)
        w_v_a = cos_v_a * (conf_v * conf_a)
        w_v_m = cos_v_m * (conf_v * conf_m)
        return torch.stack([w_v_a, w_v_m], dim=-1)

    def guided_cross_attention(self, visual_feat, audio_feat, motion_feat, W_guide):
        Q = self.query_proj(visual_feat)
        KV_features = torch.stack([audio_feat, motion_feat], dim=1)
        K = self.key_proj(KV_features)
        V = self.value_proj(KV_features)
        scores = torch.matmul(Q.unsqueeze(1), K.transpose(1, 2)) / (self.unified_dim ** 0.5)
        scores = scores.squeeze(1) + self.lambda_guide * W_guide
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.sum(attn_weights.unsqueeze(-1) * V, dim=1)
        return visual_feat + self.out_proj(out)

    def forward(self, visual, audio, motion, labels=None, return_loss=False):
        if visual.dim() == 3:
            visual = visual.mean(dim=1)
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        if motion.dim() == 3:
            motion = motion.mean(dim=1)

        visual_proj = self.proj_visual(visual)
        audio_proj = self.proj_audio(audio)
        motion_proj = self.proj_motion(motion)

        visual_norm = F.normalize(visual_proj, dim=-1)
        audio_norm = F.normalize(audio_proj, dim=-1)
        motion_norm = F.normalize(motion_proj, dim=-1)

        logits_v = self.shared_classifier(visual_proj)
        logits_a = self.shared_classifier(audio_proj)
        logits_m = self.shared_classifier(motion_proj)

        W_guide = self.build_correlation_matrix(
            logits_v, logits_a, logits_m,
            self.compute_confidence(logits_v),
            self.compute_confidence(logits_a),
            self.compute_confidence(logits_m),
        )
        fused_features = self.guided_cross_attention(visual_proj, audio_proj, motion_proj, W_guide)

        if not return_loss:
            return fused_features

        losses = {}
        batch_size = visual.shape[0]
        sim_v_a = torch.matmul(visual_norm, audio_norm.t()) / self.tau_assign
        sim_v_m = torch.matmul(visual_norm, motion_norm.t()) / self.tau_assign
        targets = torch.arange(batch_size, device=visual.device)

        losses["loss_align"] = (
            F.cross_entropy(sim_v_a, targets)
            + F.cross_entropy(sim_v_m, targets)
        ) / 2.0 * self.lambda_align

        if labels is not None:
            losses["loss_cls"] = (
                F.binary_cross_entropy_with_logits(logits_v, labels)
                + F.binary_cross_entropy_with_logits(logits_a, labels)
                + F.binary_cross_entropy_with_logits(logits_m, labels)
            ) / 3.0 * self.lambda_cls

        return fused_features, losses


class PriorGuidedDisentanglement(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.mlp_aggregator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        self.use_attention = True

    def gram_schmidt_orthogonalize(self, vectors, eps=1e-8):
        _, N, _ = vectors.shape
        ortho_basis = []
        v0 = vectors[:, 0, :]
        ortho_basis.append(F.normalize(v0, p=2, dim=-1, eps=eps))
        for i in range(1, N):
            vi = vectors[:, i, :]
            for j in range(i):
                ej = ortho_basis[j]
                vi = vi - torch.sum(vi * ej, dim=-1, keepdim=True) * ej
            ortho_basis.append(F.normalize(vi, p=2, dim=-1, eps=eps))
        return torch.stack(ortho_basis, dim=1)

    def forward(self, f_i, point_cloud):
        batch_size, N, D = point_cloud.shape
        if self.use_attention:
            aggregated, _ = self.attention(
                query=f_i.unsqueeze(1),
                key=point_cloud,
                value=point_cloud
            )
            initial_basis = torch.cat([aggregated, point_cloud], dim=1)
        else:
            point_cloud_flat = point_cloud.view(-1, D)
            aggregated_flat = self.mlp_aggregator(point_cloud_flat)
            aggregated = aggregated_flat.view(batch_size, N, D).mean(dim=1)
            initial_basis = torch.cat([aggregated.unsqueeze(1), point_cloud], dim=1)

        ortho_basis = self.gram_schmidt_orthogonalize(initial_basis)
        projection = torch.zeros_like(f_i)
        for j in range(ortho_basis.shape[1]):
            ej = ortho_basis[:, j, :]
            projection = projection + torch.sum(f_i * ej, dim=-1, keepdim=True) * ej

        f_s = F.normalize(projection, p=2, dim=-1)
        f_c = f_i - projection
        return f_s, f_c, ortho_basis


class SemanticSubspacePredictor(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_basis: int = 31):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.basis_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_basis * input_dim),
        )

    def _gram_schmidt_orthogonalize(self, vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        _, N, _ = vectors.shape
        ortho_basis = []
        v0 = vectors[:, 0, :]
        ortho_basis.append(F.normalize(v0, p=2, dim=-1, eps=eps))
        for i in range(1, N):
            vi = vectors[:, i, :]
            for j in range(i):
                ej = ortho_basis[j]
                vi = vi - torch.sum(vi * ej, dim=-1, keepdim=True) * ej
            ortho_basis.append(F.normalize(vi, p=2, dim=-1, eps=eps))
        return torch.stack(ortho_basis, dim=1)

    def forward(self, f_i: torch.Tensor) -> torch.Tensor:
        batch_size = f_i.shape[0]
        basis_flat = self.basis_predictor(f_i)
        basis = basis_flat.view(batch_size, self.num_basis, self.input_dim)
        return self._gram_schmidt_orthogonalize(basis)

    def project_and_disentangle(self, f_i: torch.Tensor, ortho_basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projection = torch.zeros_like(f_i)
        for j in range(ortho_basis.shape[1]):
            ej = ortho_basis[:, j, :]
            projection = projection + torch.sum(f_i * ej, dim=-1, keepdim=True) * ej
        f_s = F.normalize(projection, p=2, dim=-1)
        f_c = f_i - projection
        return f_s, f_c


CAMA = GuidedMultimodalFusion
PGMD = PriorGuidedDisentanglement

__all__ = ["CAMA", "PGMD", "PriorGuidedDisentanglement", "SemanticSubspacePredictor"]
