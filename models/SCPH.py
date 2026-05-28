from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .PGMD import PriorGuidedDisentanglement, SemanticSubspacePredictor


class ContextAnchorModule(nn.Module):
    """VQ-style context anchor module, implements L_context in the paper."""

    def __init__(self, feature_dim: int = 512, num_anchors: int = 8, beta: float = 0.25):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_anchors = int(num_anchors)
        self.beta = float(beta)
        self.anchors = nn.Parameter(torch.randn(num_anchors, feature_dim))
        nn.init.xavier_uniform_(self.anchors)

    def forward(self, f_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = torch.cdist(f_c, self.anchors)
        min_indices = torch.argmin(dists, dim=1)
        a_hat = self.anchors[min_indices]
        loss_anchor = torch.mean((f_c.detach() - a_hat) ** 2)
        loss_commit = torch.mean((f_c - a_hat.detach()) ** 2)
        return loss_anchor + self.beta * loss_commit, a_hat


class SplitHashingModule(nn.Module):
    """Dual-stream projection: Phi_s maps f_i^s -> h_i^s, Phi_c maps f_i^c -> h_i^c."""

    def __init__(self, input_dim: int = 512, hash_bits: int = 64, hidden_dim: int = 256,
                 semantic_bits: int = None, context_bits: int = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hash_bits = int(hash_bits)
        if semantic_bits is not None and context_bits is not None:
            assert semantic_bits + context_bits == hash_bits, "SPCDH"
            self.semantic_bits = int(semantic_bits)
            self.context_bits = int(context_bits)
        else:
            assert hash_bits % 2 == 0, "SPCDH"
            self.semantic_bits = self.hash_bits // 2
            self.context_bits = self.hash_bits // 2
        self.phi_s = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.semantic_bits),
        )
        self.phi_c = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.context_bits),
        )

    def forward(self, f_s: torch.Tensor, f_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_s = torch.tanh(self.phi_s(f_s))
        h_c = torch.tanh(self.phi_c(f_c))
        h = torch.cat([h_s, h_c], dim=-1)
        return h_s, h_c, h


@dataclass
class SCPHConfig:
    input_dim: int = 512
    hash_bits: int = 64
    hidden_dim: int = 256
    num_classes: int = 63
    semantic_bits: int = None
    context_bits: int = None
    kappa_max: float = 200.0
    lambda_sem: float = 1.0
    lambda_context: float = 0.1
    lambda_distill: float = 1.0
    neg_margin_zeta: float = 0.2
    gate_scale: float = 8.0
    gate_shift: float = 0.30
    context_num_anchors: int = 8
    context_beta: float = 0.25


class SCPH(nn.Module):
    def __init__(self, cfg: SCPHConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.semantic_bits is not None and cfg.context_bits is not None:
            assert cfg.semantic_bits + cfg.context_bits == cfg.hash_bits, "SPCDH"
            self.semantic_bits = cfg.semantic_bits
            self.context_bits = cfg.context_bits
        else:
            assert self.cfg.hash_bits % 2 == 0, "SPCDH"
            self.semantic_bits = self.cfg.hash_bits // 2
            self.context_bits = self.cfg.hash_bits // 2

        self.prior_guided_disentanglement = PriorGuidedDisentanglement(
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
        )
        self.ssp = SemanticSubspacePredictor(
            input_dim=self.cfg.input_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_basis=31,
        )
        self.split_hash = SplitHashingModule(
            input_dim=self.cfg.input_dim,
            hash_bits=self.cfg.hash_bits,
            hidden_dim=self.cfg.hidden_dim,
            semantic_bits=self.semantic_bits,
            context_bits=self.context_bits,
        )
        self.proxy_generator = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.hidden_dim, self.semantic_bits),
        )
        self.global_proxies = nn.Parameter(torch.randn(self.cfg.num_classes, self.semantic_bits))
        nn.init.orthogonal_(self.global_proxies)
        self.context_anchor = ContextAnchorModule(
            feature_dim=self.cfg.input_dim,
            num_anchors=self.cfg.context_num_anchors,
            beta=self.cfg.context_beta,
        )

        self.lambda_sem = float(self.cfg.lambda_sem)
        self.lambda_context = float(self.cfg.lambda_context)
        self.lambda_distill = float(self.cfg.lambda_distill)
        self.gate_shift = float(self.cfg.gate_shift)
        self.gate_scale = float(self.cfg.gate_scale)
        self.neg_margin_zeta = float(self.cfg.neg_margin_zeta)

    def set_kappa_max(self, kappa_max: float) -> None:
        self.cfg.kappa_max = float(kappa_max)

    def set_lambda_distill(self, lambda_distill: float) -> None:
        self.lambda_distill = float(lambda_distill)

    def set_gate_shift(self, gate_shift: float) -> None:
        self.gate_shift = float(gate_shift)

    def set_gate_scale(self, gate_scale: float) -> None:
        self.gate_scale = float(gate_scale)

    def set_neg_margin_zeta(self, zeta: float) -> None:
        self.neg_margin_zeta = float(zeta)

    def _estimate_kappa(self, point_cloud: torch.Tensor) -> torch.Tensor:
        _, N, _ = point_cloud.shape
        if self.training:
            point_cloud = F.dropout(point_cloud, p=0.1)
        point_cloud = F.normalize(point_cloud, p=2, dim=-1)
        R_i = torch.norm(torch.sum(point_cloud, dim=1), p=2, dim=-1) / N
        R_i = torch.clamp(R_i, min=0.0, max=0.95)
        denom = torch.clamp(1.0 - R_i ** 2, min=1e-6)
        kappa_i = (R_i * 64.0 - R_i ** 3) / denom
        return torch.clamp(kappa_i, min=0.0, max=float(self.cfg.kappa_max))

    def _gate_from_kappa(self, kappa_i: torch.Tensor) -> torch.Tensor:
        kappa_norm = (kappa_i / float(self.cfg.kappa_max)).clamp(0.0, 1.0)
        return torch.sigmoid(float(self.gate_scale) * (kappa_norm - float(self.gate_shift)))

    def generate_dynamic_proxy(self, f_s_teacher: torch.Tensor,
                                point_cloud: torch.Tensor,
                                kappa_i: torch.Tensor) -> torch.Tensor:
        attn_scale = kappa_i.detach().clamp(min=1.0).unsqueeze(-1)
        similarities = torch.bmm(
            f_s_teacher.detach().unsqueeze(1),
            point_cloud.transpose(1, 2)
        ).squeeze(1) * attn_scale
        pi = F.softmax(similarities, dim=-1)
        mu_weighted = F.normalize(
            torch.sum(pi.unsqueeze(-1) * point_cloud, dim=1), p=2, dim=-1
        )
        return torch.tanh(self.proxy_generator(mu_weighted))

    def _anchor_proxy_loss(self, h_s: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Static anchor part of L_sem (Eq. L_sem in the paper):

            (1) positive term:
                sum_{i,c} y_{ic} * (1 - Cos(h_i^s, p_anchor^c)) / sum_{i,c} y_{ic}
            (2) negative term:
                sum_{i,c} (1 - y_{ic}) * max(Cos(h_i^s, p_anchor^c) - zeta, 0)
                / sum_{i,c} (1 - y_{ic})
        """
        if labels is None:
            return torch.tensor(0.0, device=h_s.device)
        h = F.normalize(h_s, p=2, dim=-1)
        P = F.normalize(self.global_proxies, p=2, dim=-1)
        cos = torch.matmul(h, P.t())
        y = labels.float()
        pos_term = ((1.0 - cos) * y).sum() / y.sum().clamp_min(1.0)
        neg_term = (
            F.relu(cos - float(self.neg_margin_zeta)) * (1.0 - y)
        ).sum() / (1.0 - y).sum().clamp_min(1.0)
        return pos_term + neg_term

    def _adaptive_proxy_loss(self, h_s: torch.Tensor,
                              p_i: torch.Tensor,
                              kappa_i: torch.Tensor) -> torch.Tensor:
        """Adaptive proxy part of L_sem: g(kappa_i) * (1 - Cos(h_i^s, p_i))."""
        if p_i is None or kappa_i is None:
            return torch.tensor(0.0, device=h_s.device)
        g = self._gate_from_kappa(kappa_i).detach()
        h_norm = F.normalize(h_s, p=2, dim=-1)
        p_norm = F.normalize(p_i, p=2, dim=-1)
        return torch.mean(g * (1.0 - torch.sum(h_norm * p_norm, dim=-1)))

    def _sem_loss(self, h_s: torch.Tensor, p_i: torch.Tensor,
                  kappa_i: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """L_sem = anchor pos/neg loss + g(kappa_i)(1 - Cos(h_i^s, p_i))."""
        return self._anchor_proxy_loss(h_s, labels) + self._adaptive_proxy_loss(h_s, p_i, kappa_i)

    def _distill_loss(self, basis_teacher: torch.Tensor, basis_student: torch.Tensor) -> torch.Tensor:
        """L_distill = 1 - mean Cos(e_i^h, e_hat_i^h)."""
        if basis_teacher is None or basis_student is None:
            device = (basis_student if basis_student is not None else basis_teacher).device
            return torch.tensor(0.0, device=device)
        min_h = min(basis_teacher.shape[1], basis_student.shape[1])
        et = F.normalize(basis_teacher[:, :min_h, :], p=2, dim=-1)
        es = F.normalize(basis_student[:, :min_h, :], p=2, dim=-1)
        return (1.0 - torch.sum(et * es, dim=-1)).mean()

    def forward(self, f_i: torch.Tensor, point_cloud: Optional[torch.Tensor] = None,
                return_all: bool = False, training: bool = True):
        basis_student = self.ssp(f_i)
        f_s, f_c = self.ssp.project_and_disentangle(f_i, basis_student)
        h_s, h_c, h = self.split_hash(f_s, f_c)
        b_i = torch.sign(h) + (h - h.detach())

        f_s_teacher = None
        f_c_teacher = None
        p_i = None
        kappa_i = None
        basis_teacher = None
        if training and point_cloud is not None:
            f_s_teacher, f_c_teacher, basis_teacher = self.prior_guided_disentanglement(f_i, point_cloud)
            kappa_i = self._estimate_kappa(point_cloud)
            p_i = self.generate_dynamic_proxy(f_s_teacher, point_cloud, kappa_i)

        if return_all:
            return {
                "b_i": b_i,
                "h_s": h_s,
                "h_c": h_c,
                "h_visual": h,
                "f_s": f_s,
                "f_c": f_c,
                "f_s_teacher": f_s_teacher,
                "f_c_teacher": f_c_teacher,
                "p_i": p_i,
                "kappa_i": kappa_i,
                "basis_teacher": basis_teacher,
                "basis_student": basis_student,
            }
        return b_i

    def compute_loss(self, f_i: torch.Tensor, point_cloud: torch.Tensor,
                     labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(f_i, point_cloud=point_cloud, return_all=True, training=True)
        h_s = out["h_s"]
        f_c_teacher = out["f_c_teacher"]
        p_i = out["p_i"]
        kappa_i = out["kappa_i"]
        basis_teacher = out["basis_teacher"]
        basis_student = out["basis_student"]

        loss_sem = self._sem_loss(h_s, p_i, kappa_i, labels)

        loss_context = torch.tensor(0.0, device=f_i.device)
        if f_c_teacher is not None:
            loss_context, _ = self.context_anchor(f_c_teacher)

        loss_distill = torch.tensor(0.0, device=f_i.device)
        if basis_teacher is not None and basis_student is not None:
            loss_distill = self._distill_loss(basis_teacher, basis_student)

        total = (
            float(self.lambda_sem) * loss_sem
            + float(self.lambda_context) * loss_context
            + float(self.lambda_distill) * loss_distill
        )
        losses = {
            "total_loss": total,
            "loss_sem": loss_sem,
            "loss_context": loss_context,
            "loss_distill": loss_distill,
            "kappa_i": kappa_i.mean() if isinstance(kappa_i, torch.Tensor) else torch.tensor(0.0, device=f_i.device),
        }
        return total, losses

    @torch.no_grad()
    def get_hash_codes(self, f_i: torch.Tensor) -> torch.Tensor:
        self.eval()
        basis_student = self.ssp(f_i)
        f_s, f_c = self.ssp.project_and_disentangle(f_i, basis_student)
        _, _, h = self.split_hash(f_s, f_c)
        return torch.sign(h)


__all__ = ["SCPHConfig", "SCPH"]
