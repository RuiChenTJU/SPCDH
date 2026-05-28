import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMetricReranker(nn.Module):
    """Intent-Calibrated Meta-Filtering (ICMF).

    Given z = [f^s; f^c] with the first sem_dim dimensions as the semantic
    stream and the remaining ctx_dim dimensions as the context stream, the
    module generates a query-specific low-rank Mahalanobis matrix M_q and
    optimises:

        L_ICMF = L_triplet + eta * L_reg
        L_reg  = lambda_c * ||M_cc||_F^2 + lambda_sc * ||M_sc||_F^2

    where M_cc is the context-context sub-block and M_sc is the
    semantic-context cross sub-block.
    """

    def __init__(
        self,
        feature_dim=1024,
        sem_dim=512,
        ctx_dim=512,
        rank=64,
        epsilon=1e-5,
        tau=0.5,
        lambda_c=0.1,
        lambda_sc=0.05,
        dist_scale=1.0,
        alpha_init=0.1,
        alpha_learnable=True,
        init_std=0.03,
        margin=0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.sem_dim = sem_dim
        self.ctx_dim = ctx_dim
        self.rank = rank
        self.epsilon = epsilon
        self.tau = tau
        self.lambda_c = lambda_c
        self.lambda_sc = lambda_sc
        self.dist_scale = float(dist_scale)
        self.margin = margin
        self.init_std = float(init_std)

        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)), persistent=False)

        self.meta_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim * rank),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.meta_network[-1]:
                    nn.init.normal_(m.weight, std=self.init_std)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def generate_metric_matrix(self, query_feature):
        batch_size = query_feature.shape[0]
        L_flat = self.meta_network(query_feature)
        L_q = L_flat.view(batch_size, self.feature_dim, self.rank)
        return L_q

    def compute_distance(self, query_feature, candidate_features, L_q):
        query_proj = torch.matmul(query_feature.unsqueeze(1), L_q).squeeze(1)
        candidate_proj = torch.matmul(candidate_features, L_q)
        diff = query_proj.unsqueeze(1) - candidate_proj
        distances = torch.sum(diff ** 2, dim=-1).float() * self.dist_scale
        similarities = torch.exp(-distances / (2 * self.tau ** 2))
        return distances, similarities

    def forward(self, query_feature, candidate_features, return_loss=False,
                positive_mask=None, negative_mask=None, use_residual=True):
        query_feature = F.normalize(query_feature, p=2, dim=-1)
        candidate_features = F.normalize(candidate_features, p=2, dim=-1)
        L_q = self.generate_metric_matrix(query_feature)
        distances, learned_similarities = self.compute_distance(
            query_feature, candidate_features, L_q
        )

        if use_residual:
            base_scores = torch.sum(query_feature.unsqueeze(1) * candidate_features, dim=-1)
            alpha = torch.clamp(self.alpha, 0.0, 1.0)
            final_similarities = base_scores + alpha * learned_similarities
        else:
            final_similarities = learned_similarities

        if return_loss:
            losses = self.compute_loss(
                query_feature, candidate_features, L_q,
                distances, final_similarities, positive_mask, negative_mask,
            )
            return final_similarities, losses
        return final_similarities

    def compute_loss(self, query_feature, candidate_features, L_q,
                     distances, final_similarities, positive_mask, negative_mask):
        losses = {}
        batch_size = query_feature.shape[0]

        # L_triplet: max(0, d_pos - d_neg + m)
        if positive_mask is not None and negative_mask is not None:
            triplet_losses = []
            d_all = distances.float()
            for b in range(batch_size):
                pos_indices = torch.where(positive_mask[b])[0]
                neg_indices = torch.where(negative_mask[b])[0]
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    d_b = d_all[b]
                    pos_dist = d_b[pos_indices].max()
                    neg_dist = d_b[neg_indices].min()
                    triplet_losses.append(F.relu(pos_dist - neg_dist + self.margin))
            if triplet_losses:
                losses["loss_triplet"] = torch.stack(triplet_losses).mean()
            else:
                losses["loss_triplet"] = torch.tensor(0.0, device=query_feature.device)
        else:
            losses["loss_triplet"] = torch.tensor(0.0, device=query_feature.device)

        # L_reg = lambda_c * ||M_cc||_F^2 + lambda_sc * ||M_sc||_F^2
        # z = [f^s; f^c], so [:sem_dim] = semantic, [sem_dim:] = context
        L_q32 = L_q.float()
        M_q = torch.matmul(L_q32, L_q32.transpose(1, 2))
        if self.sem_dim >= self.feature_dim:
            raise ValueError("SPCDH: sem_dim must be less than feature_dim")
        M_cc = M_q[:, self.sem_dim:, self.sem_dim:]   # context-context sub-block
        M_sc = M_q[:, :self.sem_dim, self.sem_dim:]   # semantic-context cross sub-block

        losses["loss_reg_cc"] = self.lambda_c * torch.mean(M_cc ** 2)
        losses["loss_reg_sc"] = self.lambda_sc * torch.mean(M_sc ** 2)
        losses["loss_reg"] = losses["loss_reg_cc"] + losses["loss_reg_sc"]
        return losses

    def rerank(self, query_feature, candidate_features, top_k=100, use_residual=True):
        with torch.no_grad():
            similarities = self.forward(
                query_feature, candidate_features,
                return_loss=False, use_residual=use_residual,
            ).squeeze(0)
            top_k = min(top_k, similarities.shape[0])
            top_similarities, top_indices = torch.topk(similarities, top_k, largest=True)
            return top_indices, top_similarities


ICMF = MetaMetricReranker

__all__ = ["ICMF", "MetaMetricReranker"]
