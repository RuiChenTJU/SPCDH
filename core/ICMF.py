import torch
from torch import nn
import torch.nn.functional as F

class MetaMetricReranker(nn.Module):

    def __init__(self, feature_dim, sem_dim, ctx_dim, rank, hidden_dim, epsilon, lambda_c, lambda_sc, eta, margin):
        super().__init__()
        if feature_dim != sem_dim + ctx_dim or min(sem_dim, ctx_dim, rank, hidden_dim) < 1:
            raise ValueError('Invalid metric dimensions')
        if epsilon <= 0 or min(lambda_c, lambda_sc, eta, margin) < 0:
            raise ValueError('Invalid metric stabilizer or loss weights')
        self.feature_dim, self.sem_dim, self.ctx_dim, self.rank = (feature_dim, sem_dim, ctx_dim, rank)
        self.epsilon, self.lambda_c, self.lambda_sc = (epsilon, lambda_c, lambda_sc)
        self.eta, self.margin = (eta, margin)
        self.meta_network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, feature_dim * rank))
        for layer in self.meta_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.meta_network[-1].weight, std=0.03)

    def generate_metric_matrix(self, query_feature):
        return self.meta_network(query_feature).reshape(-1, self.feature_dim, self.rank)

    def compute_distance(self, query_feature, candidate_features, factor):
        delta = query_feature.unsqueeze(1) - candidate_features
        projected = torch.bmm(delta, factor)
        augmented = torch.cat([projected, self.epsilon ** 0.5 * delta], dim=-1)
        return torch.linalg.vector_norm(augmented, dim=-1)

    def regularization(self, factor):
        if not len(factor):
            return factor.sum() * 0.0
        semantic, context = (factor[:, :self.sem_dim], factor[:, self.sem_dim:])
        gram_s, gram_c = (semantic.transpose(1, 2) @ semantic, context.transpose(1, 2) @ context)
        context_norm = gram_c.square().sum((1, 2)) + 2 * self.epsilon * context.square().sum((1, 2)) + self.ctx_dim * self.epsilon ** 2
        cross_norm = (gram_s * gram_c).sum((1, 2))
        return self.lambda_c * context_norm.mean() / self.ctx_dim ** 2 + self.lambda_sc * cross_norm.mean() / (self.sem_dim * self.ctx_dim)

    def compute_loss(self, features, triplets):
        if triplets.ndim != 2 or triplets.shape[1] != 3 or triplets.dtype != torch.long:
            raise ValueError('triplets must be a LongTensor of shape (T, 3)')
        if len(triplets) and (triplets.min() < 0 or triplets.max() >= len(features)):
            raise ValueError('Triplet index outside the feature batch')
        factor = self.generate_metric_matrix(features)
        if len(triplets):
            query, positive, negative = triplets.unbind(dim=1)
            candidates = torch.stack([features[positive], features[negative]], dim=1)
            distances = self.compute_distance(features[query], candidates, factor[query])
            triplet = F.relu(distances[:, 0] - distances[:, 1] + self.margin).mean()
        else:
            triplet = factor.sum() * 0.0
        valid_queries = triplets[:, 0].unique()
        regularizer = self.regularization(factor[valid_queries])
        return {'loss_triplet': triplet, 'loss_reg': regularizer, 'loss_total': triplet + self.eta * regularizer}

    def forward(self, query_feature, candidate_features):
        factor = self.generate_metric_matrix(query_feature)
        distances = self.compute_distance(query_feature, candidate_features, factor)
        return distances

    @torch.no_grad()
    def rerank(self, query_feature, candidate_features, top_k=100):
        distances = self(query_feature, candidate_features).squeeze(0)
        values, indices = distances.topk(min(top_k, len(distances)), largest=False)
        return (indices, values)
ICMF = MetaMetricReranker
