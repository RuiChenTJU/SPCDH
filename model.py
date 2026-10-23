import torch
from torch import nn
from torch.nn import functional as F
from core.PGMD import CAMA
from core.SCPH import SCPH
from core.ICMF import ICMF

def adaptive_weight(epoch):
    if epoch <= 8:
        return 0.0
    if epoch <= 18:
        return 0.4 * (epoch - 8) / 10.0
    return 0.4 + 0.6 * min(1.0, (epoch - 18) / 20.0)

@torch.no_grad()
def mine_triplets(metric, features, labels, factor):
    overlap = labels @ labels.T > 0
    diagonal = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positive = overlap & ~diagonal
    negative = ~overlap & ~diagonal
    valid = positive.any(1) & negative.any(1)
    distances = metric.compute_distance(features.detach(), features.detach()[None].expand(len(features), -1, -1), factor.detach())
    query = torch.where(valid)[0]
    pos = distances.masked_fill(~positive, -torch.inf).argmax(1)[query]
    neg = distances.masked_fill(~negative, torch.inf).argmin(1)[query]
    return torch.stack((query, pos, neg), dim=1)

class SPCDH(nn.Module):

    def __init__(self, bits, num_classes, *, fusion_params, hash_params, metric_params):
        super().__init__()
        self.cama = CAMA(num_classes=num_classes, **fusion_params)
        self.scph = SCPH(hash_bits=bits, num_classes=num_classes, **hash_params)
        self.icmf = ICMF(**metric_params)

    def forward(self, visual, audio, motion):
        out = self.scph(self.cama(visual, audio, motion), return_all=True, training=False)
        return (out['b_i'], torch.cat((out['f_s'], out['f_c']), dim=-1))

    def compute_loss(self, visual, audio, motion, labels, cloud, epoch):
        feature, parts = self.cama(visual, audio, motion, labels, return_loss=True)
        scph_loss, scph_parts, out = self.scph.compute_loss(feature, cloud, labels, lambda_adaptive=adaptive_weight(epoch), return_outputs=True)
        parts.update(scph_parts)
        total = parts['loss_align'] + parts['loss_cls'] + scph_loss
        if epoch > 20:
            features = torch.cat((out['f_s'], out['f_c']), dim=-1)
            factor = self.icmf.generate_metric_matrix(features)
            triplets = mine_triplets(self.icmf, features, labels, factor)
            if len(triplets):
                q, p, n = triplets.unbind(1)
                distances = self.icmf.compute_distance(features[q], torch.stack((features[p], features[n]), dim=1), factor[q])
                triplet_loss = F.relu(distances[:, 0] - distances[:, 1] + self.icmf.margin).mean()
            else:
                triplet_loss = factor.sum() * 0.0
            valid_queries = triplets[:, 0].unique()
            reg = self.icmf.regularization(factor[valid_queries])
            parts.update(loss_triplet=triplet_loss, loss_reg=reg)
            total = total + 0.3 * (triplet_loss + self.icmf.eta * reg)
            parts['triplets'] = total.new_tensor(len(triplets))
        return (total, parts)
