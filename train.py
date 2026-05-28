

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from models.SCPH import SCPHConfig, SCPH
from models.PGMD import CAMA
from models.ICMF import ICMF


class TwoStageDataset(Dataset):

    def __init__(
        self,
        audio_features,
        motion_features,
        visual_features,
        video_ids,
        labels_dict: Dict,
        semantic_embeddings: Dict,
        label_to_id: Dict = None,
    ):
        self.audio = audio_features
        self.motion = motion_features
        self.visual = visual_features
        self.video_ids = video_ids
        self.labels_dict = labels_dict
        self.semantic_embeddings = semantic_embeddings
        self.label_to_id = label_to_id

        self.valid_indices: List[int] = []
        self.tag_strings: List[str] = []
        self.multi_labels: List[torch.Tensor] = []

        for idx, video_id in enumerate(video_ids):
            if video_id not in labels_dict:
                continue
            tags = labels_dict[video_id]["tags"]
            tag_string = ", ".join(sorted(tags))
            if tag_string not in semantic_embeddings:
                continue

            self.valid_indices.append(idx)
            self.tag_strings.append(tag_string)

            if label_to_id is not None:
                multi_label = torch.zeros(len(label_to_id))
                for tag in tags:
                    if tag in label_to_id:
                        multi_label[label_to_id[tag]] = 1.0
                self.multi_labels.append(multi_label)

        print("SPCDH")
        if label_to_id is not None:
            print("SPCDH")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        audio = torch.from_numpy(self.audio[real_idx]).float()
        motion = torch.from_numpy(self.motion[real_idx]).float()
        visual = torch.from_numpy(self.visual[real_idx]).float()

        tag_string = self.tag_strings[idx]
        text_cloud = self.semantic_embeddings[tag_string].float()

        if self.multi_labels:
            multi_label = self.multi_labels[idx]
        else:
            multi_label = torch.zeros(63)

        return {
            "visual": visual,
            "audio": audio,
            "motion": motion,
            "text_cloud": text_cloud,
            "multi_label": multi_label,
        }


def load_features_and_labels(data_dir: str, json_dir: str, semantic_path: str = None):
    print("SPCDH")

    audio = np.load(os.path.join(data_dir, "audio_768.npy"))
    motion = np.load(os.path.join(data_dir, "motion.npy"))
    visual = np.load(os.path.join(data_dir, "visual.npy"))

    with open(os.path.join(data_dir, "video_ids.json"), "r") as f:
        video_ids = json.load(f)

    def load_json_labels(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        labels = {}
        for item in data:
            video_id = item["video"]
            labels[video_id] = {"tags": item["tags"], "indexs": item.get("indexs", [])}
        return labels

    train_labels = load_json_labels(os.path.join(json_dir, "train.json"))
    val_labels = load_json_labels(os.path.join(json_dir, "val.json"))
    test_labels = load_json_labels(os.path.join(json_dir, "test.json"))

    semantic_candidates = [semantic_path] if semantic_path else [
        os.path.join(os.path.dirname(__file__), "data", "prototypes", "semantic_embeddings.pt"),
    ]
    semantic_embeddings = None
    for candidate in semantic_candidates:
        if candidate and os.path.exists(candidate):
            semantic_embeddings = torch.load(candidate)
            print(f"Semantic prior: {candidate}")
            break
    if semantic_embeddings is None:
        raise FileNotFoundError(f"Semantic prior file not found: {semantic_candidates}")

    all_tags = set()
    for emb_key in semantic_embeddings.keys():
        tags = [tag.strip() for tag in emb_key.split(",")]
        all_tags.update(tags)
    label_to_id = {tag: idx for idx, tag in enumerate(sorted(all_tags))}

    print("SPCDH")

    return (
        audio,
        motion,
        visual,
        video_ids,
        train_labels,
        val_labels,
        test_labels,
        semantic_embeddings,
        label_to_id,
    )


def log_print(message: str, log_file: str):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _dcph_hamming_dist(q_h: np.ndarray, r_h: np.ndarray) -> np.ndarray:
    return (q_h[:, None, :] != r_h[None, :, :]).mean(axis=2).astype(np.float32)


def _dcph_cal_rel(q_l: np.ndarray, r_l: np.ndarray) -> np.ndarray:
    return (q_l @ r_l.T).astype(int)


def _dcph_mean_average_precision(database_hash, test_hash, database_labels, test_labels, K: int):
    query_num = test_hash.shape[0]
    K = int(min(K, database_hash.shape[0]))

    sim = np.dot(database_hash.astype(np.int32), test_hash.astype(np.int32).T)
    ids = np.argsort(-sim, axis=0)

    APx, Recall = [], []
    for i in range(query_num):
        label = test_labels[i, :].copy()
        if np.sum(label) == 0:
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = (np.sum(database_labels[idx[0:K], :] == label, axis=1) > 0)
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, K + 1, 1)

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        else:
            APx.append(0)

        all_relevant = (np.sum(database_labels == label, axis=1) > 0)
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float64(all_num) if all_num > 0 else 0.0
        Recall.append(r)

    return float(np.mean(np.array(APx))) if APx else 0.0, float(np.mean(np.array(Recall))) if Recall else 0.0, APx


def _dcph_average_cumulative_gain(Dist: np.ndarray, Rel: np.ndarray, K: int = 100) -> float:
    n, m = Dist.shape
    if (K < 0) or (K > m):
        K = m
    Rank = np.argsort(Dist, axis=1)
    _ACG = 0.0
    for g, rnk in zip(Rel, Rank):
        _ACG += g[rnk[:K]].mean()
    return float(_ACG / n) if n > 0 else 0.0


def _dcph_normalized_discounted_cumulative_gain(Dist: np.ndarray, Rel: np.ndarray, K: int = 100) -> float:
    n, m = Dist.shape
    if (K < 0) or (K > m):
        K = m
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(K))
    Rank = np.argsort(Dist, axis=1)
    _NDCG = 0.0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:K] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:K]] / D).sum()
            _NDCG += dcg / dcg_best
    return float(_NDCG / n) if n > 0 else 0.0


def _dcph_weighted_average_precision(Dist: np.ndarray, Rel: np.ndarray, K: int = 100) -> float:
    n, m = Dist.shape
    if (K < 0) or (K > m):
        K = m
    Gain = Rel
    S = (Gain > 0).astype(int)
    pos = np.arange(K) + 1
    Rank = np.argsort(Dist, axis=1)
    _WAP = 0.0
    for s, g, rnk in zip(S, Gain, Rank):
        _rnk = rnk[:K]
        s_k, g_k = s[_rnk], g[_rnk]
        n_rel = s_k.sum()
        if n_rel > 0:
            acg = np.cumsum(g_k) / pos
            _WAP += (acg * s_k).sum() / n_rel
    return float(_WAP / n) if n > 0 else 0.0


def _dcph_map_from_ranked_indices(db_labels: np.ndarray, q_labels: np.ndarray, ranked_indices: List[np.ndarray], K: int):
    APx, Recall = [], []
    query_num = q_labels.shape[0]
    for i in range(query_num):
        label = q_labels[i, :].copy()
        if np.sum(label) == 0:
            continue
        label[label == 0] = -1
        idx = ranked_indices[i][:K]

        imatch = (np.sum(db_labels[idx, :] == label, axis=1) > 0)
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, len(idx) + 1, 1)

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        else:
            APx.append(0)

        all_relevant = (np.sum(db_labels == label, axis=1) > 0)
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float64(all_num) if all_num > 0 else 0.0
        Recall.append(r)

    return float(np.mean(np.array(APx))) if APx else 0.0, float(np.mean(np.array(Recall))) if Recall else 0.0


def _dcph_metrics_from_ranked_indices(
    Dist: np.ndarray, Rel: np.ndarray, ranked_indices: List[np.ndarray], K: int
) -> Tuple[float, float, float]:
    n_query = Dist.shape[0]
    K = int(min(K, max(len(idx) for idx in ranked_indices) if ranked_indices else 0))

    acg_list, ndcg_list, wap_list = [], [], []

    for i in range(n_query):
        if i >= len(ranked_indices) or len(ranked_indices[i]) == 0:
            continue

        idx = ranked_indices[i][:K]
        dist_i = Dist[i, idx]
        rel_i = Rel[i, idx]

        acg_list.append(rel_i.mean())

        rel_all = Rel[i, :]
        G_all = 2 ** rel_all - 1
        G = 2 ** rel_i - 1
        D = np.log2(2 + np.arange(len(G)))
        dcg = (G / D).sum()
        dcg_best = (np.sort(G_all)[::-1][:K] / D).sum()
        if dcg_best > 0:
            ndcg_list.append(dcg / dcg_best)
        else:
            ndcg_list.append(0.0)

        Gain = rel_i
        S = (Gain > 0).astype(int)
        pos = np.arange(len(Gain)) + 1
        n_rel = S.sum()
        if n_rel > 0:
            acg_cum = np.cumsum(Gain) / pos
            wap_list.append((acg_cum * S).sum() / n_rel)
        else:
            wap_list.append(0.0)

    acg = float(np.mean(acg_list)) if acg_list else 0.0
    ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    wap = float(np.mean(wap_list)) if wap_list else 0.0

    return acg, ndcg, wap


def train_epoch(
    cama,
    pgmd_scph,
    icmf,
    dataloader,
    optimizer,
    device,
    epoch: int,
    log_file: str,
    total_epochs: int,
    use_amp: bool = True,
    scaler=None,
    grad_clip_max_norm: float = 20.0,
    sem_warmup_epochs: int = 8,
    sem_ramp_epochs: int = 10,
    rerank_warmup_epochs: int = 10,
    rerank_weight: float = 0.1,
    rerank_use_residual: bool = False,
):
    cama.train()
    pgmd_scph.train()
    icmf.train() if epoch > int(rerank_warmup_epochs) else icmf.eval()

    hash_bits = int(pgmd_scph.cfg.hash_bits)

    if hasattr(pgmd_scph, "set_gate_shift") and hasattr(pgmd_scph, "set_gate_scale"):
        if hash_bits <= 16:

            if epoch <= sem_warmup_epochs:
                pgmd_scph.set_gate_shift(0.50)
                pgmd_scph.set_gate_scale(5.0)
            elif epoch <= sem_warmup_epochs + sem_ramp_epochs:
                ramp_progress = (epoch - sem_warmup_epochs) / float(sem_ramp_epochs)
                pgmd_scph.set_gate_shift(0.50 - 0.05 * ramp_progress)
                pgmd_scph.set_gate_scale(5.0 + 3.0 * ramp_progress)
            else:

                pgmd_scph.set_gate_shift(0.45)
                pgmd_scph.set_gate_scale(8.0)
        elif hash_bits <= 32:

            if epoch <= sem_warmup_epochs:
                pgmd_scph.set_gate_shift(0.45)
                pgmd_scph.set_gate_scale(6.0)
            elif epoch <= sem_warmup_epochs + sem_ramp_epochs:
                ramp_progress = (epoch - sem_warmup_epochs) / float(sem_ramp_epochs)
                pgmd_scph.set_gate_shift(0.45 - 0.08 * ramp_progress)
                pgmd_scph.set_gate_scale(6.0 + 3.0 * ramp_progress)
            else:

                pgmd_scph.set_gate_shift(0.37)
                pgmd_scph.set_gate_scale(9.0)
        else:

            if epoch <= sem_warmup_epochs:
                pgmd_scph.set_gate_shift(0.40)
                pgmd_scph.set_gate_scale(6.0)
            elif epoch <= sem_warmup_epochs + sem_ramp_epochs:
                ramp_progress = (epoch - sem_warmup_epochs) / float(sem_ramp_epochs)
                pgmd_scph.set_gate_shift(0.40 - 0.10 * ramp_progress)
                pgmd_scph.set_gate_scale(6.0 + 4.0 * ramp_progress)
            else:
                pgmd_scph.set_gate_shift(0.30)
                pgmd_scph.set_gate_scale(12.0)

    if hasattr(pgmd_scph, "set_neg_margin_zeta"):
        if hash_bits <= 16:
            zeta = 0.15
        elif hash_bits <= 32:
            zeta = 0.18
        else:
            zeta = 0.20
        pgmd_scph.set_neg_margin_zeta(zeta)

    enable_rerank = epoch > int(rerank_warmup_epochs)

    total_loss = 0.0
    loss_components: Dict[str, float] = {}

    kappa_samples = []

    pbar = tqdm(dataloader, desc="SPCDH")
    for batch_idx, batch in enumerate(pbar):
        visual = batch["visual"].to(device)
        audio = batch["audio"].to(device)
        motion = batch["motion"].to(device)
        text_cloud = batch["text_cloud"].to(device)
        multi_label = batch["multi_label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        fused_features, cama_losses = cama(
            visual, audio, motion, labels=multi_label, return_loss=True)

        if text_cloud.dim() == 2:
            point_cloud = text_cloud.unsqueeze(0).expand(fused_features.shape[0], -1, -1)
        else:
            point_cloud = text_cloud

        is_256 = (hash_bits >= 256)
        if pgmd_scph.training and epoch > 30 and not is_256:

            noise_scale = 0.02 * min(1.0, (epoch - 30) / 30.0)
            noise = torch.randn_like(point_cloud) * noise_scale
            point_cloud = point_cloud + noise
            point_cloud = torch.nn.functional.normalize(point_cloud, p=2, dim=-1)

        loss2, pgmd_scph_losses = pgmd_scph.compute_loss(
            fused_features, point_cloud, labels=multi_label)

        loss3_triplet = torch.tensor(0.0, device=device)
        loss3_reg = torch.tensor(0.0, device=device)

        if enable_rerank:

            out2 = pgmd_scph.forward(fused_features, point_cloud=None,
                                     return_all=True, training=False)
            f_s = out2["f_s"]
            f_c = out2["f_c"]

            B = int(f_s.shape[0])

            if B >= 4:
                candidate_features = torch.cat([f_s, f_c], dim=-1)

                triplet_list, reg_list = [], []
                for i in range(B):

                    candidate_mask = torch.ones(B, dtype=torch.bool, device=device)
                    candidate_mask[i] = False
                    candidates = candidate_features[candidate_mask].unsqueeze(0)
                    query_feature = candidate_features[i: i + 1]

                    labels_i = (multi_label[i] > 0.5)
                    labels_cand = (multi_label[candidate_mask] > 0.5)
                    overlap = (labels_cand & labels_i.unsqueeze(0)).sum(dim=1)
                    positive_mask_1d = overlap > 0
                    negative_mask_1d = overlap == 0

                    if not (positive_mask_1d.any() and negative_mask_1d.any()):
                        continue

                    positive_mask = positive_mask_1d.unsqueeze(0)
                    negative_mask = negative_mask_1d.unsqueeze(0)

                    _, losses3 = icmf(
                        query_feature,
                        candidates,
                        return_loss=True,
                        positive_mask=positive_mask,
                        negative_mask=negative_mask,
                        use_residual=bool(rerank_use_residual),
                    )
                    triplet_list.append(losses3.get(
                        "loss_triplet", torch.tensor(0.0, device=device)))
                    reg_list.append(losses3.get("loss_reg", torch.tensor(0.0, device=device)))

                if triplet_list:
                    loss3_triplet = torch.stack(triplet_list).mean()
                if reg_list:
                    loss3_reg = torch.stack(reg_list).mean()

        loss = 0.0

        for k, v in cama_losses.items():
            loss = loss + v
            loss_components[f"cama_{k}"] = loss_components.get(
                f"cama_{k}", 0.0) + float(v.item())

        loss = loss + loss2
        loss_components["scph_total_loss"] = loss_components.get(
            "scph_total_loss", 0.0) + float(loss2.item())
        for k, v in pgmd_scph_losses.items():
            loss_components[f"scph_{k}"] = loss_components.get(
                f"scph_{k}", 0.0) + float(v.item())

        if "kappa_i" in pgmd_scph_losses:
            kappa_val = pgmd_scph_losses["kappa_i"]
            if isinstance(kappa_val, torch.Tensor):
                kappa_samples.append(float(kappa_val.item()))
            else:
                kappa_samples.append(float(kappa_val))

        if enable_rerank:
            loss = loss + float(rerank_weight) * (loss3_triplet + loss3_reg)
            loss_components["icmf_loss_triplet"] = loss_components.get(
                "icmf_loss_triplet", 0.0) + float(loss3_triplet.item())
            loss_components["icmf_loss_reg"] = loss_components.get(
                "icmf_loss_reg", 0.0) + float(loss3_reg.item())

            try:
                alpha_val = icmf.alpha
                if isinstance(alpha_val, torch.Tensor):
                    alpha_val = float(torch.clamp(alpha_val.detach(), 0.0, 1.0).cpu().item())
                else:
                    alpha_val = float(alpha_val)
                loss_components["icmf_alpha"] = loss_components.get(
                    "icmf_alpha", 0.0) + alpha_val
            except Exception:
                pass
        else:
            loss_components["icmf_loss_triplet"] = loss_components.get(
                "icmf_loss_triplet", 0.0) + 0.0
            loss_components["icmf_loss_reg"] = loss_components.get("icmf_loss_reg", 0.0) + 0.0

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(cama.parameters()) + list(pgmd_scph.parameters()) +
                (list(icmf.parameters()) if enable_rerank else []),
                max_norm=float(grad_clip_max_norm),
            )
            scaler.step(optimizer)
            scaler.update()

            if enable_rerank and hasattr(icmf, "alpha"):
                try:
                    with torch.no_grad():
                        if isinstance(icmf.alpha, torch.Tensor):
                            icmf.alpha.clamp_(0.0, 1.0)
                except Exception:
                    pass
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cama.parameters()) + list(pgmd_scph.parameters()) +
                (list(icmf.parameters()) if enable_rerank else []),
                max_norm=float(grad_clip_max_norm),
            )
            optimizer.step()

            if enable_rerank and hasattr(icmf, "alpha"):
                try:
                    with torch.no_grad():
                        if isinstance(icmf.alpha, torch.Tensor):
                            icmf.alpha.clamp_(0.0, 1.0)
                except Exception:
                    pass

        total_loss += float(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(1, len(dataloader))
    for k in list(loss_components.keys()):
        loss_components[k] /= max(1, len(dataloader))

    if hasattr(pgmd_scph, "set_kappa_max") and kappa_samples and epoch % 5 == 0:
        kappa_mean = np.mean(kappa_samples)
        kappa_std = np.std(kappa_samples)
        kappa_p95 = np.percentile(kappa_samples, 95)

        suggested_kappa_max = max(kappa_p95 * 1.2, kappa_mean * 2.0)
        suggested_kappa_max = np.clip(suggested_kappa_max, 100.0, 250.0)

        current_kappa_max = pgmd_scph.cfg.kappa_max

        new_kappa_max = 0.7 * current_kappa_max + 0.3 * suggested_kappa_max
        pgmd_scph.set_kappa_max(float(new_kappa_max))

        if epoch % 10 == 0:
            log_print("SPCDH"
                      f"kappa_max: {current_kappa_max:.1f} → {new_kappa_max:.1f}", log_file)

    return avg_loss, loss_components


@torch.no_grad()
def evaluate_hash_and_rerank(
    cama,
    pgmd_scph,
    icmf,
    query_loader,
    database_loader,
    device,
    epoch: int,
    topK: int = 100,
    recall_topR: int = 200,
    use_rerank: bool = True,
    rerank_use_residual: bool = False,
):
    cama.eval()
    pgmd_scph.eval()
    icmf.eval()

    def _collect(loader, desc: str):
        all_b, all_y, all_feat = [], [], []
        for batch in tqdm(loader, desc=desc):
            visual = batch["visual"].to(device)
            audio = batch["audio"].to(device)
            motion = batch["motion"].to(device)
            multi_labels = batch["multi_label"].to(device)

            fused_features = cama(visual, audio, motion, return_loss=False)
            out2 = pgmd_scph.forward(fused_features, point_cloud=None,
                                     return_all=True, training=False)
            h_visual = out2["h_visual"]
            b = torch.sign(h_visual).to(torch.int8).cpu().numpy()
            y = multi_labels.to(torch.int8).cpu().numpy()

            f_s = out2["f_s"]
            f_c = out2["f_c"]
            feat = torch.cat([f_s, f_c], dim=-1).float().cpu().numpy()

            all_b.append(b)
            all_y.append(y)
            all_feat.append(feat)

        b_all = np.concatenate(all_b, axis=0) if all_b else np.zeros((0, 1), dtype=np.int8)
        y_all = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, 1), dtype=np.int8)
        f_all = np.concatenate(all_feat, axis=0) if all_feat else np.zeros((0, 1), dtype=np.float32)
        return b_all, y_all, f_all

    db_b, db_y, db_f = _collect(database_loader, desc=f"Eval-DB Epoch {epoch}")
    q_b, q_y, q_f = _collect(query_loader, desc=f"Eval-Q Epoch {epoch}")

    Dist = _dcph_hamming_dist(q_b, db_b)
    Rel = _dcph_cal_rel(q_y, db_y)

    K100 = int(min(100, db_b.shape[0] if db_b.ndim == 2 else 0))
    mAP_100, recall_100, _ = _dcph_mean_average_precision(db_b, q_b, db_y, q_y, K100)
    acg_100 = _dcph_average_cumulative_gain(Dist, Rel, K100)
    ndcg_100 = _dcph_normalized_discounted_cumulative_gain(Dist, Rel, K100)
    wap_100 = _dcph_weighted_average_precision(Dist, Rel, K100)

    metrics_hash_100 = {
        "mAP": float(mAP_100),
        "Recall@100": float(recall_100),
        "ACG@100": float(acg_100),
        "NDCG@100": float(ndcg_100),
        "WAP@100": float(wap_100),
    }

    K200 = int(min(int(recall_topR), db_b.shape[0] if db_b.ndim == 2 else 0))
    mAP_200, recall_200, _ = _dcph_mean_average_precision(db_b, q_b, db_y, q_y, K200)
    acg_200 = _dcph_average_cumulative_gain(Dist, Rel, K200)
    ndcg_200 = _dcph_normalized_discounted_cumulative_gain(Dist, Rel, K200)
    wap_200 = _dcph_weighted_average_precision(Dist, Rel, K200)

    metrics_hash_200 = {
        "mAP": float(mAP_200),
        "Recall@200": float(recall_200),
        "ACG@200": float(acg_200),
        "NDCG@200": float(ndcg_200),
        "WAP@200": float(wap_200),
    }

    if (not use_rerank) or (db_f.shape[0] == 0) or (q_f.shape[0] == 0):
        return metrics_hash_100, metrics_hash_200, None

    R = int(min(int(recall_topR), db_b.shape[0]))
    K = int(min(int(topK), R))

    sim = np.dot(db_b.astype(np.int32), q_b.astype(np.int32).T)
    ids = np.argsort(-sim, axis=0)

    ranked_indices: List[np.ndarray] = []
    for qi in tqdm(range(q_b.shape[0]), desc=f"Rerank Epoch {epoch} (topR={R}→topK={K})"):
        topR_idx = ids[:R, qi]
        q_feat = torch.from_numpy(q_f[qi: qi + 1]).to(device).float()
        cand_feat = torch.from_numpy(db_f[topR_idx]).to(device).float().unsqueeze(0)
        topk_in_cand, _ = icmf.rerank(q_feat, cand_feat, top_k=K,
                                      use_residual=bool(rerank_use_residual))
        final_idx = topR_idx[topk_in_cand.cpu().numpy()]
        ranked_indices.append(final_idx)

    rerank_map, rerank_recall = _dcph_map_from_ranked_indices(db_y, q_y, ranked_indices, K=K)
    rerank_acg, rerank_ndcg, rerank_wap = _dcph_metrics_from_ranked_indices(
        Dist, Rel, ranked_indices, K=K)
    metrics_reranked = {
        "mAP": float(rerank_map),
        "Recall@100": float(rerank_recall),
        "ACG@100": float(rerank_acg),
        "NDCG@100": float(rerank_ndcg),
        "WAP@100": float(rerank_wap),
    }

    return metrics_hash_100, metrics_hash_200, metrics_reranked


def main():
    parser = argparse.ArgumentParser(description="SPCDH")
    parser.add_argument("--hash_bits", type=int, default=64, choices=[16, 32, 64, 128, 256])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--topK", type=int, default=100)
    parser.add_argument("--recall_topR", type=int, default=200)
    parser.add_argument("--sem_warmup_epochs", type=int, default=8)
    parser.add_argument("--sem_ramp_epochs", type=int, default=10, help="SPCDH")
    parser.add_argument("--rerank_warmup_epochs", type=int, default=20, help="SPCDH")
    parser.add_argument("--rerank_weight", type=float, default=0.3)
    parser.add_argument("--rerank_use_residual", type=int, default=0,
                        help="0=disable cosine residual, 1=enable")
    parser.add_argument("--grad_clip", type=float, default=20.0)
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluation interval in epochs")
    parser.add_argument("--data_dir", type=str, default="data/features")
    parser.add_argument("--json_dir", type=str, default="../dataset_hash/json")
    parser.add_argument("--semantic_path", type=str,
                        default="data/prototypes/semantic_embeddings.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/stage123")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("SPCDH")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.output_dir
    workdir = os.path.join(base_dir, f"{args.hash_bits}bits_sem_context_rerank_{timestamp}")
    os.makedirs(workdir, exist_ok=True)
    log_file = os.path.join(workdir, "training.log")

    log_print("SPCDH", log_file)
    log_print("SPCDH", log_file)

    log_print("\n" + "=" * 70, log_file)
    log_print("SPCDH", log_file)
    log_print("=" * 70, log_file)

    if args.hash_bits <= 16:
        log_print("SPCDH", log_file)
        log_print("", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print(f"  - Static Proxy: K=4, margin=0.15", log_file)
        log_print("", log_file)
    elif args.hash_bits <= 32:
        log_print("SPCDH", log_file)
        log_print("", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print(f"  - Static Proxy: K=6, margin=0.18", log_file)
        log_print("", log_file)
    else:
        is_256 = (args.hash_bits >= 256)
        if is_256:
            log_print("SPCDH", log_file)
            log_print("", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print("SPCDH", log_file)
        log_print(f"  - Static Proxy: K=8, margin=0.20", log_file)
        if is_256:
            log_print("", log_file)
            log_print("SPCDH", log_file)
            log_print("SPCDH", log_file)
        log_print("", log_file)

    log_print("SPCDH", log_file)
    log_print("SPCDH", log_file)
    log_print("SPCDH", log_file)
    log_print("SPCDH", log_file)
    log_print("=" * 70 + "\n", log_file)

    data_dir = args.data_dir
    json_dir = args.json_dir
    (
        audio,
        motion,
        visual,
        video_ids,
        train_labels,
        val_labels,
        test_labels,
        semantic_embeddings,
        label_to_id,
    ) = load_features_and_labels(data_dir, json_dir, args.semantic_path)

    train_dataset = TwoStageDataset(audio, motion, visual, video_ids,
                                    train_labels, semantic_embeddings, label_to_id)
    val_dataset = TwoStageDataset(audio, motion, visual, video_ids,
                                  val_labels, semantic_embeddings, label_to_id)
    test_dataset = TwoStageDataset(audio, motion, visual, video_ids,
                                   test_labels, semantic_embeddings, label_to_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    cama = CAMA(
        visual_dim=512,
        audio_dim=768,
        motion_dim=768,
        unified_dim=512,
        num_classes=len(label_to_id),
        tau_assign=0.07,
        lambda_align=1.0,
        lambda_cls=0.5,
    ).to(device)

    semantic_bits = None
    context_bits = None
    if int(args.hash_bits) == 16:
        semantic_bits = 12
        context_bits = 4
    elif int(args.hash_bits) == 32:
        semantic_bits = 24
        context_bits = 8

    cfg2 = SCPHConfig(
        input_dim=512,
        hash_bits=int(args.hash_bits),
        hidden_dim=256,
        num_classes=len(label_to_id),
        semantic_bits=semantic_bits,
        context_bits=context_bits,
    )
    pgmd_scph = SCPH(cfg2).to(device)

    icmf = ICMF(
        feature_dim=1024,
        sem_dim=512,
        ctx_dim=512,
        rank=64,
        tau=0.5,
        lambda_c=0.1,
        lambda_sc=0.05,
        margin=0.2,
        alpha_init=0.5,
    ).to(device)

    log_print("SPCDH", log_file)
    log_print("SPCDH", log_file)
    log_print("SPCDH", log_file)
    if semantic_bits is not None and context_bits is not None:
        log_print("SPCDH", log_file)
    if int(args.hash_bits) == 16:
        log_print("SPCDH"
                  f"λ_sem={cfg2.lambda_sem}, λ_context={cfg2.lambda_context}, λ_distill={cfg2.lambda_distill}", log_file)

    optimizer = optim.AdamW(
        list(cama.parameters()) + list(pgmd_scph.parameters()) + list(icmf.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=1e-5)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, int(args.epochs) + 1):
        log_print(f"\n{'='*70}", log_file)
        log_print(f"Epoch {epoch}/{args.epochs}", log_file)

        avg_loss, loss_components = train_epoch(
            cama=cama,
            pgmd_scph=pgmd_scph,
            icmf=icmf,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_file=log_file,
            total_epochs=int(args.epochs),
            use_amp=use_amp,
            scaler=scaler,
            grad_clip_max_norm=float(args.grad_clip),
            sem_warmup_epochs=int(args.sem_warmup_epochs),
            sem_ramp_epochs=int(args.sem_ramp_epochs),
            rerank_warmup_epochs=int(args.rerank_warmup_epochs),
            rerank_weight=float(args.rerank_weight),
            rerank_use_residual=bool(int(args.rerank_use_residual)),
        )

        log_print("SPCDH", log_file)
        for k, v in sorted(loss_components.items()):
            log_print(f"  {k}: {v:.6f}", log_file)

        if (epoch % int(args.eval_every) == 0) or (epoch == int(args.epochs)):
            metrics_hash_100, metrics_hash_200, metrics_reranked = evaluate_hash_and_rerank(
                cama=cama,
                pgmd_scph=pgmd_scph,
                icmf=icmf,
                query_loader=test_loader,
                database_loader=val_loader,
                device=device,
                epoch=epoch,
                topK=int(args.topK),
                recall_topR=int(args.recall_topR),
                use_rerank=True,
                rerank_use_residual=bool(int(args.rerank_use_residual)),
            )

            log_print("SPCDH", log_file)
            for key, val in metrics_hash_100.items():
                log_print(f"  {key}: {val:.4f}", log_file)

            log_print("SPCDH", log_file)
            for key, val in metrics_hash_200.items():
                log_print(f"  {key}: {val:.4f}", log_file)

            if metrics_reranked is not None:
                log_print("SPCDH", log_file)
                for key, val in metrics_reranked.items():
                    log_print(f"  {key}: {val:.4f}", log_file)

                score = float(metrics_reranked.get("mAP", 0.0))
            else:

                score = float(metrics_hash_100.get("mAP", 0.0))

            if score > best_score:
                best_score = score
                best_epoch = epoch
                ckpt_path = os.path.join(workdir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "cama_state_dict": cama.state_dict(),
                        "scph_state_dict": pgmd_scph.state_dict(),
                        "icmf_state_dict": icmf.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_score": best_score,
                        "metrics_hash_100": metrics_hash_100,
                        "metrics_hash_200": metrics_hash_200,
                        "metrics_reranked": metrics_reranked,
                        "cfg2": cfg2.__dict__,
                    },
                    ckpt_path,
                )
                log_print(
                    f"✓ New best score={best_score:.4f} @ epoch {epoch}, saved: {ckpt_path}", log_file)

        scheduler.step()
        log_print("SPCDH", log_file)

    log_print("SPCDH", log_file)


if __name__ == "__main__":
    main()
