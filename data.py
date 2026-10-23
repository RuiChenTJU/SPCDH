import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class MeituData:

    def __init__(self, root):
        root = Path(root)
        self.ids = [str(x) for x in read_json(root / 'features/video_ids.json')]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate feature IDs')
        records = {s: read_json(root / f'splits/{s}.json') for s in ('train', 'val', 'test')}
        self.classes = sorted({tag for row in records['train'] for tag in row['tags']})
        self.label_to_id = {tag: i for i, tag in enumerate(self.classes)}
        clouds = torch.load(root / 'semantic_embeddings.pt', map_location='cpu', weights_only=True)
        self.cloud_keys = sorted(clouds)
        self.cloud_index = {key: i for i, key in enumerate(self.cloud_keys)}
        self.clouds = torch.stack([clouds[key].float() for key in self.cloud_keys])
        if self.clouds.shape[1:] != (30, 512) or not torch.isfinite(self.clouds).all():
            raise ValueError('Expected finite 30 x 512 existing point clouds')
        if (self.clouds.norm(dim=-1) == 0).any():
            raise ValueError('Zero semantic point')
        self.splits = {}
        seen = set()
        for split, rows in records.items():
            by_id = {str(row['video']): row for row in rows}
            if len(by_id) != len(rows) or seen.intersection(by_id):
                raise ValueError('Duplicate or overlapping split IDs')
            seen.update(by_id)
            if set(by_id) - set(self.ids):
                raise ValueError(f'Missing feature IDs: {split}')
            indices, labels, cloud_indices = ([], [], [])
            for idx, video in enumerate(self.ids):
                if video not in by_id:
                    continue
                tags = by_id[video]['tags']
                key = ', '.join(sorted(tags))
                if not tags or len(set(tags)) != len(tags):
                    raise ValueError(f'Invalid tags: {video}')
                label = torch.zeros(len(self.classes), dtype=torch.float32)
                for tag in tags:
                    label[self.label_to_id[tag]] = 1
                indices.append(idx)
                labels.append(label)
                cloud_indices.append(self.cloud_index[key])
            self.splits[split] = (torch.tensor(indices), torch.stack(labels), torch.tensor(cloud_indices))
        if seen != set(self.ids):
            raise ValueError('Split union differs from feature IDs')
        self.features = []
        for name, dim in (('visual', 512), ('audio_768', 768), ('motion', 768)):
            array = np.load(root / f'features/{name}.npy', mmap_mode='r', allow_pickle=False)
            if array.shape != (len(self.ids), 16, dim) or array.dtype != np.float32:
                raise ValueError(f'Unexpected {name} feature format: {array.shape}, {array.dtype}')
            pooled = torch.empty(len(self.ids), dim)
            for start in range(0, len(array), 256):
                chunk = torch.from_numpy(np.array(array[start:start + 256], copy=True))
                if not torch.isfinite(chunk).all():
                    raise ValueError(f'Non-finite {name} features')
                pooled[start:start + len(chunk)] = chunk.mean(1)
            self.features.append(pooled)

    def dataset(self, split, limit=None):
        return MeituSplit(self, split, limit)

class MeituSplit(Dataset):

    def __init__(self, data, split, limit=None):
        self.data, self.training = (data, split == 'train')
        indices, labels, clouds = data.splits[split]
        count = len(indices) if limit is None else min(limit, len(indices))
        self.indices, self.labels, self.cloud_indices = (indices[:count], labels[:count], clouds[:count])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        values = tuple((feature[row] for feature in self.data.features))
        if self.training:
            return values + (self.labels[index], self.data.clouds[self.cloud_indices[index]])
        return values + (self.labels[index],)
