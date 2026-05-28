

from transformers import VideoMAEImageProcessor, VideoMAEModel
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warnings
from torch.nn import DataParallel
from multiprocessing import Process, Queue, cpu_count
import time
warnings.filterwarnings('ignore')


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class MotionFeatureExtractor:
    def __init__(self, model_name="MCG-NJU/videomae-base", device='cuda', cache_dir=None):
        print("SPCDH")
        if cache_dir:
            print("SPCDH")
        print("SPCDH")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = VideoMAEModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

        if torch.cuda.device_count() > 1:
            print("SPCDH")
            self.model = DataParallel(self.model)
            self.use_multi_gpu = True
        else:
            self.use_multi_gpu = False

        print("SPCDH")

    def extract_from_images_batch(self, image_paths_list):
        try:
            batch_size = len(image_paths_list)
            if batch_size == 0:
                return []

            batch_images = []
            valid_indices = []

            for idx, image_paths in enumerate(image_paths_list):
                if len(image_paths) != 16:
                    continue

                images = []
                valid = True
                for p in image_paths:
                    if not os.path.exists(p):
                        valid = False
                        break
                    images.append(Image.open(p).convert("RGB"))

                if valid:
                    batch_images.append(images)
                    valid_indices.append(idx)

            if len(batch_images) == 0:
                return [np.zeros((16, 768)) for _ in image_paths_list]

            batch_pixel_values = []

            for images in batch_images:
                inputs = self.processor(images, return_tensors="pt")

                if 'pixel_values' in inputs:
                    batch_pixel_values.append(inputs['pixel_values'])
                else:

                    batch_pixel_values.append(None)

            can_batch = all(pv is not None for pv in batch_pixel_values)

            if can_batch and len(batch_pixel_values) > 1:

                batch_pixel_values_tensor = torch.cat(batch_pixel_values, dim=0)
                batch_inputs = {'pixel_values': batch_pixel_values_tensor.to(self.device)}

                with torch.no_grad():
                    outputs = self.model(**batch_inputs)

                last_hidden_state = outputs.last_hidden_state

                batch_size = last_hidden_state.shape[0]
                feat_reshaped = last_hidden_state.view(batch_size, 16, -1, 768)
                feat_pooled = torch.mean(feat_reshaped, dim=2)

                batch_features = feat_pooled.cpu().numpy()
            else:

                batch_features = []
                for images in batch_images:
                    inputs = self.processor(images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    last_hidden_state = outputs.last_hidden_state
                    feat_reshaped = last_hidden_state.view(1, 16, -1, 768)
                    feat_pooled = torch.mean(feat_reshaped, dim=2)
                    batch_features.append(feat_pooled.squeeze(0).cpu().numpy())

                batch_features = np.array(batch_features)

            result = []
            result_idx = 0
            for i in range(len(image_paths_list)):
                if i in valid_indices:
                    result.append(batch_features[result_idx])
                    result_idx += 1
                else:
                    result.append(np.zeros((16, 768)))

            return result

        except Exception as e:
            print("SPCDH")
            import traceback
            traceback.print_exc()
            return [np.zeros((16, 768)) for _ in image_paths_list]

    def extract_from_images(self, image_paths):
        try:

            if len(image_paths) != 16:
                print("SPCDH")
                return np.zeros((16, 768))

            images = []
            for p in image_paths:
                if not os.path.exists(p):
                    print("SPCDH")
                    return np.zeros((16, 768))
                images.append(Image.open(p).convert("RGB"))

            inputs = self.processor(images, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            if self.use_multi_gpu:
                last_hidden_state = outputs.last_hidden_state
            else:
                last_hidden_state = outputs.last_hidden_state

            feat_reshaped = last_hidden_state.view(1, 16, -1, 768)

            feat_pooled = torch.mean(feat_reshaped, dim=2)

            return feat_pooled.squeeze(0).cpu().numpy()

        except Exception as e:
            print("SPCDH")
            import traceback
            traceback.print_exc()
            return np.zeros((16, 768))


def extract_motion_features_from_list(
    video_ids_json,
    image_root,
    output_path,
    device='cuda'
):

    print("=" * 60)
    print("SPCDH")
    with open(video_ids_json, 'r') as f:
        video_ids = json.load(f)
    print("SPCDH")

    print("=" * 60)
    print("SPCDH")
    cache_dir = "featureEncoder/videomae"
    os.makedirs(cache_dir, exist_ok=True)
    extractor = MotionFeatureExtractor(device=device, cache_dir=cache_dir)

    print("=" * 60)
    print("SPCDH")

    features_list = []
    missing_images = []
    failed_extraction = []

    image_dirs_map = {}
    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(image_root, subdir)
        if os.path.exists(subdir_path):
            for video_dir in os.listdir(subdir_path):
                video_dir_path = os.path.join(subdir_path, video_dir)
                if os.path.isdir(video_dir_path):
                    image_dirs_map[video_dir] = video_dir_path

    print("SPCDH")

    tasks = []
    task_indices = []

    for idx, video_id in enumerate(video_ids):
        if video_id not in image_dirs_map:
            missing_images.append(video_id)
            tasks.append(None)
            task_indices.append(idx)
            continue

        image_dir = image_dirs_map[video_id]

        image_files = []
        for i in range(16):
            img_path = os.path.join(image_dir, f"{i:04d}.jpg")
            if os.path.exists(img_path):
                image_files.append(img_path)

        if len(image_files) != 16:
            failed_extraction.append(video_id)
            tasks.append(None)
            task_indices.append(idx)
            continue

        tasks.append(image_files)
        task_indices.append(idx)

    print("SPCDH")
    print("SPCDH")
    print("SPCDH")

    if extractor.use_multi_gpu:
        batch_size = 8
        print("SPCDH")
    else:
        batch_size = 1
        print("SPCDH")

    valid_tasks = [(idx, task) for idx, task in zip(task_indices, tasks) if task is not None]

    pbar = tqdm(range(0, len(valid_tasks), batch_size), desc="SPCDH")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(valid_tasks))
        batch = valid_tasks[batch_start:batch_end]

        batch_image_paths = [task for _, task in batch]
        batch_indices = [idx for idx, _ in batch]

        if batch_size > 1 and extractor.use_multi_gpu:

            batch_features = extractor.extract_from_images_batch(batch_image_paths)
        else:

            batch_features = [extractor.extract_from_images(paths) for paths in batch_image_paths]

        for idx, feature in zip(batch_indices, batch_features):

            while len(features_list) <= idx:
                features_list.append(None)
            features_list[idx] = feature

        completed = sum(1 for f in features_list if f is not None)
        pbar.set_postfix({"SPCDH": completed, "SPCDH": len(video_ids)})

    for idx, task in zip(task_indices, tasks):
        if task is None:
            while len(features_list) <= idx:
                features_list.append(None)
            if features_list[idx] is None:
                if len(features_list) > 0:
                    valid_feat = next((f for f in features_list if f is not None), None)
                    if valid_feat is not None:
                        features_list[idx] = np.zeros_like(valid_feat)
                    else:
                        features_list[idx] = np.zeros((16, 768))
                else:
                    features_list[idx] = np.zeros((16, 768))

    if features_list[0] is None:
        valid_features = [f for f in features_list if f is not None]
        if len(valid_features) > 0:
            zero_sequence = np.zeros_like(valid_features[0])
            features_list = [zero_sequence if f is None else f for f in features_list]

    print("=" * 60)
    print("SPCDH")

    features_array = np.array(features_list, dtype=np.float32)
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, features_array)
    print("SPCDH")

    print("=" * 60)
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")

    if missing_images:
        print("SPCDH")
        for vid in missing_images[:10]:
            print(f"  - {vid}")
        if len(missing_images) > 10:
            print("SPCDH")

    if failed_extraction:
        print("SPCDH")
        for vid in failed_extraction[:10]:
            print(f"  - {vid}")
        if len(failed_extraction) > 10:
            print("SPCDH")


if __name__ == "__main__":

    VIDEO_IDS_JSON = "data/features/video_ids.json"
    IMAGE_ROOT = "/media/Backup/cr/dataset_hash/image"
    OUTPUT_PATH = "data/features/motion.npy"

    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print("SPCDH")
    else:
        DEVICE = 'cpu'
        print("SPCDH")

    print("=" * 60)
    print("SPCDH")
    print("=" * 60)
    print("SPCDH")
    print(f"  video_ids.json: {VIDEO_IDS_JSON}")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("=" * 60)

    if not os.path.exists(VIDEO_IDS_JSON):
        print("SPCDH")
        exit(1)

    if not os.path.exists(IMAGE_ROOT):
        print("SPCDH")
        exit(1)

    extract_motion_features_from_list(
        video_ids_json=VIDEO_IDS_JSON,
        image_root=IMAGE_ROOT,
        output_path=OUTPUT_PATH,
        device=DEVICE
    )

    print("=" * 60)
    print("SPCDH")
    print("=" * 60)
