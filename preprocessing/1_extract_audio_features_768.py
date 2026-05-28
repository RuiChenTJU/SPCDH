

import os
import sys
import json
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


beats_dir = 'featureEncoder/BEATS'
sys.path.insert(0, beats_dir)


try:
    from BEATs import BEATs, BEATsConfig
except ImportError as e:
    print("SPCDH")
    print("SPCDH")
    import traceback
    traceback.print_exc()
    exit(1)


def load_beats_model(checkpoint_path, device='cuda'):
    print("SPCDH")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    print("SPCDH")
    print(f"  Encoder embed dim: {cfg.encoder_embed_dim}")
    print(f"  Fine-tuned model: {cfg.finetuned_model}")
    return model


def extract_audio_feature_768(model, audio_path, device='cuda', target_sr=16000, target_time_steps=16):
    try:

        waveform, sr = torchaudio.load(audio_path)

        if os.path.getsize(audio_path) == 0:
            return None

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)

        waveform = waveform.unsqueeze(0)

        waveform = waveform.to(device)

        with torch.no_grad():

            fbank = model.preprocess(waveform)
            fbank = fbank.unsqueeze(1)
            features = model.patch_embedding(fbank)
            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.transpose(1, 2)
            features = model.layer_norm(features)

            if model.post_extract_proj is not None:
                features = model.post_extract_proj(features)

            x = model.dropout_input(features)

            x, layer_results = model.encoder(x, padding_mask=None)

            if len(x.shape) == 3:
                features = x[0]

            time_steps, feature_dim = features.shape
            features = features.transpose(0, 1)

            features = features.unsqueeze(0)

            adaptive_pool = torch.nn.AdaptiveAvgPool1d(target_time_steps)
            features = adaptive_pool(features)

            features = features.squeeze(0).transpose(0, 1)

            feature_sequence = features.cpu().numpy()

        return feature_sequence

    except Exception as e:
        print("SPCDH")
        print("SPCDH")
        return None


def extract_audio_features_from_list(
    video_ids_json,
    audio_root,
    output_path,
    beats_checkpoint,
    device='cuda'
):

    print("=" * 60)
    print("SPCDH")
    with open(video_ids_json, 'r') as f:
        video_ids = json.load(f)
    print("SPCDH")

    print("=" * 60)
    print("SPCDH")
    model = load_beats_model(beats_checkpoint, device=device)

    print("=" * 60)
    print("SPCDH")

    features_list = []
    missing_audio = []
    failed_extraction = []

    audio_files_map = {}
    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(audio_root, subdir)
        if os.path.exists(subdir_path):
            for audio_file in os.listdir(subdir_path):
                if audio_file.endswith('.mp3'):
                    video_id = audio_file.replace('.mp3', '')
                    audio_files_map[video_id] = os.path.join(subdir_path, audio_file)

    print("SPCDH")
    print("SPCDH")

    for idx, video_id in enumerate(tqdm(video_ids, desc="SPCDH")):

        if video_id not in audio_files_map:
            missing_audio.append(video_id)

            if len(features_list) > 0:
                feature_vector = np.zeros_like(features_list[0])
            else:

                feature_vector = None
            features_list.append(feature_vector)
            continue

        audio_path = audio_files_map[video_id]

        feature_sequence = extract_audio_feature_768(
            model, audio_path, device=device, target_time_steps=16)

        if feature_sequence is None:
            failed_extraction.append(video_id)

            if len(features_list) > 0:
                feature_sequence = np.zeros_like(features_list[0])
            else:
                feature_sequence = None

        features_list.append(feature_sequence)

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

    if missing_audio:
        print("SPCDH")
        for vid in missing_audio[:10]:
            print(f"  - {vid}")
        if len(missing_audio) > 10:
            print("SPCDH")

    if failed_extraction:
        print("SPCDH")
        for vid in failed_extraction[:10]:
            print(f"  - {vid}")
        if len(failed_extraction) > 10:
            print("SPCDH")


if __name__ == "__main__":

    VIDEO_IDS_JSON = "data/features/video_ids.json"
    AUDIO_ROOT = "/media/Backup/cr/dataset_hash/audio"
    OUTPUT_PATH = "data/features/audio_768.npy"

    BEATS_CHECKPOINT = "featureEncoder/BEATS/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print("SPCDH")
        print("SPCDH")
        print("SPCDH")
    else:
        DEVICE = 'cpu'
        print("SPCDH")
        if hasattr(torch.version, 'cuda'):
            print("SPCDH")

    print("=" * 60)
    print("SPCDH")
    print("=" * 60)
    print("SPCDH")
    print(f"  video_ids.json: {VIDEO_IDS_JSON}")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("SPCDH")
    print("=" * 60)

    if not os.path.exists(VIDEO_IDS_JSON):
        print("SPCDH")
        exit(1)

    if not os.path.exists(AUDIO_ROOT):
        print("SPCDH")
        exit(1)

    if not os.path.exists(BEATS_CHECKPOINT):
        print("SPCDH")
        print("SPCDH")
        exit(1)

    extract_audio_features_from_list(
        video_ids_json=VIDEO_IDS_JSON,
        audio_root=AUDIO_ROOT,
        output_path=OUTPUT_PATH,
        beats_checkpoint=BEATS_CHECKPOINT,
        device=DEVICE
    )

    print("=" * 60)
    print("SPCDH")
    print("=" * 60)
