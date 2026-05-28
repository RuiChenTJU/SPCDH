

import json
import os
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


INPUT_JSON = 'data/prototypes/semantic_clouds2.json'
OUTPUT_PT = 'data/prototypes/semantic_embeddings.pt'
CLIP_MODEL_DIR = 'featureEncoder/CLIP'
MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def encode_prototypes():
    print("=" * 60)
    print("SPCDH")
    print("=" * 60)

    if not os.path.exists(INPUT_JSON):
        print("SPCDH")
        print("SPCDH")
        return

    print("SPCDH")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("SPCDH")
    if device == "cuda":
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print("SPCDH")
    else:
        print("SPCDH")

    if os.path.exists(CLIP_MODEL_DIR):
        print("SPCDH")
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_DIR)
        model = CLIPTextModel.from_pretrained(CLIP_MODEL_DIR).to(device)
    else:
        print("SPCDH")
        print("SPCDH")
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
        model = CLIPTextModel.from_pretrained(MODEL_ID).to(device)

    model.eval()
    print("SPCDH")

    print("=" * 60)
    print("SPCDH")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("SPCDH")

    total_descriptions = 0
    valid_labels = {}
    for label_key, label_data in data.items():
        if isinstance(label_data, dict) and "semantic_point_cloud" in label_data:
            descriptions = label_data["semantic_point_cloud"]
            if isinstance(descriptions, list) and len(descriptions) > 0:
                valid_labels[label_key] = descriptions
                total_descriptions += len(descriptions)
        else:
            print("SPCDH")

    print("SPCDH")
    print("SPCDH")

    embeddings_dict = {}

    print("=" * 60)
    print("SPCDH")
    print("SPCDH")

    with torch.no_grad():
        for label_key, descriptions in tqdm(valid_labels.items(), desc="SPCDH"):

            if not descriptions or len(descriptions) == 0:
                print("SPCDH")
                continue

            inputs = tokenizer(
                descriptions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            feats = outputs.pooler_output

            embeddings_dict[label_key] = feats.cpu()

    print("=" * 60)
    print("SPCDH")
    print("SPCDH")

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)

    torch.save(embeddings_dict, OUTPUT_PT)

    print("=" * 60)
    print("SPCDH")
    print("SPCDH")

    if len(embeddings_dict) > 0:
        first_key = list(embeddings_dict.keys())[0]
        first_shape = embeddings_dict[first_key].shape
        print("SPCDH")
        print("SPCDH")

    print("SPCDH")
    print("SPCDH")
    print("=" * 60)


if __name__ == "__main__":
    encode_prototypes()
