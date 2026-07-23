import argparse
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from huggingface_hub import login

# SAM3 from Hugging Face
from transformers import Sam3Processor, Sam3Model

# -----------------------------
# Utils
# -----------------------------
def read_file(path):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# -----------------------------
# Dataset
# -----------------------------
class ImageDataset(Dataset):
    def __init__(self, directory, valid_exts=(".jpg", ".jpeg", ".png", ".bmp")):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(valid_exts)
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
            return img, os.path.basename(path)
        except Exception as e:
            print("Error:", path, e)
            return None, None

def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    imgs, names = zip(*batch)
    return list(imgs), list(names)

# -----------------------------
# Pooling
# -----------------------------
def pool_embedding(emb, mode):
    # SAM 3 vision encoder outputs flattened patches: (batch_size, sequence_length, hidden_size)
    if mode == "gap":
        # Average across the sequence length (dimension 1)
        return emb.mean(dim=1).squeeze(0)
    elif mode == "flatten":
        return emb.flatten(start_dim=1).squeeze(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SAM3 Feature Extractor")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache/")
    parser.add_argument(
        "--mode",
        type=str,
        default="gap",
        choices=["gap", "flatten"],
        help="Pooling strategy",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # -------- cache --------
    os.environ["HF_HOME"] = args.cache_dir

    # -------- device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------- login --------
    if args.token:
        token_str = read_file(args.token)
        if token_str:
            login(token=token_str)

    # -------- model --------
    print("Loading SAM3 from Hugging Face...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()

    # -------- data --------
    dataset = ImageDataset(args.input_dir)

    if len(dataset) == 0:
        raise ValueError("No images found in the specified directory.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
    )

    all_features = []
    all_filenames = []

    print("Starting extraction...")

    # -------- loop --------
    with torch.no_grad():
        for images, filenames in tqdm(dataloader):

            if images is None:
                continue

            batch_feats = []

            for img in images:
                # 1. Use processor to format image into tensors
                inputs = processor(images=img, return_tensors="pt").to(device)

                # 2. Bypass the text/prompt requirements by calling the vision encoder directly!
                vision_outputs = model.vision_encoder(inputs["pixel_values"])
                
                # 3. Extract the last hidden state
                emb = vision_outputs.last_hidden_state

                # 4. Pool the embeddings (now adjusted for 3D tensor shape)
                pooled = pool_embedding(emb, args.mode)
                batch_feats.append(pooled.cpu().numpy())

            # Stack individual feature arrays back together into a batch
            batch_feats = np.stack(batch_feats)

            all_features.append(batch_feats)
            all_filenames.extend(filenames)

    # -------- save --------
    if not all_features:
        print("No features extracted. Exiting.")
        return

    final_features = np.vstack(all_features)
    final_filenames = np.array(all_filenames)

    print("Final features shape:", final_features.shape)

    np.savez_compressed(
        args.output_file,
        features=final_features,
        filenames=final_filenames,
    )

    print("Successfully saved to:", args.output_file)

if __name__ == "__main__":
    main()

