import argparse
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from huggingface_hub import login

# Default Model ID (can be overridden via args)
DEFAULT_MODEL = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

os.environ['HF_HOME'] = './cache/'

def read_file(path):
    """Safely reads external text files."""
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

class ImageDataset(Dataset):
    """
    A PyTorch Dataset that handles directory traversing, loading, AND preprocessing.
    Moving preprocessing here prevents GPU starvation.
    """
    def __init__(self, directory, processor, valid_exts=('.jpg', '.jpeg', '.png', '.bmp')):
        self.processor = processor
        self.files = [
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.lower().endswith(valid_exts)
        ]
        self.files.sort() # Ensure deterministic order

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # Open as RGB to ensure 3 channels
            image = Image.open(path).convert("RGB")
            
            # Preprocess directly inside the worker to free up the main thread
            # squeeze(0) removes the batch dimension added by the processor
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            
            return pixel_values, os.path.basename(path)
        except Exception as e:
            print(f"Error loading/processing {path}: {e}")
            # Return dummy/None to filter out later (handled in collate)
            return None, None

def custom_collate(batch):
    """Filters out failed image loads and correctly stacks the tensors."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return None, None
    
    pixel_values, filenames = zip(*batch)
    
    # Stack individual image tensors into a single batch tensor: [B, C, H, W]
    batched_pixel_values = torch.stack(pixel_values)
    
    return batched_pixel_values, list(filenames)

def get_config_attribute(model, attr_name, default):
    """Safely retrieves config attributes (like num_registers)."""
    return getattr(model.config, attr_name, default)

def main():
    parser = argparse.ArgumentParser(description="Batch DINOv3 Feature Extractor (Optimized)")
    
    # Inputs
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save .npz file (e.g., features.npz)")
    
    # Model Config
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument("--token", type=str, default=None, help="Path to HF Token file (optional)")
    
    # Processing Options
    parser.add_argument("--mode", type=str, choices=["cls", "cls_avg"], default="cls_avg", 
                        help="Extraction mode: 'cls' (Class token only) or 'cls_avg' (Class + Avg Patch)")
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch (can likely be increased now)")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading/preprocessing workers")

    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine optimal precision (use float16 on GPU, float32 on CPU)
    #dtype = torch.float16 if device.type == "cuda" else torch.float32
    dtype = torch.float32

    token = read_file(args.token)
    if token:
        login(token=token)

    # 2. Load Model & Processor
    print(f"Loading Model: {args.model_id} in {dtype}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id, token=token)
    
    # Load model directly into target precision to save VRAM and boost speed
    model = AutoModel.from_pretrained(
        args.model_id, 
        token=token,
        torch_dtype=dtype
    ).to(device)
    model.eval()

    # Determine Register count dynamically
    num_registers = get_config_attribute(model, "num_register_tokens", 4)
    print(f"Model Configuration Detected: {num_registers} register tokens.")

    # 3. Setup Data Loader
    # Notice we now pass the processor into the dataset
    dataset = ImageDataset(args.input_dir, processor=processor)
    if len(dataset) == 0:
        raise ValueError(f"No valid images found in {args.input_dir}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True if device.type == "cuda" else False # Speeds up CPU to GPU transfer
    )

    all_features = []
    all_filenames = []

    print(f"Starting extraction for {len(dataset)} images in mode: [{args.mode}]...")

    # 4. Processing Loop
    # inference_mode is faster and uses less memory than no_grad
    with torch.inference_mode():
        for pixel_values, filenames in tqdm(dataloader):
            if pixel_values is None: continue # Skip empty batches
            
            # Move pre-processed batch to GPU and cast to appropriate precision (e.g., float16)
            pixel_values = pixel_values.to(device, dtype=dtype)
            
            # Forward Pass
            outputs = model(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state 
            # Shape: [Batch, Seq_Len, Hidden_Dim]

            # --- Feature Logic ---
            
            # A. Extract CLS Token (Index 0)
            cls_token = last_hidden_state[:, 0, :]
            
            if args.mode == "cls":
                batch_feats = cls_token
                
            elif args.mode == "cls_avg":
                # B. Extract Patches (Skip CLS + Registers)
                patch_start_idx = 1 + num_registers
                patch_tokens = last_hidden_state[:, patch_start_idx:, :]
                
                # C. Average Patches
                avg_patch_token = patch_tokens.mean(dim=1)
                
                # D. Concatenate (CLS + AVG)
                batch_feats = torch.cat((cls_token, avg_patch_token), dim=1)

            # Move to CPU, cast back to float32 (standard for saving/NumPy), and convert to array
            # Keeping it in float16 for NumPy saving is possible, but float32 is safer for downstream ML tools
            all_features.append(batch_feats.cpu().to(torch.float32).numpy())
            all_filenames.extend(filenames)

    # 5. Concatenate and Save
    final_features = np.vstack(all_features)
    final_filenames = np.array(all_filenames)

    print("-" * 40)
    print(f"Extraction Complete.")
    print(f"Features Shape: {final_features.shape}")
    print(f"Saving to {args.output_file}...")
    
    # Save as compressed NumPy archive
    np.savez_compressed(
        args.output_file, 
        features=final_features, 
        filenames=final_filenames
    )
    print("Done.")

if __name__ == "__main__":
    main()

