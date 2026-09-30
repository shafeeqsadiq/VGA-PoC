"""
data/vq_depth/generate_codebook.py
Precomputes discrete f=32 VQ-depth codebook indices for demonstration frames.
Trains or loads DepthVQVAE, saves codebook_k256.pt, and updates .npz files with 'depth_tokens'.
"""
import os
import glob
import argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from configs.poc_config import CONFIG
from data.vq_depth.depth_vqvae import DepthVQVAE

def extract_sanitized_depth(data_dict: dict) -> np.ndarray:
    """
    Extracts and standardizes depth maps to shape [T, 1, 256, 256] in float32.
    Falls back to RGB grayscale proxy only if depth is completely missing or zero.
    """
    has_real_depth = False
    if "depth" in data_dict:
        raw_depth = data_dict["depth"]
        if raw_depth.size > 0 and np.any(raw_depth != 0):
            has_real_depth = True
            depth_arr = raw_depth.astype(np.float32)

            if depth_arr.ndim == 3:
                # [T, 256, 256] -> [T, 1, 256, 256]
                depth_arr = depth_arr[:, None, :, :]
            elif depth_arr.ndim == 4 and depth_arr.shape[-1] == 1:
                # [T, 256, 256, 1] -> [T, 1, 256, 256]
                depth_arr = np.transpose(depth_arr, (0, 3, 1, 2))

    if not has_real_depth:
        rgb = data_dict["rgb"].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Calculate grayscale luminosity: 0.299 R + 0.587 G + 0.114 B
        if rgb.ndim == 4 and rgb.shape[-1] == 3:
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            depth_arr = gray[:, None, :, :]
        elif rgb.ndim == 4 and rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            depth_arr = gray[:, None, :, :]
        else:
            raise ValueError(f"Unrecognized RGB observation format with shape {rgb.shape}")

    # Enforce standard spatial dimensions (256, 256)
    if depth_arr.shape[-2:] != (256, 256):
        tensor_d = torch.from_numpy(depth_arr)
        tensor_d = F.interpolate(tensor_d, size=(256, 256), mode="bilinear", align_corners=False)
        depth_arr = tensor_d.numpy()

    return depth_arr

def train_vqvae(
    vqvae: DepthVQVAE,
    depth_tensors: torch.Tensor,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3
):
    """Fits VQ-VAE codebook on collected demonstration depth frames."""
    vqvae.train()
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=lr)
    dataset = TensorDataset(depth_tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f">> Training DepthVQVAE on {len(depth_tensors)} depth frames for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        last_perp = torch.tensor(0.0)

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon, vq_loss, _, perp = vqvae(batch_x)
            recon_loss = F.mse_loss(recon, batch_x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            last_perp = perp

        n_batches = max(len(loader), 1)
        print(
            f"   Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Loss: {total_loss / n_batches:.4f} | "
            f"Recon: {total_recon / n_batches:.4f} | "
            f"VQ: {total_vq / n_batches:.4f} | "
            f"Perplexity: {last_perp.item():.1f}"
        )
    vqvae.eval()

def generate_and_cache_tokens(
    data_root: str,
    codebook_out: str,
    device_str: str = "cpu",
    epochs: int = 5,
    force_retrain: bool = False
):
    dev = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print(f">> Initializing DepthVQVAE pipeline on {dev}...")
    vqvae = DepthVQVAE(codebook_size=CONFIG.vq_codebook_size, embedding_dim=64).to(dev)

    npz_files = sorted(glob.glob(os.path.join(data_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under: {data_root}")

    print(f">> Discovered {len(npz_files)} demonstration archives. Ingesting depth buffers...")
    file_depth_data = {}
    collected_frames = []

    for file_path in tqdm(npz_files, desc="Loading depth arrays"):
        with np.load(file_path, allow_pickle=True) as data:
            data_dict = {k: data[k] for k in data.files}
        depth_arr = extract_sanitized_depth(data_dict)
        file_depth_data[file_path] = (data_dict, depth_arr)
        # Collect frames for fitting codebook (subsample by factor of 2 to conserve memory)
        collected_frames.append(depth_arr[::2])

    # 1. Train or Load Codebook Weights
    if os.path.exists(codebook_out) and not force_retrain:
        print(f">> Found existing codebook at: {codebook_out}. Loading weights...")
        vqvae.load_state_dict(torch.load(codebook_out, map_location=dev))
    else:
        stacked_depth = np.concatenate(collected_frames, axis=0)
        tensor_depth = torch.from_numpy(stacked_depth).float()
        train_vqvae(vqvae, tensor_depth, dev, epochs=epochs)
        
        os.makedirs(os.path.dirname(os.path.abspath(codebook_out)), exist_ok=True)
        torch.save(vqvae.state_dict(), codebook_out)
        print(f">> Saved trained VQ-VAE model and codebook weights to: {codebook_out}")

    vqvae.eval()

    # 2. Tokenize Depth Maps and Update Demonstration Archives
    print(f">> Precomputing [64] discrete spatial tokens per frame...")
    batch_size = 32

    for file_path, (data_dict, depth_maps) in tqdm(file_depth_data.items(), desc="Tokenizing"):
        num_frames = depth_maps.shape[0]
        token_batches = []

        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                batch_depth = depth_maps[i : i + batch_size]
                tensor_batch = torch.from_numpy(batch_depth).float().to(dev)
                indices = vqvae.encode_to_indices(tensor_batch)
                token_batches.append(indices.cpu().numpy().astype(np.int64))

        all_tokens = np.concatenate(token_batches, axis=0)  # Shape [T, 64]
        data_dict["depth_tokens"] = all_tokens

        # Persist updated archive with depth_tokens key
        np.savez_compressed(file_path, **data_dict)

    print(f">> Successfully tokenized all episodes with discrete geometric tokens.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--codebook_out", type=str, default="data/vq_depth/codebook_k256.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    generate_and_cache_tokens(args.data_root, args.codebook_out, args.device, args.epochs, args.retrain)