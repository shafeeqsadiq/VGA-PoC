"""
data/dataset.py
PyTorch Dataset for LIBERO demonstration trajectories.
Handles 16-step chunking, P=4 tail prefix extraction, image resizing to 384x384,
and metric action normalization. Includes an automated HDF5/Zarr disk reader.
"""
from typing import Any, Dict, List, Optional, Union
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from configs.poc_config import CONFIG

class LiberoSpatialDataset(Dataset):
    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        chunk_horizon: int = CONFIG.chunk_horizon,
        prefix_len: int = CONFIG.buffer_prefix_len,
        action_stats: Optional[Dict[str, Any]] = None,
        target_img_size: tuple = CONFIG.vis_input_size  # (384, 384)
    ):
        self.episodes = episodes
        self.chunk_horizon = chunk_horizon
        self.prefix_len = prefix_len
        self.target_img_size = target_img_size
        self.indices = []

        # Parse action normalizer stats (supports lists, numpy arrays, or tensors)
        if action_stats is not None:
            self.mu = torch.as_tensor(action_stats["mu"], dtype=torch.float32)
            self.sigma = torch.as_tensor(action_stats["sigma"], dtype=torch.float32)
        else:
            self.mu = torch.zeros(CONFIG.action_dim, dtype=torch.float32)
            self.sigma = torch.ones(CONFIG.action_dim, dtype=torch.float32)

        # Index all valid trajectory slice start positions
        for ep_idx, ep in enumerate(self.episodes):
            traj_len = len(ep["actions"])
            for t in range(traj_len - chunk_horizon + 1):
                self.indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        ep = self.episodes[ep_idx]

        # -------------------------------------------------------------
        # 1. Visual observation: load, convert to [C, H, W], and resize to 384x384
        # -------------------------------------------------------------
        raw_rgb = ep["rgb"][t]
        if isinstance(raw_rgb, np.ndarray):
            rgb_tensor = torch.from_numpy(raw_rgb).float()
        else:
            rgb_tensor = raw_rgb.float()

        # Ensure shape [C, H, W]
        if rgb_tensor.ndim == 3 and rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)

        # Normalize to [0.0, 1.0] if in uint8 range
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0

        # Resize to 384x384 to guarantee exactly 576 patches for 3x space-to-depth
        if rgb_tensor.shape[1:] != self.target_img_size:
            rgb_tensor = TF.resize(
                rgb_tensor,
                list(self.target_img_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True
            )

        # -------------------------------------------------------------
        # 2. Camera pose & translation (supports static and per-step extrinsics)
        # -------------------------------------------------------------
        t_cam = ep.get("t_base_cam", np.zeros(3, dtype=np.float32))
        R_cam = ep.get("R_base_cam", np.eye(3, dtype=np.float32))

        t_base_cam = torch.as_tensor(t_cam[t] if t_cam.ndim > 1 else t_cam, dtype=torch.float32)
        R_base_cam = torch.as_tensor(R_cam[t] if R_cam.ndim > 2 else R_cam, dtype=torch.float32)

        # -------------------------------------------------------------
        # 3. Action chunk x_1: [H, 7]
        # -------------------------------------------------------------
        raw_actions = torch.as_tensor(ep["actions"][t : t + self.chunk_horizon], dtype=torch.float32)
        norm_actions = (raw_actions - self.mu) / (self.sigma + 1e-8)

        # -------------------------------------------------------------
        # 4. Prefix history a_prev: [P, 6] (steps t-P .. t-1)
        # -------------------------------------------------------------
        if t >= self.prefix_len:
            a_prev = torch.as_tensor(ep["actions"][t - self.prefix_len : t, :6], dtype=torch.float32)
        else:
            pad_len = self.prefix_len - t
            act_slice = torch.as_tensor(ep["actions"][:t, :6], dtype=torch.float32) if t > 0 else torch.empty(0, 6)
            zeros = torch.zeros(pad_len, 6, dtype=torch.float32)
            a_prev = torch.cat([zeros, act_slice], dim=0)

        # -------------------------------------------------------------
        # 5. Precomputed discrete depth tokens z*: [64]
        # -------------------------------------------------------------
        if "depth_tokens" in ep:
            z_star = torch.as_tensor(ep["depth_tokens"][t], dtype=torch.long)
        else:
            # Fallback to zeros if offline tokenizer has not executed yet
            z_star = torch.zeros(CONFIG.compressed_tokens, dtype=torch.long)

        # Prompt string extraction
        prompt = ep.get("prompt", ep.get("task", ep.get("language_instruction", "")))

        return {
            "rgb": rgb_tensor,
            "t_base_cam": t_base_cam,
            "R_base_cam": R_base_cam,
            "actions_norm": norm_actions,
            "actions_raw": raw_actions,
            "a_prev": a_prev,
            "z_star": z_star,
            "prompt": prompt
        }

    @classmethod
    def from_disk(
        cls,
        data_path: str,
        chunk_horizon: int = CONFIG.chunk_horizon,
        prefix_len: int = CONFIG.buffer_prefix_len,
        action_stats: Optional[Dict[str, Any]] = None
    ) -> "LiberoSpatialDataset":
        """
        Directly loads episodes stored in HDF5 or NPZ format from disk.
        """
        import h5py
        episodes = []

        if os.path.isfile(data_path) and data_path.endswith(".hdf5") or data_path.endswith(".h5"):
            with h5py.File(data_path, "r") as f:
                data_grp = f["data"]
                for demo_key in sorted(data_grp.keys()):
                    demo = data_grp[demo_key]
                    episodes.append({
                        "rgb": np.array(demo["obs"]["agentview_rgb"]),
                        "actions": np.array(demo["actions"]),
                        "prompt": demo.attrs.get("prompt", "manipulate object")
                    })
        elif os.path.isdir(data_path):
            import glob
            npz_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
            for file_path in npz_files:
                with np.load(file_path) as data:
                    episodes.append({k: data[k] for k in data.files})
        else:
            raise FileNotFoundError(f"No valid HDF5 or NPZ demonstrations found at {data_path}")

        return cls(episodes, chunk_horizon, prefix_len, action_stats)