"""
scripts/compute_action_stats.py
Calculates empirical action mean, standard deviation, and physical variances
(sigma_pos_sq, sigma_rot_sq) across demonstrations. Saves to configs/action_stats.json.
"""
import os
import json
import argparse
import glob
import numpy as np

def compute_and_save_stats(data_dir: str, output_path: str):
    print(f">> Searching for demonstration archives in: {data_dir}")
    # Search recursively to support nested task directories
    npz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir} (or its subdirectories).")

    print(f">> Ingesting actions across {len(npz_files)} demonstration files...")
    all_actions = []
    for f in npz_files:
        with np.load(f) as data:
            if "actions" in data:
                all_actions.append(data["actions"])

    if not all_actions:
        raise ValueError(f"Found .npz files in {data_dir}, but none contained an 'actions' array.")

    stacked_actions = np.concatenate(all_actions, axis=0)  # Shape [Total_Steps, 7]
    assert stacked_actions.shape[-1] == 7, f"Expected 7-DoF actions, found shape {stacked_actions.shape}"

    # 1. Compute empirical mean and standard deviation
    mu = np.mean(stacked_actions, axis=0)
    sigma = np.std(stacked_actions, axis=0)

    # 2. Prevent division-by-zero on static channels (e.g. constant discrete gripper states)
    sigma_clean = np.where(sigma < 1e-4, 1.0, sigma)

    # 3. Compute empirical physical variance normalizers for CompositeVGALoss
    # Indices 0..2: Cartesian translational deltas [dx, dy, dz]
    # Indices 3..5: Axis-angle rotational deltas [drx, dry, drz]
    sigma_pos_sq = float(np.mean(sigma_clean[:3] ** 2))
    sigma_rot_sq = float(np.mean(sigma_clean[3:6] ** 2))

    # Guard against zero variances
    sigma_pos_sq = max(sigma_pos_sq, 1e-6)
    sigma_rot_sq = max(sigma_rot_sq, 1e-6)

    # 4. Action percentile clipping thresholds (for post-inference safety bounds)
    q01 = np.percentile(stacked_actions, 1, axis=0).tolist()
    q99 = np.percentile(stacked_actions, 99, axis=0).tolist()

    stats = {
        "mu": mu.tolist(),
        "sigma": sigma_clean.tolist(),
        "sigma_pos_sq": sigma_pos_sq,
        "sigma_rot_sq": sigma_rot_sq,
        "clip_q01": q01,
        "clip_q99": q99,
        "total_steps": int(stacked_actions.shape[0]),
        "action_dim": int(stacked_actions.shape[1])
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=4)

    print(f">> Successfully written action normalizers to: {output_path}")
    print(f"   Mean:         {np.round(mu, 4).tolist()}")
    print(f"   Std:          {np.round(sigma_clean, 4).tolist()}")
    print(f"   sigma_pos_sq: {sigma_pos_sq:.6e}")
    print(f"   sigma_rot_sq: {sigma_rot_sq:.6e}")
    print(f"   Total Frames: {stats['total_steps']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/libero_spatial_10shot")
    parser.add_argument("--output_path", type=str, default="configs/action_stats.json")
    args = parser.parse_args()
    compute_and_save_stats(args.data_dir, args.output_path)