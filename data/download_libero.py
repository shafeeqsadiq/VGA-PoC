"""
data/download_libero.py
Downloads, isolates, and partitions LIBERO-Spatial demonstration tasks via LeRobot.
Extracts 5-shot, 10-shot, and validation splits and saves them as structured .npz files.
"""
import os
import argparse
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TARGET_TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "push_the_plate_to_the_front_of_the_stove"
]

def sanitize_task_name(task_str: str) -> str:
    """Matches raw instruction strings against canonical target tasks."""
    task_clean = task_str.lower().replace(" ", "_").replace("-", "_")
    for target in TARGET_TASKS:
        if target in task_clean or task_clean in target:
            return target
        # Fuzzy match key sub-tokens
        if "black_bowl" in task_clean and "plate" in task_clean:
            return TARGET_TASKS[0]
        if "alphabet_soup" in task_clean and "basket" in task_clean:
            return TARGET_TASKS[1]
        if "plate" in task_clean and "stove" in task_clean:
            return TARGET_TASKS[2]
    return ""

def process_and_partition(repo_id: str, raw_cache_dir: str, output_root: str):
    print(f">> Initializing LeRobot download for '{repo_id}'...")
    os.makedirs(raw_cache_dir, exist_ok=True)
    
    # Download nominal dataset from Hugging Face via LeRobot
    dataset = LeRobotDataset(repo_id, root=raw_cache_dir)
    print(f">> Successfully loaded raw dataset ({dataset.num_episodes} total episodes).")

    # Group episode indices by target task
    task_to_episodes: Dict[str, List[int]] = {task: [] for task in TARGET_TASKS}
    
    print(">> Indexing episodes across benchmark tasks...")
    for ep_idx in range(dataset.num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        task_str = ep_meta.get("task", ep_meta.get("language_instruction", ""))
        matched_task = sanitize_task_name(task_str)
        if matched_task in task_to_episodes:
            task_to_episodes[matched_task].append(ep_idx)

    for task, eps in task_to_episodes.items():
        print(f"   - Found {len(eps)} episodes for: {task}")
        assert len(eps) >= 10, f"Insufficient episodes ({len(eps)}) for task: {task}"

    # Partition schemes
    split_configs = {
        "5_shot": 5,
        "10_shot": 10
    }

    # Nominal LIBERO camera parameters (agentview)
    default_t_base_cam = np.array([0.25, -0.35, 0.45], dtype=np.float32)
    default_R_base_cam = np.array([
        [0.0, -0.7071, 0.7071],
        [1.0,  0.0,    0.0   ],
        [0.0,  0.7071, 0.7071]
    ], dtype=np.float32)

    for split_name, count_per_task in split_configs.items():
        split_dir = os.path.join(output_root, f"libero_spatial_{split_name}")
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n>> Extracting {count_per_task} demonstrations per task into: {split_dir}")

        total_saved = 0
        for task in TARGET_TASKS:
            selected_episodes = task_to_episodes[task][:count_per_task]
            for local_idx, ep_idx in enumerate(tqdm(selected_episodes, desc=f"Exporting {task[:25]}...")):
                from_idx = dataset.episode_data_index["from"][ep_idx].item()
                to_idx = dataset.episode_data_index["to"][ep_idx].item()

                frames = dataset.hf_dataset.select(range(from_idx, to_idx))
                
                # Extract observations and actions
                rgb_frames = np.stack([np.array(x["observation.images.agentview"]) for x in frames])
                actions = np.stack([np.array(x["action"]) for x in frames]).astype(np.float32)

                # Extract depth if available in dataset, else allocate dummy zero channel
                if "observation.images.depth" in frames.column_names:
                    depth_frames = np.stack([np.array(x["observation.images.depth"]) for x in frames]).astype(np.float32)
                else:
                    depth_frames = np.zeros((len(frames), 256, 256, 1), dtype=np.float32)

                out_filename = os.path.join(split_dir, f"task_{TARGET_TASKS.index(task)}_ep_{local_idx:02d}.npz")
                np.savez_compressed(
                    out_filename,
                    rgb=rgb_frames,
                    depth=depth_frames,
                    actions=actions,
                    t_base_cam=default_t_base_cam,
                    R_base_cam=default_R_base_cam,
                    prompt=task
                )
                total_saved += 1

        print(f">> Wrote {total_saved} episodes to {split_dir}")

    # Export held-out validation set (nominal demonstrations after index 10)
    val_dir = os.path.join(output_root, "libero_spatial_val")
    os.makedirs(val_dir, exist_ok=True)
    val_count = 0
    for task in TARGET_TASKS:
        val_episodes = task_to_episodes[task][10:30]  # Next 20 episodes
        for local_idx, ep_idx in enumerate(val_episodes):
            from_idx = dataset.episode_data_index["from"][ep_idx].item()
            to_idx = dataset.episode_data_index["to"][ep_idx].item()
            frames = dataset.hf_dataset.select(range(from_idx, to_idx))
            
            rgb_frames = np.stack([np.array(x["observation.images.agentview"]) for x in frames])
            actions = np.stack([np.array(x["action"]) for x in frames]).astype(np.float32)
            
            out_filename = os.path.join(val_dir, f"val_{TARGET_TASKS.index(task)}_ep_{local_idx:02d}.npz")
            np.savez_compressed(
                out_filename,
                rgb=rgb_frames,
                actions=actions,
                t_base_cam=default_t_base_cam,
                R_base_cam=default_R_base_cam,
                prompt=task
            )
            val_count += 1
    print(f">> Wrote {val_count} held-out validation episodes to {val_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="lerobot/libero_spatial")
    parser.add_argument("--raw_cache_dir", type=str, default="/workspace/vga_poc/data/raw_lerobot")
    parser.add_argument("--output_root", type=str, default="/workspace/vga_poc/data")
    args = parser.parse_args()

    process_and_partition(args.repo_id, args.raw_cache_dir, args.output_root)