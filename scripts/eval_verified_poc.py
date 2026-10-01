import argparse
import json
import os
import sys
import numpy as np

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch

_orig_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _compat_torch_load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vga_policy import VGAPolicy
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def resolve_bddl(task):
    bddl_dir = benchmark.get_libero_path("bddl_files")
    p1 = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(bddl_dir, task.bddl_file)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Cannot resolve BDDL: {task.bddl_file}")


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean_k = "mean" if "mean" in stats else "mu"
    std_k = "std" if "std" in stats else "sigma"
    mu = torch.as_tensor(stats[mean_k], dtype=torch.float32, device=device)
    sigma = torch.as_tensor(stats[std_k], dtype=torch.float32, device=device)

    with open(args.extrinsics_path, "r", encoding="utf-8") as f:
        extrinsics = json.load(f)
    t_base = torch.as_tensor(extrinsics["t_base_cam"], dtype=torch.float32, device=device).unsqueeze(0)

    print(f">> Initializing policy: {args.checkpoint}", flush=True)
    model = VGAPolicy(load_pretrained=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    bench = benchmark.get_benchmark("libero_spatial")()
    canonical_indices = [0, 1, 2]

    results = {}
    total_successes = 0
    total_trials = 0

    print(f">> Evaluating Canonical Task Indices: {canonical_indices}", flush=True)
    print(f">> Executing {args.num_trials} Trials/Task | Receding Horizon: {args.exec_horizon}/16", flush=True)

    for task_id in canonical_indices:
        task = bench.get_task(task_id)
        task_name = task.name
        task_prompt = task.language
        task_successes = 0

        bddl_path = resolve_bddl(task)
        env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
        init_states = bench.get_task_init_states(task_id)

        for trial in range(args.num_trials):
            print(f"   [{task_name[:36]}... | Trial {trial+1:02d}/{args.num_trials:02d}] Running...", flush=True)
            obs = env.reset()
            if init_states is not None and len(init_states) > 0:
                init_idx = trial % len(init_states)
                init_s = init_states[init_idx]
                if isinstance(init_s, torch.Tensor):
                    init_s = init_s.cpu().numpy()
                obs = env.set_init_state(init_s)

            gripper_closed = False
            success = False
            step_count = 0
            a_prev = torch.zeros((1, 4, 6), device=device)

            while step_count < args.max_steps and not success:
                raw_frame = obs["agentview_image"]
                flipped_frame = np.ascontiguousarray(raw_frame[::-1, :, :])
                rgb = torch.from_numpy(flipped_frame).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_chunk = model.sample_actions(
                            rgb=rgb,
                            t_base_cam=t_base,
                            a_prev=a_prev,
                            prompt=[task_prompt]
                        )

                if isinstance(pred_chunk, tuple):
                    pred_chunk = pred_chunk[0]
                pred_chunk = pred_chunk.squeeze(0).float()
                pred_unnorm = (pred_chunk * sigma + mu).cpu().numpy()

                slice_len = min(args.exec_horizon, len(pred_unnorm))
                for h in range(slice_len):
                    action = pred_unnorm[h].copy()

                    if action[-1] > 0.5:
                        gripper_closed = True
                    elif action[-1] < -0.5:
                        gripper_closed = False
                    action[-1] = 1.0 if gripper_closed else -1.0

                    obs, reward, done, info = env.step(action)
                    step_count += 1

                    env_success = False
                    if hasattr(env, "check_success"):
                        env_success = env.check_success()
                    elif hasattr(env, "env") and hasattr(env.env, "check_success"):
                        env_success = env.env.check_success()

                    if done or env_success:
                        success = True
                        task_successes += 1
                        break

                if slice_len >= 4:
                    recent_actions = pred_unnorm[slice_len - 4 : slice_len, :6]
                else:
                    pad = np.zeros((4 - slice_len, 6), dtype=np.float32)
                    recent_actions = np.vstack([pad, pred_unnorm[:slice_len, :6]])
                a_prev = torch.from_numpy(recent_actions).float().unsqueeze(0).to(device)

            status = "SUCCESS [✓]" if success else "FAILED [✗]"
            print(f"      --> Trial {trial+1:02d}: {status} in {step_count} steps", flush=True)

        env.close()
        torch.cuda.empty_cache()

        sr = (task_successes / args.num_trials) * 100.0
        results[task_name] = {"successes": task_successes, "trials": args.num_trials, "sr": sr}
        total_successes += task_successes
        total_trials += args.num_trials
        print(f"   [Task Result] {task_name}: {sr:5.1f}%\n", flush=True)

    overall_sr = (total_successes / total_trials) * 100.0
    results["overall_canonical_sr"] = overall_sr
    print(f">> Evaluation Complete. Overall Success Rate: {overall_sr:.2f}%", flush=True)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--extrinsics_path", type=str, default="configs/camera_extrinsics.json")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--exec_horizon", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate(args)
