"""
controllers/eval_runner.py
Closed-loop simulation rollout engine.
Manages asynchronous double-buffering, Schmitt-trigger gripper filtering,
Gymnasium/Robosuite observation unpacking, and step execution across LIBERO tasks.
"""
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from configs.poc_config import CONFIG
from controllers.async_buffer import AsyncActionBuffer
from controllers.schmitt_trigger import SchmittTriggerGripper
from models.vga_policy import VGAPolicy

class RolloutRunner:
    def __init__(
        self,
        policy: VGAPolicy,
        action_stats: Dict[str, Any],
        device: torch.device,
        max_steps: int = 350
    ):
        self.policy = policy
        self.device = device
        self.max_steps = max_steps

        # Action normalizers
        self.mu = np.array(action_stats["mu"], dtype=np.float32)
        self.sigma = np.array(action_stats["sigma"], dtype=np.float32)

        # Schmitt-trigger gripper hysteresis filter
        self.gripper_filter = SchmittTriggerGripper(
            low_thresh=CONFIG.schmitt_low,
            high_thresh=CONFIG.schmitt_high
        )
        self.buffer: Optional[AsyncActionBuffer] = None
        self.last_jerk: float = 0.0

    def unnormalize_action(self, norm_action: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Denormalizes policy outputs: a_raw = a_norm * sigma + mu."""
        if isinstance(norm_action, torch.Tensor):
            norm_action = norm_action.detach().cpu().numpy()
        return norm_action * self.sigma + self.mu

    def _extract_rgb(self, obs: Any) -> np.ndarray:
        """Robustly extracts RGB image from Gym/Gymnasium/Robosuite observation dicts."""
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            for key in ["agentview_rgb", "agentview_image", "rgb", "image"]:
                if key in obs:
                    return obs[key]
            # Search for any 3-channel array
            for v in obs.values():
                if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim >= 3:
                    if v.shape[0] == 3 or v.shape[-1] == 3:
                        return v if isinstance(v, np.ndarray) else v.cpu().numpy()
        return obs

    def run_episode(
        self,
        env: Any,
        task_prompt: str,
        return_metrics: bool = False
    ) -> Union[Tuple[bool, float, int], Tuple[bool, float, int, float]]:
        """
        Executes a single closed-loop manipulation episode.
        
        Returns:
            success: Boolean indicator if task was completed.
            total_reward: Cumulative reward collected.
            steps_taken: Total execution steps until termination.
            (optional) mean_jerk: Average trajectory jerk norm.
        """
        # 1. Reset controllers and instantiate a fresh double-buffer queue
        obs = env.reset()
        self.gripper_filter.reset()
        self.buffer = AsyncActionBuffer(
            chunk_horizon=CONFIG.chunk_horizon,
            trigger_step=CONFIG.async_trigger_step,
            prefix_len=CONFIG.buffer_prefix_len,
            action_dim=CONFIG.action_dim
        )

        # Camera extrinsics (defaults from config if environment does not specify)
        t_cam = getattr(env, "t_base_cam", np.array([0.25, -0.35, 0.45], dtype=np.float32))

        # Initial cold-start inference chunk
        rgb_current = self._extract_rgb(obs)
        a_prev_cold = self.buffer.get_tail_prefix().to(self.device)

        with torch.no_grad():
            initial_chunk = self.policy.sample_actions(
                rgb=rgb_current,
                t_base_cam=t_cam,
                a_prev=a_prev_cold,
                prompt=task_prompt
            )

        self.buffer.load_initial_chunk(initial_chunk.squeeze(0))

        total_reward = 0.0
        success = False
        steps_taken = 0
        executed_actions = []

        # 2. Main Closed-Loop Control Loop
        for step in range(self.max_steps):
            steps_taken += 1

            # Retrieve next step action from queue
            action_norm, triggered, exhausted = self.buffer.step()
            raw_action = self.unnormalize_action(action_norm)

            # Apply Schmitt trigger deadband filter to gripper command (dim 6)
            grip_decision = self.gripper_filter.update(float(raw_action[6]))
            # Franka/Robosuite gripper bounds check: map [0, 1] to [-1, 1] if required
            if hasattr(env, "action_space") and env.action_space.low[6] < -0.1:
                raw_action[6] = 1.0 if grip_decision >= 0.5 else -1.0
            else:
                raw_action[6] = float(grip_decision)

            executed_actions.append(raw_action.copy())

            # Step environment with universal Gym / Gymnasium signature unpacking
            step_result = env.step(raw_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_result

            total_reward += float(reward)

            # Verification of task completion
            if bool(info.get("success", False)) or (hasattr(env, "check_success") and env.check_success()):
                success = True
                break
            if done:
                break

            # Asynchronous background inference trigger at Step 11
            if triggered:
                rgb_current = self._extract_rgb(obs)
                tail_prefix = self.buffer.get_tail_prefix().to(self.device)

                with torch.no_grad():
                    next_chunk = self.policy.sample_actions(
                        rgb=rgb_current,
                        t_base_cam=t_cam,
                        a_prev=tail_prefix,
                        prompt=task_prompt
                    )

                self.buffer.stage_next_chunk(next_chunk.squeeze(0))

        # 3. Trajectory Smoothness / Jerk Metric Calculation
        if len(executed_actions) >= 3:
            acts = np.array(executed_actions)[:, :3]  # Cartesian position deltas
            acc = np.diff(acts, n=1, axis=0)
            jerk = np.diff(acc, n=1, axis=0)
            self.last_jerk = float(np.mean(np.linalg.norm(jerk, axis=-1)))
        else:
            self.last_jerk = 0.0

        if return_metrics:
            return success, total_reward, steps_taken, self.last_jerk
        return success, total_reward, steps_taken