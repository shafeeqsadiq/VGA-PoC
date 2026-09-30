"""
controllers/async_buffer.py
Thread-safe asynchronous double-buffering queue manager for 50 Hz closed-loop control.
Supports background chunk staging, automatic step-16 rollover, and cold-start prefixes.
"""
from typing import Optional, Tuple, Union
import torch
import numpy as np
from configs.poc_config import CONFIG

class AsyncActionBuffer:
    def __init__(
        self,
        chunk_horizon: int = CONFIG.chunk_horizon,
        trigger_step: int = CONFIG.async_trigger_step,
        prefix_len: int = CONFIG.buffer_prefix_len,
        action_dim: int = CONFIG.action_dim):
        self.horizon = chunk_horizon
        self.trigger_step = trigger_step
        self.prefix_len = prefix_len
        self.action_dim = action_dim

        self.active_buffer: Optional[torch.Tensor] = None
        self.staged_chunk: Optional[torch.Tensor] = None
        self.current_idx: int = 0
        self.total_steps_executed: int = 0

    def load_initial_chunk(self, chunk: Union[torch.Tensor, np.ndarray]):
        """
        Initializes the queue with the very first action chunk of an episode.
        Args:
            chunk: Action chunk of shape [16, 7]
        """
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).float()

        assert chunk.shape == (self.horizon, self.action_dim), (
            f"Chunk shape mismatch: expected ({self.horizon}, {self.action_dim}), got {chunk.shape}"
        )
        self.active_buffer = chunk.clone()
        self.staged_chunk = None
        self.current_idx = 0
        self.total_steps_executed = 0

    def stage_next_chunk(self, chunk: Union[torch.Tensor, np.ndarray]):
        """
        Thread-safe deposit: stores newly computed trajectory from the background
        inference worker without interrupting the currently executing active chunk.
        """
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).float()

        assert chunk.shape == (self.horizon, self.action_dim), (
            f"Staged chunk shape mismatch: expected ({self.horizon}, {self.action_dim}), got {chunk.shape}"
        )
        self.staged_chunk = chunk.clone()

    def step(self) -> Tuple[torch.Tensor, bool, bool]:
        """
        Advances the robot control loop by 1 step.

        Returns:
            action: Action tensor [7] to send to robot controllers
            should_trigger: True if active step equals trigger_step (launches background thread)
            is_exhausted: True if active chunk has reached step 16 and handed over
        """
        if self.active_buffer is None:
            raise RuntimeError("Buffer empty: call load_initial_chunk() before stepping.")

        action = self.active_buffer[self.current_idx]
        self.current_idx += 1
        self.total_steps_executed += 1

        # Check if background inference should be triggered (Step 11 / 70% consumed)
        should_trigger = (self.current_idx == self.trigger_step)

        # Check if chunk boundary reached (Step 16)
        is_exhausted = False
        if self.current_idx >= self.horizon:
            is_exhausted = True
            # Seamless queue handover at step 16
            if self.staged_chunk is not None:
                self.active_buffer = self.staged_chunk
                self.staged_chunk = None
                self.current_idx = 0
            else:
                # Underrun warning: background thread did not return in time
                self.active_buffer = None

        return action, should_trigger, is_exhausted

    def get_tail_prefix(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Extracts tail waypoints (steps 13-16) for conditioning the next DiT chunk.
        Handles cold-start at t=0 by returning stationary zero actions.

        Returns:
            prefix: Tensor of shape [1, P, 6] (translational + rotational deltas)
        """
        if self.active_buffer is None:
            # Cold-start fallback: stationary prefix of zeros
            return torch.zeros(1, self.prefix_len, 6, device=device, dtype=dtype)

        # Slice steps 13..16 (translational + rotational components only, exclude gripper)
        tail_6d = self.active_buffer[-self.prefix_len:, :6].to(device=device, dtype=dtype)
        return tail_6d.unsqueeze(0)  # Shape [1, 4, 6]

    def reset(self):
        """Resets the queue for a new rollout episode."""
        self.active_buffer = None
        self.staged_chunk = None
        self.current_idx = 0
        self.total_steps_executed = 0