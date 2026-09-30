"""
controllers/schmitt_trigger.py
Hysteresis deadband filter preventing high-frequency gripper fluttering.
Robust to floats, NumPy arrays, and scalar/batched PyTorch tensors.
"""
from typing import Any
import numpy as np
import torch
from configs.poc_config import CONFIG

class SchmittTriggerGripper:
    def __init__(
        self,
        low_thresh: float = CONFIG.schmitt_low,
        high_thresh: float = CONFIG.schmitt_high,
        initial_state: float = 0.0):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.initial_state = float(initial_state)
        self.current_state = float(initial_state)

    def update(self, raw_gripper_act: Any) -> float:
        """
        Maps continuous [-1, 1] output to affine [0, 1], then applies deadband.

        Args:
            raw_gripper_act: Gripper command (float, torch.Tensor, or np.ndarray)
        Returns:
            Filtered discrete state: 0.0 (open) or 1.0 (closed)
        """
        # Safely extract scalar value across all input types
        if isinstance(raw_gripper_act, torch.Tensor):
            val = float(raw_gripper_act.detach().cpu().flatten()[0].item())
        elif isinstance(raw_gripper_act, np.ndarray):
            val = float(raw_gripper_act.flatten()[0])
        else:
            val = float(raw_gripper_act)

        # Affine mapping: [-1.0, 1.0] -> [0.0, 1.0] with safety clamp
        g_bar = min(max((val + 1.0) / 2.0, 0.0), 1.0)

        # Apply hysteresis deadband
        if g_bar > self.high_thresh:
            self.current_state = 1.0
        elif g_bar < self.low_thresh:
            self.current_state = 0.0

        return self.current_state

    def reset(self, state: float = None):
        """Resets trigger state to initial configuration or specified value."""
        if state is not None:
            self.current_state = float(state)
        else:
            self.current_state = self.initial_state