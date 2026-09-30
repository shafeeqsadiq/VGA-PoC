"""
losses/kinematics.py
Differentiable SO(3) Lie Algebra conversions, Rodrigues rotation transformations,
and relative geodesic distance metrics for kinematic trajectory optimization.
Uses double-precision intermediate accumulation to guarantee sub-1e-6 round-trip precision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle representation [..., 3] to rotation matrices [..., 3, 3]
    via Rodrigues' formula with high-precision Taylor expansions near zero.
    """
    orig_dtype = axis_angle.dtype
    orig_device = axis_angle.device
    shape = axis_angle.shape[:-1]

    v = axis_angle.reshape(-1, 3).to(torch.float64)
    theta_sq = torch.sum(v ** 2, dim=-1, keepdim=True)
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-16))

    # Skew-symmetric cross product matrix K = [v]_x
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    zeros = torch.zeros_like(vx)

    K = torch.stack([
        zeros, -vz, vy,
        vz, zeros, -vx,
        -vy, vx, zeros
    ], dim=-1).view(-1, 3, 3)

    I = torch.eye(3, dtype=torch.float64, device=orig_device).unsqueeze(0)

    # Taylor expansion near zero:
    # sin(theta)/theta = 1 - theta^2 / 6 + theta^4 / 120
    # (1 - cos(theta))/theta^2 = 1/2 - theta^2 / 24 + theta^4 / 720
    small_angle = (theta_sq < 1e-8)

    c1 = torch.where(
        small_angle,
        1.0 - theta_sq / 6.0 + (theta_sq ** 2) / 120.0,
        torch.sin(theta) / torch.clamp(theta, min=1e-12)
    ).unsqueeze(-1)

    c2 = torch.where(
        small_angle,
        0.5 - theta_sq / 24.0 + (theta_sq ** 2) / 720.0,
        (1.0 - torch.cos(theta)) / torch.clamp(theta_sq, min=1e-12)
    ).unsqueeze(-1)

    R = I + c1 * K + c2 * torch.bmm(K, K)
    return R.to(orig_dtype).view(*shape, 3, 3)

def safe_rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices [..., 3, 3] to axis-angle vectors [..., 3]
    via well-conditioned atan2 matrix logarithm.
    """
    orig_dtype = R.dtype
    orig_device = R.device
    shape = R.shape[:-2]

    R_64 = R.reshape(-1, 3, 3).to(torch.float64)

    # Skew-symmetric part: (R - R^T) / 2
    diff = R_64 - R_64.transpose(1, 2)
    v_skew = torch.stack([
        diff[:, 2, 1],
        diff[:, 0, 2],
        diff[:, 1, 0]
    ], dim=-1) * 0.5  # Shape [N, 3]

    sin_theta = torch.norm(v_skew, p=2, dim=-1, keepdim=True)
    tr = R_64[:, 0, 0] + R_64[:, 1, 1] + R_64[:, 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).unsqueeze(-1)

    # Well-conditioned angle via atan2
    theta = torch.atan2(sin_theta, cos_theta)

    small_angle = (theta < 1e-6)
    scale = torch.where(
        small_angle,
        1.0 + (theta ** 2) / 6.0 + 7.0 * (theta ** 4) / 360.0,
        theta / torch.clamp(sin_theta, min=1e-12)
    )

    axis_angle = v_skew * scale

    # Handling theta near pi (singularity where sin(theta) -> 0 and R - R^T vanishes)
    near_pi = (theta > (torch.pi - 1e-3)).squeeze(-1)
    if torch.any(near_pi):
        R_pi = R_64[near_pi]
        diag = torch.diagonal(R_pi, dim1=1, dim2=2) + 1.0
        diag = torch.clamp(diag * 0.5, min=0.0)

        max_idx = torch.argmax(diag, dim=-1)
        v_pi = torch.zeros_like(R_pi[:, :, 0])
        for i in range(R_pi.shape[0]):
            m = max_idx[i].item()
            v_pi[i, m] = torch.sqrt(diag[i, m])
            v_m = v_pi[i, m]
            if v_m > 1e-6:
                for j in range(3):
                    if j != m:
                        v_pi[i, j] = (R_pi[i, m, j] + R_pi[i, j, m]) / (4.0 * v_m)

        theta_pi = theta[near_pi]
        axis_angle[near_pi] = F.normalize(v_pi, p=2, dim=-1) * theta_pi

    return axis_angle.to(orig_dtype).view(*shape, 3)

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternions [..., 4] (w, x, y, z) to rotation matrices [..., 3, 3].
    """
    orig_dtype = q.dtype
    shape = q.shape[:-1]
    q_norm = F.normalize(q.reshape(-1, 4).to(torch.float64), p=2, dim=-1)
    w, x, y, z = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

    r00 = 1.0 - 2.0 * (y ** 2 + z ** 2)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x ** 2 + z ** 2)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x ** 2 + y ** 2)

    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).view(-1, 3, 3)

    return R.to(orig_dtype).view(*shape, 3, 3)

def so3_relative_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Computes geodesic distance on SO(3) manifold: theta = arccos((tr(R1^T * R2) - 1) / 2)
    Uses atan2 in float64 to prevent boundary NaN or gradient explosion.
    """
    orig_dtype = R1.dtype
    R1_64 = R1.to(torch.float64)
    R2_64 = R2.to(torch.float64)
    R_rel = torch.matmul(R1_64.transpose(-1, -2), R2_64)

    diff = R_rel - R_rel.transpose(-1, -2)
    v_skew = torch.stack([
        diff[..., 2, 1],
        diff[..., 0, 2],
        diff[..., 1, 0]
    ], dim=-1) * 0.5

    sin_theta = torch.norm(v_skew, p=2, dim=-1)
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = (tr - 1.0) * 0.5

    theta = torch.atan2(sin_theta, cos_theta)
    return theta.to(orig_dtype)