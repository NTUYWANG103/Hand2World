from typing import Optional

import torch
import torch.nn.functional as F


def aa_to_rotmat(theta: torch.Tensor) -> torch.Tensor:
    """Axis-angle (B, 3) → rotation matrix (B, 3, 3) via quaternion."""
    norm = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)
    angle = norm * 0.5
    quat = torch.cat([torch.cos(angle), torch.sin(angle) * theta / norm], dim=1)
    quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz,  2 * wy + 2 * xz,
        2 * wz + 2 * xy,   w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy,   2 * wx + 2 * yz,  w2 - x2 - y2 + z2,
    ], dim=1).view(B, 3, 3)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """6D rotation (B, 6) → rotation matrix (B, 3, 3)
    (Zhou et al., "On the Continuity of Rotation Representations in NN", CVPR 2019)."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Project (B, N, 3) world points through pinhole camera → (B, N, 2) pixel coords."""
    B = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(B, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(B, 2, device=points.device, dtype=points.dtype)
    K = torch.zeros([B, 3, 3], device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center
    points = torch.einsum('bij,bkj->bki', rotation, points) + translation.unsqueeze(1)
    points = points / points[:, :, -1].unsqueeze(-1)
    return torch.einsum('bij,bkj->bki', K, points)[:, :, :-1]
