"""Camera-pose → Plucker embedding for the Wan 2.2 camera adapter (DA3 JSON input).
Adapted from https://github.com/hehao13/CameraCtrl/blob/main/inference.py.
"""
import json

import numpy as np
import torch


def channel_stack_plucker_to_latent(control_camera_video: torch.Tensor) -> torch.Tensor:
    """Pixel-temporal Plucker (B, 6, F_pix, H, W) → latent-temporal channel-stack
    (B, 24, F_lat, H, W) with F_lat = F_pix // 4. Frame 0 is repeated 4× so the
    leading pad satisfies the F%4==1 → F_lat constraint.
    """
    cc = torch.cat([
        torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
        control_camera_video[:, :, 1:],
    ], dim=2).transpose(1, 2)                                        # (B, F_pix, 6, H, W)
    b, f, c, h, w = cc.shape
    cc = cc.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    return cc.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)


def ray_condition(K, c2w, H, W, device):
    """K [B, F, 4] (fx,fy,cx,cy), c2w [B, F, 4, 4] → Plucker [B, F, H, W, 6]."""
    B = K.shape[0]
    j, i = torch.meshgrid(
        torch.arange(H, device=device, dtype=c2w.dtype),
        torch.arange(W, device=device, dtype=c2w.dtype),
        indexing="ij",
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.to(device).chunk(4, dim=-1)
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    directions = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    c2w_d = c2w.to(device)
    rays_d = directions @ c2w_d[..., :3, :3].transpose(-1, -2)
    rays_o = c2w_d[..., :3, 3][:, :, None].expand_as(rays_d)
    plucker = torch.cat([torch.linalg.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
    return plucker.reshape(B, c2w.shape[1], H, W, 6)


def process_pose_json(camera, width=672, height=384, device="cpu", frame_indices=None):
    """DA3 camera JSON → Plucker embedding ``[T, H, W, 6]``. ``camera`` may be a path
    or a pre-loaded DA3 dict (``{image_width, image_height, frames}``).
    """
    if isinstance(camera, dict):
        data = camera
    else:
        with open(camera, "r") as f:
            data = json.load(f)
    ow, oh = data["image_width"], data["image_height"]
    frames = data["frames"]
    if frame_indices is not None:
        frames = [frames[i] for i in frame_indices]

    intrinsics, c2ws = [], []
    for fr in frames:
        intr = fr["intrinsics"]
        if isinstance(intr, dict):
            fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        else:
            fx, fy, cx, cy = intr
        intrinsics.append((fx / ow, fy / oh, cx / ow, cy / oh))
        c2w = np.eye(4)
        c2w[:3, :] = np.hstack([np.array(fr["rotation"]).reshape(3, 3),
                                 np.array(fr["translation"]).reshape(3, 1)])
        c2ws.append(c2w)

    # Aspect-preserve K rescale before pixel scaling.
    pose_wh, sample_wh = ow / oh, width / height
    scale_x, scale_y = ((height * pose_wh) / width, 1.0) if pose_wh > sample_wh \
                       else (1.0, (width / pose_wh) / height)

    K = torch.as_tensor(
        [[fx * scale_x * width, fy * scale_y * height, cx * width, cy * height]
         for fx, fy, cx, cy in intrinsics],
        dtype=torch.float32,
    )[None]
    # Relative pose: first frame is identity, rest = w2c[0] @ c2w[i].
    w2c0 = np.linalg.inv(c2ws[0])
    rel_c2ws = np.stack([np.eye(4)] + [w2c0 @ c for c in c2ws[1:]], axis=0).astype(np.float32)
    c2ws_t = torch.as_tensor(rel_c2ws)[None]
    return ray_condition(K, c2ws_t, height, width, device=device)[0].cpu()       # (F, H, W, 6)
