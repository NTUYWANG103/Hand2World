"""Camera helpers for WiLoR inference."""
import torch


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    """Crop-camera (bbox-relative) → full-image camera translation.
    ``cam_bbox`` (B, 3): (s, tx_bbox, ty_bbox). Returns (B, 3) camera translation."""
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - img_w / 2.) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - img_h / 2.) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)
