"""Image patch / affine helpers used by ``ViTDetDataset``."""
import cv2
import numpy as np


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Pad bbox to match target_aspect_ratio (w, h). ``input_shape``=(w, h)."""
    if target_aspect_ratio is None:
        return input_shape
    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)])
    return np.array([max(h * w_t / h_t, w), h])


def rotate_2d(pt_2d: np.ndarray, rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    x, y = pt_2d[0], pt_2d[1]
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.ndarray:
    """Affine transform: src bbox (center=c_x,c_y; size=src_*×scale; rotated rot°) →
    dst patch (dst_width × dst_height)."""
    src_w, src_h = src_width * scale, src_height * scale
    rot_rad = np.pi * rot / 180
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)
    src = np.stack([src_center, src_center + src_downdir, src_center + src_rightdir])
    dst = np.stack([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def generate_image_patch_cv2(img: np.ndarray, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0):
    """Crop + (optional) flip + (optional) rotate + resize. Returns (patch, trans)."""
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img.shape[1] - c_x - 1
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height,
                                     patch_width, patch_height, scale, rot)
    patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                            flags=cv2.INTER_LINEAR, borderMode=border_mode,
                            borderValue=border_value)
    # Alpha channel always uses BORDER_CONSTANT.
    if img.shape[2] == 4 and border_mode != cv2.BORDER_CONSTANT:
        patch[:, :, 3] = cv2.warpAffine(img[:, :, 3], trans,
                                         (int(patch_width), int(patch_height)),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT)
    return patch, trans


def convert_cvimg_to_tensor(cvimg: np.ndarray) -> np.ndarray:
    """OpenCV HWC uint8 → CHW float32."""
    return np.transpose(cvimg, (2, 0, 1)).astype(np.float32)
