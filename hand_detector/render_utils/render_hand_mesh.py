#!/usr/bin/env python3
"""MANO hand mesh rasteriser via nvdiffrast.

Used by ``detect_hand_mesh.WiLoRPipeline`` to render hand meshes onto frames.
Modes: ``wireframe`` / ``xray`` / ``solid`` / ``joint``.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple


# ── Mesh Renderer (nvdiffrast) ─────────────────────────────────────────────

class MeshRenderer:
    """
    Rasterise MANO hand meshes with nvdiffrast.

    Render modes:
      wireframe – see-through wireframe via DepthPeeler + barycentric edge detection.
      xray      – semi-transparent "glass hand" via DepthPeeler + alpha compositing.
      solid     – fully opaque Phong-shaded mesh (standard z-buffer).
      joint     – semi-transparent 3D skeleton (joints + bones, cv2-based).
    """

    def __init__(
        self,
        faces: np.ndarray,
        render_mode: str = "wireframe",
        mesh_alpha: float = 1.0,
        wireframe_thickness: int = 1,
        wireframe_color: Tuple[int, int, int] = (255, 255, 255),
        max_depth_layers: int = 3,
    ):
        """
        Args:
            faces: MANO face array (F, 3).
            render_mode: "wireframe", "xray", or "solid".
            mesh_alpha: Front layer opacity (0–1) for xray/wireframe modes.
            wireframe_thickness: Controls barycentric edge threshold (1=thin, 2=medium, 3=thick).
            wireframe_color: BGR color for wireframe edges.
            max_depth_layers: Max depth peeling iterations for wireframe/xray modes.
        """
        import nvdiffrast.torch as dr
        self._faces = faces
        self._render_mode = render_mode
        self._mesh_alpha = mesh_alpha
        self._wireframe_thickness = wireframe_thickness
        self._wireframe_color = wireframe_color
        self._max_depth_layers = max_depth_layers
        self._dr_ctx = dr.RasterizeCudaContext()

        # Barycentric edge threshold: thickness 1→0.02, 2→0.04, 3→0.06
        self._edge_threshold = wireframe_thickness * 0.02

    @staticmethod
    def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        vn = np.zeros_like(verts)
        for i in range(3):
            np.add.at(vn, faces[:, i], fn)
        norm = np.linalg.norm(vn, axis=-1, keepdims=True) + 1e-8
        return (vn / norm).astype(np.float32)

    def render(
        self,
        verts_list: List[np.ndarray],
        cam_t_list: List[np.ndarray],
        is_right_list: List[int],
        H: int, W: int,
        intrinsics: dict,
        apply_hamer_rotation: bool = True,
        camera_rotation=None,
        camera_translation=None,
    ) -> np.ndarray:
        if len(verts_list) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        return self._render_nvdiffrast(
            verts_list, cam_t_list, is_right_list, H, W,
            intrinsics, apply_hamer_rotation, camera_rotation, camera_translation,
        )

    @torch.no_grad()
    def render_xray_batch(
        self,
        per_frame_data: List[dict],     # each: {"verts": List[np.ndarray], "cam_t": List, "is_right": List}
        H: int, W: int,
        intrinsics: dict,
        apply_hamer_rotation: bool = False,
        camera_rotation=None,
        camera_translation=None,
        max_hands_per_frame: int = 2,
    ) -> np.ndarray:
        """Batched xray render via nvdiffrast DepthPeeler.

        All frames pass through one nvdiffrast call. Frames with fewer hands
        than ``max_hands_per_frame`` get padded slots whose vertices are placed
        OUTSIDE the [-1,1]^3 clip cube so nvdiffrast clips them for free.

        Returns ``(N, H, W, 3)`` uint8 BGR.
        """
        import nvdiffrast.torch as dr

        N = len(per_frame_data)
        if N == 0:
            return np.zeros((0, H, W, 3), dtype=np.uint8)

        V_PER_HAND = int(self._faces.max()) + 1                     # 778
        F_PER_HAND = int(self._faces.shape[0])                       # 1538
        V_TOTAL = max_hands_per_frame * V_PER_HAND
        near, far = 0.01, 100.0
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        # Shared face topology (right-hand winding for ALL hand slots).
        # Phong uses .abs() of n·l so left-hand normals computed from this winding
        # shade with correct magnitude even though the sign flips.
        face_arr = np.concatenate(
            [self._faces + hi * V_PER_HAND for hi in range(max_hands_per_frame)],
            axis=0,
        ).astype(np.int32)
        face_t = torch.from_numpy(face_arr).cuda().contiguous()

        clips_np = np.empty((N, V_TOTAL, 4), dtype=np.float32)
        clips_np[:] = (2.0, 2.0, 0.0, 1.0)                           # outside-frustum default
        normals_np = np.zeros((N, V_TOTAL, 3), dtype=np.float32)
        colors_np = np.zeros((N, V_TOTAL, 3), dtype=np.float32)

        # MANO base colors (BGR): right=red, left=blue.
        color_right = np.array([0.0, 0.0, 255.0], dtype=np.float32)
        color_left  = np.array([255.0, 0.0, 0.0], dtype=np.float32)

        for fi, fd in enumerate(per_frame_data):
            verts_list = fd.get("verts") or []
            cam_t_list = fd.get("cam_t") or []
            is_right_list = fd.get("is_right") or []
            n_hands = min(len(verts_list), max_hands_per_frame)
            for hi in range(n_hands):
                v = verts_list[hi]
                ct = cam_t_list[hi]
                ir = bool(is_right_list[hi])

                vt = v.astype(np.float64) + ct.astype(np.float64)
                if camera_rotation is not None and camera_translation is not None:
                    vt = camera_rotation @ vt.T + camera_translation.reshape(3, 1)
                    vt = vt.T

                vn = self._compute_vertex_normals(vt.astype(np.float32), self._faces)

                if apply_hamer_rotation:
                    x, y, z = vt[:, 0], -vt[:, 1], -vt[:, 2]
                    vn[:, 1] = -vn[:, 1]
                    vn[:, 2] = -vn[:, 2]
                else:
                    x, y, z = vt[:, 0], vt[:, 1], vt[:, 2]

                nx = 2.0 * fx * x / (z * W) + 2.0 * cx / W - 1.0
                ny = -2.0 * fy * y / (z * H) - 2.0 * cy / H + 1.0
                nz = (far + near) / (far - near) - 2.0 * far * near / (z * (far - near))
                clip_xyzw = np.stack([nx * z, ny * z, nz * z, z], axis=-1).astype(np.float32)

                v0 = hi * V_PER_HAND
                v1 = v0 + V_PER_HAND
                clips_np[fi, v0:v1, :] = clip_xyzw
                normals_np[fi, v0:v1, :] = vn.astype(np.float32)
                colors_np[fi, v0:v1, :] = color_right if ir else color_left

        clip_t = torch.from_numpy(clips_np).cuda().contiguous()      # [N, V_TOTAL, 4]
        normals_t = torch.from_numpy(normals_np).cuda().contiguous()  # [N, V_TOTAL, 3]
        colors_t = torch.from_numpy(colors_np).cuda().contiguous()    # [N, V_TOTAL, 3]

        # Batched DepthPeeler. Each rasterize_next_layer returns rast = [N, H, W, 4].
        layers = []
        with dr.DepthPeeler(self._dr_ctx, clip_t, face_t, resolution=[H, W]) as peeler:
            for _ in range(self._max_depth_layers):
                rast = peeler.rasterize_next_layer()[0]
                rast = rast.flip(1)                                    # match per-frame's H-flip
                tri = rast[..., 3].int()                               # [N, H, W]
                if (tri == 0).all():
                    break

                normals_interp, _ = dr.interpolate(normals_t, rast, face_t)        # [N, H, W, 3]
                normals_interp = normals_interp / (
                    normals_interp.norm(dim=-1, keepdim=True) + 1e-8
                )
                colors_interp, _ = dr.interpolate(colors_t, rast, face_t)          # [N, H, W, 3]

                mask = tri > 0                                                      # [N, H, W]
                light = torch.tensor([0.0, 0.0, 1.0], device="cuda")
                ndotl = (normals_interp * light).sum(dim=-1).abs()                 # [N, H, W]
                intensity = (0.3 + 0.7 * ndotl).clamp(0, 1).unsqueeze(-1)          # [N, H, W, 1]

                ndoth = (normals_interp * light).sum(dim=-1).abs()
                spec = (torch.pow(ndoth, 32.0) * 0.4 * 255.0).unsqueeze(-1)        # [N, H, W, 1]

                img = colors_interp * intensity + spec                              # [N, H, W, 3]
                img = img.clamp(0, 255)
                img = img * mask.unsqueeze(-1).to(img.dtype)
                layers.append((img, mask))

        if not layers:
            return np.zeros((N, H, W, 3), dtype=np.uint8)

        # Composite back-to-front, additive blend (mirrors per-frame xray weights).
        result = torch.zeros(N, H, W, 3, dtype=torch.float32, device="cuda")
        n_layers = len(layers)
        for li, (layer_img, layer_mask) in enumerate(reversed(layers)):
            depth_order = n_layers - 1 - li
            weight = self._mesh_alpha * max(0.18, 0.55 - depth_order * 0.10)
            mexp = layer_mask.unsqueeze(-1).expand_as(layer_img)
            result = torch.where(mexp, result + layer_img * weight, result)

        return result.clamp(0, 255).byte().cpu().numpy()

    def _phong_shade_layer(self, rast, normals_t, face_t, cidx_t, H, W):
        """Apply Phong + Blinn-Phong shading to a rasterized layer. Returns (img, mask)."""
        import nvdiffrast.torch as dr
        normals_interp, _ = dr.interpolate(normals_t, rast, face_t)
        normals_interp = normals_interp[0]
        normals_interp = normals_interp / (normals_interp.norm(dim=-1, keepdim=True) + 1e-8)

        tri = rast[0, :, :, 3].int()
        mask = tri > 0

        light_dir = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        ndotl = torch.sum(normals_interp * light_dir, dim=-1).abs()
        intensity = (0.3 + 0.7 * ndotl).clamp(0, 1)

        base_colors = torch.tensor(
            [[0, 0, 255], [255, 0, 0]], dtype=torch.float32, device="cuda"
        )  # right=red, left=blue (BGR)

        img = torch.zeros(H, W, 3, dtype=torch.float32, device="cuda")
        if mask.any():
            face_color_idx = cidx_t[tri[mask] - 1]
            base = base_colors[face_color_idx]
            shade = intensity[mask].unsqueeze(-1)
            img[mask] = (base * shade).clamp(0, 255)

            half_vec = torch.tensor([0.0, 0.0, 1.0], device="cuda")
            ndoth = torch.sum(normals_interp * half_vec, dim=-1).abs()
            spec = torch.pow(ndoth, 32.0) * 0.4
            img[mask] = (img[mask] + spec[mask].unsqueeze(-1) * 255).clamp(0, 255)

        return img, mask

    def _barycentric_edge_mask(self, rast):
        """Detect triangle edges from barycentric coordinates. Returns bool mask (H, W)."""
        u = rast[0, :, :, 0]
        v = rast[0, :, :, 1]
        w = 1.0 - u - v
        min_bary = torch.min(torch.stack([u, v, w], dim=-1), dim=-1).values
        tri = rast[0, :, :, 3].int()
        return (min_bary < self._edge_threshold) & (tri > 0)

    # OpenPose 21-joint hand-skeleton edges, identical ordering to DWPose's
    # `draw_handpose` (annotator/dwpose/util.py).
    DWPOSE_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # index
        (0, 9), (9, 10), (10, 11), (11, 12),    # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    ]

    def render_dwpose(
        self,
        joints_list: List[np.ndarray],
        cam_t_list: List[np.ndarray],
        is_right_list: List[int],
        H: int, W: int,
        intrinsics: dict,
        apply_hamer_rotation: bool = False,
        bone_thickness: int = 2,
        joint_radius: int = 4,
    ) -> np.ndarray:
        """DWPose-style 21-joint hand skeleton: HSV-cycle bones + red joint dots.

        Mirrors ``annotator/dwpose/util.py::draw_handpose`` in DWPose. Each
        of the 20 hand bones gets a unique hue from an HSV color wheel; all
        21 joints are drawn as filled red circles. Output is a uint8 BGR
        canvas on a black background (drop-in for ``render`` in the pipeline).

        ``joints_list`` is expected in OpenPose 21-joint order (the
        canonical output of HaMeR / WiLoR / WildHands). Joints are projected
        with the pinhole intrinsics (no NDC); for left hands the input
        skeleton is assumed to already be in the proper image-camera frame
        (the WiLoR pipeline pre-mirrors x for the left hand the same way it
        does for vertices).
        """
        try:
            import matplotlib.colors as mcolors
        except ImportError as exc:  # pragma: no cover
            raise ImportError("render_dwpose requires matplotlib") from exc

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if not joints_list:
            return canvas

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        scale_factor = max(H, W) / 480.0
        bt = max(1, int(round(bone_thickness * scale_factor)))
        jr = max(1, int(round(joint_radius * scale_factor)))

        # Pre-compute the HSV bone palette (identical math to DWPose).
        # Stored as BGR so a cv2.imwrite-saved PNG renders the intended hue
        # wheel under any normal viewer.
        n_edges = len(self.DWPOSE_EDGES)
        bone_bgr = []
        for ie in range(n_edges):
            r, g, b = mcolors.hsv_to_rgb([ie / float(n_edges), 1.0, 1.0])
            bone_bgr.append((int(b * 255), int(g * 255), int(r * 255)))

        for joints, ct, ir in zip(joints_list, cam_t_list, is_right_list):
            j3d = joints.astype(np.float64) + ct.astype(np.float64)  # (21, 3) cam frame
            if apply_hamer_rotation:
                x, y, z = j3d[:, 0], -j3d[:, 1], -j3d[:, 2]
            else:
                x, y, z = j3d[:, 0], j3d[:, 1], j3d[:, 2]

            valid = z > 0.05  # in front of camera + finite depth
            pts_2d = np.stack([fx * x / z + cx, fy * y / z + cy], axis=-1).astype(np.int32)

            for ie, (i1, i2) in enumerate(self.DWPOSE_EDGES):
                if i1 >= len(pts_2d) or i2 >= len(pts_2d):
                    continue
                if not (valid[i1] and valid[i2]):
                    continue
                cv2.line(canvas, tuple(pts_2d[i1]), tuple(pts_2d[i2]),
                         bone_bgr[ie], thickness=bt, lineType=cv2.LINE_AA)

            for ji in range(min(21, len(pts_2d))):
                if not valid[ji]:
                    continue
                # DWPose's joint color: BGR (0, 0, 255) = pure red.
                cv2.circle(canvas, tuple(pts_2d[ji]), jr, (0, 0, 255),
                           thickness=-1, lineType=cv2.LINE_AA)

        return canvas

    def _render_nvdiffrast(self, verts_list, cam_t_list, is_right_list, H, W,
                           intrinsics, apply_hamer_rotation, camera_rotation, camera_translation):
        import nvdiffrast.torch as dr

        faces_r = self._faces
        faces_l = faces_r[:, [0, 2, 1]]
        near, far = 0.01, 100.0

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        clips, face_arr, cidx, all_normals = [], [], [], []
        hand_proj_2d, hand_faces_local, hand_is_right = [], [], []  # for cv2 wireframe
        self._hand_z_vals = []  # z-values per hand for near-plane clipping
        off = 0
        for v, ct, ir in zip(verts_list, cam_t_list, is_right_list):
            vt = v.astype(np.float64) + ct.astype(np.float64)

            if camera_rotation is not None and camera_translation is not None:
                vt = camera_rotation @ vt.T + camera_translation.reshape(3, 1)
                vt = vt.T

            f = faces_r if ir else faces_l
            vn = self._compute_vertex_normals(vt.astype(np.float32), f)

            if apply_hamer_rotation:
                x, y, z = vt[:, 0], -vt[:, 1], -vt[:, 2]
                vn[:, 1] = -vn[:, 1]
                vn[:, 2] = -vn[:, 2]
            else:
                x, y, z = vt[:, 0], vt[:, 1], vt[:, 2]

            # 2D projection + z-values for cv2 wireframe overlay (near-plane clipping).
            hand_proj_2d.append(np.stack([fx * x / z + cx, fy * y / z + cy], axis=-1).astype(np.float32))
            hand_faces_local.append(f.copy())
            hand_is_right.append(ir)
            self._hand_z_vals.append(np.array(z))

            nx = 2.0 * fx * x / (z * W) + 2.0 * cx / W - 1.0
            ny = -2.0 * fy * y / (z * H) - 2.0 * cy / H + 1.0
            nz = (far + near) / (far - near) - 2.0 * far * near / (z * (far - near))
            clips.append(np.stack([nx * z, ny * z, nz * z, z], axis=-1).astype(np.float32))

            face_arr.append(f + off)
            cidx.extend([0 if ir else 1] * len(f))
            all_normals.append(vn)
            off += len(v)

        clip_t = torch.from_numpy(np.concatenate(clips)).cuda().unsqueeze(0).contiguous()
        face_t = torch.from_numpy(np.concatenate(face_arr).astype(np.int32)).cuda().contiguous()
        cidx_t = torch.tensor(cidx, dtype=torch.long, device="cuda")
        normals_t = torch.from_numpy(np.concatenate(all_normals)).cuda().unsqueeze(0).contiguous()

        # ── Solid mode: single pass, standard z-buffer ──
        if self._render_mode == "solid":
            rast, _ = dr.rasterize(self._dr_ctx, clip_t, face_t, resolution=[H, W])
            rast = rast.flip(1)
            img, _ = self._phong_shade_layer(rast, normals_t, face_t, cidx_t, H, W)
            return img.clamp(0, 255).byte().cpu().numpy()

        # ── Wireframe: solid Phong pass + sub-pixel-thin line overlay ──
        # Draw cv2.polylines at 2x resolution with thickness=1 AA, then
        # INTER_AREA-downsample. The ~0.5 px effective line width keeps the
        # bright Phong solid visible between edges on dense MANO regions.
        # cv2 has no z-test, so edges of back-facing / occluded triangles
        # show through the front surface — intentional for the wireframe look.
        if self._render_mode == "wireframe":
            rast, _ = dr.rasterize(self._dr_ctx, clip_t, face_t, resolution=[H, W])
            rast = rast.flip(1)
            img, _ = self._phong_shade_layer(rast, normals_t, face_t, cidx_t, H, W)
            result = img.clamp(0, 255).byte().cpu().numpy()

            scale = 2
            hi_W, hi_H = W * scale, H * scale
            hi_result = cv2.resize(result, (hi_W, hi_H), interpolation=cv2.INTER_LINEAR)
            line_color = (60, 60, 60)  # dark gray
            thickness = max(1, self._wireframe_thickness)
            z_vals_list = getattr(self, '_hand_z_vals', [None] * len(hand_proj_2d))
            for pts_2d, faces_local, ir, z_vals in zip(hand_proj_2d, hand_faces_local, hand_is_right, z_vals_list):
                if z_vals is not None:
                    face_z = z_vals[faces_local]
                    valid_faces = np.all(face_z > 0.05, axis=1)
                    faces_local = faces_local[valid_faces]
                if len(faces_local) == 0:
                    continue
                triangles_hi = (pts_2d[faces_local] * scale).astype(np.int32)
                cv2.polylines(hi_result, list(triangles_hi), isClosed=True,
                              color=line_color, thickness=thickness, lineType=cv2.LINE_AA)
            self._hand_z_vals = []
            return cv2.resize(hi_result, (W, H), interpolation=cv2.INTER_AREA)

        # ── X-ray: semi-transparent hand via DepthPeeler + alpha compositing ──
        # Per-frame path. ``render_xray_batch`` is the batched equivalent used
        # by the pipeline; this branch handles direct ``render()`` calls and
        # any non-{solid, wireframe, dwpose} mode such as ``joint``.
        layers = []
        with dr.DepthPeeler(self._dr_ctx, clip_t, face_t, resolution=[H, W]) as peeler:
            for _ in range(self._max_depth_layers):
                rast, _ = peeler.rasterize_next_layer()
                rast = rast.flip(1)
                tri = rast[0, :, :, 3].int()
                if (tri == 0).all():
                    break
                img, mask = self._phong_shade_layer(rast, normals_t, face_t, cidx_t, H, W)
                layers.append((img, mask))

        if not layers:
            return np.zeros((H, W, 3), dtype=np.uint8)

        # Back-to-front additive blend: front layer brighter, back layers dimmer.
        result = torch.zeros(H, W, 3, dtype=torch.float32, device="cuda")
        n_layers = len(layers)
        for li, (layer_img, layer_mask) in enumerate(reversed(layers)):
            depth_order = n_layers - 1 - li
            weight = self._mesh_alpha * max(0.18, 0.55 - depth_order * 0.10)
            result[layer_mask] = result[layer_mask] + layer_img[layer_mask] * weight
        return result.clamp(0, 255).byte().cpu().numpy()

