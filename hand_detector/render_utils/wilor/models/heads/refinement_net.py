import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.geometry import aa_to_rotmat, perspective_projection, rot6d_to_rotmat


def _conv_block(in_c, out_c, *, kernel=3, stride=1, padding=1, bnrelu=True):
    layers = [nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding)]
    if bnrelu:
        layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
    return layers


def _deconv_block(in_c, out_c, *, bnrelu=True):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1,
                                  output_padding=0, bias=False)]
    if bnrelu:
        layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
    return layers


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        is_final = (i == len(feat_dims) - 2)
        layers += _conv_block(feat_dims[i], feat_dims[i + 1],
                              kernel=kernel, stride=stride, padding=padding,
                              bnrelu=(not is_final) or bnrelu_final)
    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        is_final = (i == len(feat_dims) - 2)
        layers += _deconv_block(feat_dims[i], feat_dims[i + 1],
                                 bnrelu=(not is_final) or bnrelu_final)
    return nn.Sequential(*layers)


def sample_joint_features(img_feat, joint_xy):
    """Bilinear-sample (B, C, H, W) feature at (B, J, 2) pixel coords → (B, J, C)."""
    H, W = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (W - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (H - 1) * 2 - 1
    grid = torch.stack((x, y), 2)[:, :, None, :]
    return F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0].permute(0, 2, 1).contiguous()


class DeConvNet(nn.Module):
    """Cascaded 2×-upsample deconv branches → multi-scale feature pyramid (high→low)."""

    def __init__(self, feat_dim=768, upscale=4):
        super().__init__()
        self.first_conv = make_conv_layers(
            [feat_dim, feat_dim // 2], kernel=1, stride=1, padding=0, bnrelu_final=False,
        )
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale)) + 1):
            dims = {
                0: [feat_dim // 2, feat_dim // 4],
                1: [feat_dim // 2, feat_dim // 4, feat_dim // 8],
                2: [feat_dim // 2, feat_dim // 4, feat_dim // 8, feat_dim // 8],
            }[i]
            self.deconv.append(make_deconv_layers(dims))

    def forward(self, img_feat):
        img_feat = self.first_conv(img_feat)
        feats = [img_feat] + [d(img_feat) for d in self.deconv]
        return feats[::-1]


class RefineNet(nn.Module):
    """Per-vertex feature sampling on the ViT-H pyramid → MANO param delta."""

    def __init__(self, cfg, feat_dim=1280, upscale=3):
        super().__init__()
        self.deconv = DeConvNet(feat_dim=feat_dim, upscale=upscale)
        self.out_dim = feat_dim // 8 + feat_dim // 4 + feat_dim // 2
        self.dec_pose = nn.Linear(self.out_dim, 96)
        self.dec_cam = nn.Linear(self.out_dim, 3)
        self.dec_shape = nn.Linear(self.out_dim, 10)

        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]

    def forward(self, img_feat, verts_3d, pred_cam, pred_mano_feats, focal_length):
        B = img_feat.shape[0]
        img_feats = self.deconv(img_feat)
        sizes = [f.shape[2] for f in img_feats]
        temp_cams = [
            torch.stack([pred_cam[:, 1], pred_cam[:, 2],
                          2 * focal_length[:, 0] / (s * pred_cam[:, 0] + 1e-9)], dim=-1)
            for s in sizes
        ]
        verts_2d = [
            perspective_projection(verts_3d, translation=temp_cams[i],
                                    focal_length=focal_length / sizes[i])
            for i in range(len(sizes))
        ]
        vert_feats = torch.cat([
            sample_joint_features(img_feats[i], verts_2d[i]).max(1).values
            for i in range(len(sizes))
        ], dim=-1)

        pred_hand_pose = pred_mano_feats['hand_pose'] + self.dec_pose(vert_feats)
        pred_betas = pred_mano_feats['betas'] + self.dec_shape(vert_feats)
        pred_cam = pred_mano_feats['cam'] + self.dec_cam(vert_feats)

        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(
            B, self.cfg.MANO.NUM_HAND_JOINTS + 1, 3, 3,
        )
        pred_mano_params = {
            'global_orient': pred_hand_pose[:, [0]],
            'hand_pose': pred_hand_pose[:, 1:],
            'betas': pred_betas,
        }
        return pred_mano_params, pred_cam
