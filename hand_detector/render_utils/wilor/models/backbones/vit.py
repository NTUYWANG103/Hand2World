"""ViT-H backbone with WiLoR pose / shape / camera regression tokens."""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from ...utils.geometry import aa_to_rotmat, rot6d_to_rotmat


def vit(cfg):
    return ViT(img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32,
                num_heads=16, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.55, cfg=cfg)


class DropPath(nn.Module):
    """Stochastic depth — no-op at inference (drop_prob ignored when ``training=False``)."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop)
        return self.proj_drop(self.proj(attn.transpose(1, 2).reshape(B, N, -1)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Image (B, C, H, W) → patch tokens (B, Hp*Wp, embed_dim)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio),
                             int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]),
                                    int(img_size[1] // patch_size[1]))
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                               stride=(patch_size[0] // ratio),
                               padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), (Hp, Wp)


class ViT(nn.Module):
    """WiLoR ViT-H backbone with pose/shape/cam regression tokens.
    Tokens layout: [pose_tokens (J+1), shape_token (1), cam_token (1), patch_tokens (Hp*Wp)].
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, ratio=1, last_norm=True, cfg=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)
        self.npose = npose
        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        self.register_buffer('init_cam', torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0))
        self.register_buffer('init_hand_pose', torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0))
        self.register_buffer('init_betas', torch.from_numpy(mean_params['shape'].astype(np.float32)).unsqueeze(0))

        self.pose_emb = nn.Linear(self.joint_rep_dim, embed_dim)
        self.shape_emb = nn.Linear(10, embed_dim)
        self.cam_emb = nn.Linear(3, embed_dim)
        self.decpose = nn.Linear(embed_dim, 6)
        self.decshape = nn.Linear(embed_dim, 10)
        self.deccam = nn.Linear(embed_dim, 3)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        # First slot of pos_embed is the (unused) class-token position; the
        # remaining positions match the pretraining patch layout.
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        J = self.cfg.MANO.NUM_HAND_JOINTS + 1
        pose_tokens = self.pose_emb(
            self.init_hand_pose.reshape(1, J, self.joint_rep_dim)
        ).repeat(B, 1, 1)
        shape_tokens = self.shape_emb(self.init_betas).unsqueeze(1).repeat(B, 1, 1)
        cam_tokens = self.cam_emb(self.init_cam).unsqueeze(1).repeat(B, 1, 1)
        x = torch.cat([pose_tokens, shape_tokens, cam_tokens, x], 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)

        pose_feat = x[:, :J]
        shape_feat = x[:, J:J + 1]
        cam_feat = x[:, J + 1:J + 2]

        pred_hand_pose = self.decpose(pose_feat).reshape(B, -1) + self.init_hand_pose
        pred_betas = self.decshape(shape_feat).reshape(B, -1) + self.init_betas
        pred_cam = self.deccam(cam_feat).reshape(B, -1) + self.init_cam

        pred_mano_feats = {'hand_pose': pred_hand_pose, 'betas': pred_betas, 'cam': pred_cam}
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(B, J, 3, 3)
        pred_mano_params = {
            'global_orient': pred_hand_pose[:, [0]],
            'hand_pose': pred_hand_pose[:, 1:],
            'betas': pred_betas,
        }
        img_feat = x[:, J + 2:].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
        return pred_mano_params, pred_cam, pred_mano_feats, img_feat

    def forward(self, x):
        return self.forward_features(x)
