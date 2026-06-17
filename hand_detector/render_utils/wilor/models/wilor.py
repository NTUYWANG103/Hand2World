"""WiLoR LightningModule (inference only)."""
import pytorch_lightning as pl
import torch
from typing import Dict
from yacs.config import CfgNode

from ..utils.geometry import perspective_projection
from .backbones import create_backbone
from .heads import RefineNet
from . import MANO


class WiLoR(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """``init_renderer`` is accepted for API compatibility and ignored;
        no internal renderer is constructed."""
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            self.backbone.load_state_dict(
                torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'],
                strict=False,
            )
        self.refine_net = RefineNet(cfg, feat_dim=1280, upscale=3)

        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # ActNorm-init buffer (required for checkpoint state_dict compatibility).
        self.register_buffer('initialized', torch.tensor(False))
        self.renderer = None
        self.mesh_renderer = None
        self.automatic_optimization = False

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """ViT-H backbone + RefineNet head → MANO params + projected 2D keypoints."""
        x = batch['img']
        batch_size = x.shape[0]
        # Drop the 32-px horizontal padding added by ViT bbox-aspect expansion.
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x[:, :, :, 32:-32])

        device = temp_mano_params['hand_pose'].device
        dtype = temp_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)

        # Initial MANO pass — RefineNet conditions on these vertices.
        temp_mano_params['global_orient'] = temp_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        temp_mano_params['hand_pose'] = temp_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        temp_mano_params['betas'] = temp_mano_params['betas'].reshape(batch_size, -1)
        temp_vertices = self.mano(**temp_mano_params, pose2rot=False).vertices

        pred_mano_params, pred_cam = self.refine_net(
            vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length,
        )

        output = {
            'pred_cam': pred_cam,
            'pred_mano_params': {k: v.clone() for k, v in pred_mano_params.items()},
            'focal_length': focal_length,
        }
        pred_cam_t = torch.stack([
            pred_cam[:, 1],
            pred_cam[:, 2],
            2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
        ], dim=-1)
        output['pred_cam_t'] = pred_cam_t

        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**pred_mano_params, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = mano_output.vertices.reshape(batch_size, -1, 3)

        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t.reshape(-1, 3),
            focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
        )
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)
