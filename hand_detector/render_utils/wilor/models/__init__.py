import os

from .mano_wrapper import MANO
from .wilor import WiLoR

# MANO data lives under checkpoints/wilor/mano_data (4 levels up from this file).
_DEFAULT_MANO_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "checkpoints", "wilor", "mano_data",
))


def load_wilor(checkpoint_path, cfg_path, mano_data_dir=None, init_renderer=False):
    """Load a WiLoR LightningModule from a checkpoint + yacs config.

    All asset paths resolve against the bundled package location, so the
    function is CWD-independent.
    """
    from wilor.configs import get_config

    print('Loading ', checkpoint_path)
    model_cfg = get_config(cfg_path, update_cachedir=True)

    if ('vit' in model_cfg.MODEL.BACKBONE.TYPE) and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    mano_dir = os.path.abspath(mano_data_dir) if mano_data_dir else _DEFAULT_MANO_DIR
    if 'DATA_DIR' in model_cfg.MANO:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = mano_dir
        model_cfg.MANO.MODEL_PATH = mano_dir
        model_cfg.MANO.MEAN_PARAMS = os.path.join(mano_dir, "mano_mean_params.npz")
        model_cfg.freeze()

    model = WiLoR.load_from_checkpoint(
        checkpoint_path, strict=False, cfg=model_cfg, init_renderer=init_renderer,
    )
    return model, model_cfg
