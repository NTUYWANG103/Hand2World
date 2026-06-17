"""WiLoR config loader: ``get_config`` + a minimal default-config base.

All runtime settings come from the YAML loaded by ``get_config``.
"""
import os

from yacs.config import CfgNode as CN

CACHE_DIR_PRETRAINED = "./pretrained_models/"


def _default_config() -> CN:
    cfg = CN(new_allowed=True)
    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.IMAGE_SIZE = 224
    cfg.EXTRA = CN(new_allowed=True)
    cfg.EXTRA.FOCAL_LENGTH = 5000
    cfg.LOSS_WEIGHTS = CN(new_allowed=True)
    return cfg


def get_config(config_file: str, merge: bool = True, update_cachedir: bool = False) -> CN:
    """Load YAML config file. ``update_cachedir`` rewrites relative MANO paths
    under ``CACHE_DIR_PRETRAINED``."""
    cfg = _default_config() if merge else CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    if update_cachedir:
        for k in ("MODEL_PATH", "MEAN_PARAMS"):
            v = cfg.MANO[k]
            if not os.path.isabs(v):
                cfg.MANO[k] = os.path.join(CACHE_DIR_PRETRAINED, v)
    cfg.freeze()
    return cfg
