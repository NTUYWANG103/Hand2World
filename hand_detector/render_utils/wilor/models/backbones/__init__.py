from .vit import vit


def create_backbone(cfg):
    """Construct a backbone from config. Only ``BACKBONE.TYPE == 'vit'`` is supported."""
    if cfg.MODEL.BACKBONE.TYPE != 'vit':
        raise NotImplementedError(f"backbone type {cfg.MODEL.BACKBONE.TYPE!r} not supported")
    return vit(cfg)
