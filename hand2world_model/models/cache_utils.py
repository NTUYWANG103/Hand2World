import numpy as np
import torch

def get_teacache_coefficients(model_name):
    """Polynomial coefficients for TeaCache. Returns None if unsupported."""
    if "wan2.2-fun" in model_name.lower():
        return [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02]
    print(f"The model {model_name} is not supported by TeaCache.")
    return None


class TeaCache:
    """Timestep Embedding Aware Cache (https://github.com/ali-vilab/TeaCache).
    Caches residual deltas across nearby timesteps and skips blocks when the
    timestep-embedding change is below ``rel_l1_thresh``.
    """
    def __init__(self, coefficients, num_steps, rel_l1_thresh=0.0,
                 num_skip_start_steps=0, offload=True):
        assert num_steps >= 1
        assert rel_l1_thresh >= 0
        assert 0 <= num_skip_start_steps <= num_steps
        self.num_steps = num_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.num_skip_start_steps = num_skip_start_steps
        self.offload = offload
        self.rescale_func = np.poly1d(coefficients)
        self.reset()

    @staticmethod
    def compute_rel_l1_distance(prev: torch.Tensor, cur: torch.Tensor) -> float:
        return ((torch.abs(cur - prev).mean()) / torch.abs(prev).mean()).cpu().item()

    def reset(self):
        self.cnt = 0
        self.should_calc = True
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.previous_residual = None
