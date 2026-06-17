"""FlowMatchScheduler (inference subset).

Defaults match Wan 2.2 5B teacher: shift=5.0, sigma_min=0.0, extra_one_step=True.
``step()`` is an Euler step.
"""
from __future__ import annotations

import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int = 100, denoising_strength: float = 1.0, device=None):
        sigma_start = self.sigma_min + (1.0 - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if device is not None:
            self.sigmas = self.sigmas.to(device)
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor,
             return_dict: bool = False, **kwargs):
        """Euler step ``prev = sample + pred * (sigma_next - sigma_t)``.

        Returns ``(prev_sample,)`` for diffusers-style ``step(...)[0]`` consumption.
        """
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([float(timestep)], device=sample.device, dtype=torch.float32)
        elif timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(sample.device)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin((self.timesteps - timestep[0]).abs())
        sigma = self.sigmas[timestep_id]
        if timestep_id + 1 >= len(self.timesteps):
            sigma_next = torch.tensor(0.0, device=sample.device, dtype=sample.dtype)
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_next - sigma)
        return (prev_sample,)
