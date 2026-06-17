"""Wan 2.2 Fun-Control bidirectional pipeline.

``guidance_scale=1.0`` only (the LoRA is trained at ``text_drop_ratio=0``; CFG > 1
produces artifacts), single transformer (no two-stage transformer_2 split), no callbacks.
Outputs ``(B, C, T, H, W)`` float tensor in [0, 1].
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import AutoTokenizer

from ..models import AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel


@dataclass
class WanPipelineOutput(BaseOutput):
    """``videos``: ``(B, C, T, H, W)`` float tensor in [0, 1]."""
    videos: torch.Tensor


class Wan2_2FunControlPipeline(DiffusionPipeline):
    """Bidirectional Wan 2.2 Fun-Control pipeline."""

    def __init__(self, tokenizer: AutoTokenizer, text_encoder: WanT5EncoderModel,
                 vae: AutoencoderKLWan3_8, transformer: Wan2_2Transformer3DModel,
                 scheduler=None):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
            transformer=transformer, scheduler=scheduler,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)

    def _t5_embed(self, prompt: Union[str, List[str]], device, dtype) -> List[torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        tok = self.tokenizer(
            prompt, padding="max_length", max_length=512,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        attn = tok.attention_mask.to(device)
        seq_lens = attn.gt(0).sum(dim=1).long()
        embeds = self.text_encoder(input_ids, attention_mask=attn)[0].to(dtype=dtype, device=device)
        return [u[:v] for u, v in zip(embeds, seq_lens)]

    def _prepare_latents(self, batch_size, num_channels, num_frames, height, width,
                         dtype, device, generator) -> torch.Tensor:
        shape = (
            batch_size, num_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_control_video(self, video, height, width, dtype, device) -> torch.Tensor:
        T = video.shape[2]
        video = self.image_processor.preprocess(
            rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width,
        ).to(dtype=torch.float32)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=T).to(device=device, dtype=dtype)
        # vae.encode → DiagonalGaussianDistribution; .mode() gives the deterministic mean.
        chunks = [self.vae.encode(video[i:i + 1])[0].mode() for i in range(video.shape[0])]
        return torch.cat(chunks, dim=0)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → (B, C, T, H, W) float tensor in [0, 1]."""
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        return (frames / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 480,
        width: int = 720,
        control_video: Optional[torch.FloatTensor] = None,
        control_camera_video: Optional[torch.FloatTensor] = None,
        start_image: Optional[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> WanPipelineOutput:
        assert height % 8 == 0 and width % 8 == 0, f"height/width must be %8, got {height}x{width}"

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        prompt_embeds = self._t5_embed(prompt, device, weight_dtype)
        batch_size = len(prompt_embeds)

        # Scheduler timesteps.
        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=1)
        timesteps = self.scheduler.timesteps

        latents = self._prepare_latents(
            batch_size, self.vae.config.latent_channels, num_frames, height, width,
            weight_dtype, device, generator,
        )

        # Control camera (plucker) → 24-ch latent (channel-stack 4 pixel frames per latent slot).
        if control_camera_video is not None:
            from ..data.utils import channel_stack_plucker_to_latent
            control_camera_latents = channel_stack_plucker_to_latent(control_camera_video)
            if control_video is not None:
                control_video_latents = self._encode_control_video(
                    control_video, height, width, weight_dtype, device,
                )
            else:
                control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
        elif control_video is not None:
            control_video_latents = self._encode_control_video(
                control_video, height, width, weight_dtype, device,
            )
            control_camera_latents = None
        else:
            control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            control_camera_latents = None

        # Pre-compute camera adapter embeddings once (vs per-step).
        control_camera_embed = None
        if (control_camera_latents is not None
                and getattr(self.transformer, "control_adapter", None) is not None):
            control_camera_embed = self.transformer.control_adapter(
                control_camera_latents.to(device, weight_dtype),
            )

        # Start image (frame 0 latent), zero-pad slots 1..F-1.
        if start_image is not None:
            start_latents = self._encode_control_video(
                start_image, height, width, weight_dtype, device,
            )
            start_image_conv_in = torch.zeros_like(latents)
            if latents.size(2) != 1:
                start_image_conv_in[:, :, :1] = start_latents
        else:
            start_image_conv_in = torch.zeros_like(latents)

        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio,
        )
        ps = self.transformer.config.patch_size  # (1, 2, 2)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (ps[1] * ps[2]) * target_shape[1])
        self.transformer.num_inference_steps = num_inference_steps

        # Denoising loop. Per-token timesteps: frame 0 stays clean (t=0), others get t.
        with self.progress_bar(total=num_inference_steps) as pb:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                control_latents_input = torch.cat(
                    [control_video_latents.to(device, weight_dtype),
                     start_image_conv_in.to(device, weight_dtype)], dim=1,
                )
                # Per-token timesteps for I2V (start_image present).
                if (self.vae.spatial_compression_ratio >= 16
                        and start_image is not None
                        and start_image_conv_in.abs().sum() > 0):
                    H_lat, W_lat = latents.shape[3], latents.shape[4]
                    tokens_per_frame = (H_lat // ps[1]) * (W_lat // ps[2])
                    per_token_t = t * torch.ones(seq_len, device=device, dtype=weight_dtype)
                    per_token_t[:tokens_per_frame] = 0
                    timestep = per_token_t.unsqueeze(0).expand(latents.shape[0], -1)
                    keep_frame0 = True
                else:
                    timestep = t.expand(latents.shape[0])
                    keep_frame0 = False

                with torch.amp.autocast("cuda", dtype=weight_dtype):
                    noise_pred = self.transformer(
                        x=latents, context=prompt_embeds, t=timestep, seq_len=seq_len,
                        y=control_latents_input,
                        y_camera_embed=(control_camera_embed.to(device, weight_dtype)
                                        if control_camera_embed is not None else None),
                    )

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if keep_frame0:
                    latents[:, :, :1] = start_image_conv_in[:, :, :1]
                pb.update()

        video = self._decode_latents(latents)              # (B, C, T, H, W) float32 in [0, 1]
        self.maybe_free_model_hooks()
        return WanPipelineOutput(videos=video)
