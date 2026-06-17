"""Causal attention + KV-cache monkey-patch for Wan 2.2 ``WanSelfAttention``.

``_causal_mode`` on each self_attn:
  ``"off"``          — original bidirectional forward (default after patch).
  ``"infer"``        — AR with KV-cache; dynamic-shape slice (eager-fast, not compile-friendly).
  ``"infer_static"`` — AR via ``flash_attn_with_kvcache`` (fixed-shape; torch.compile-friendly).

Block / model forwards stay untouched: state is threaded through module attributes set
by the inference driver.
"""
from __future__ import annotations

import math
import types
from typing import Dict, List

import torch


def causal_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor,
                     start_frame: int = 0) -> torch.Tensor:
    """Apply Wan-style 3D (F, H, W) RoPE with a temporal start-frame offset.

    Args:
        x:          [B, L, n_heads, head_dim] — Q or K before rotation.
        grid_sizes: [B, 3] — (F, H, W) in token (patched) space for each batch item.
        freqs:      [1024, head_dim/2] complex — the prebuilt (or Riflex-scaled) RoPE table.
        start_frame: absolute frame index of the FIRST frame in `x`. Frame i in `x` uses
                     temporal RoPE position `start_frame + i`.

    Returns:
        x rotated in-place semantics, same shape and dtype.
    """
    n, c = x.size(2), x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs_split[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _patched_self_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    """Replaces ``WanSelfAttention.forward`` with a causal KV-cache path.

    ``self._causal_mode``: ``"off"`` → original bidirectional forward;
    ``"infer"`` → ring-buffer KV-cache + sliding-window FA2;
    ``"infer_static"`` → flash_attn_with_kvcache (shape-invariant for torch.compile).
    """
    mode = getattr(self, "_causal_mode", "off")
    if mode == "off":
        return self._original_forward(x, seq_lens, grid_sizes, freqs, dtype=dtype)
    if mode not in ("infer", "infer_static"):
        raise ValueError(f"Unknown _causal_mode: {mode!r}")

    # Shared Q/K/V projection (identical to original).
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
    k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
    v = self.v(x.to(dtype)).view(b, s, n, d)

    kv_cache: Dict[str, torch.Tensor] = self._kv_cache
    current_start = int(self._current_start)
    current_end = int(self._current_end)
    tokens_per_frame = int(math.prod(grid_sizes[0][1:].tolist()))
    assert current_start % tokens_per_frame == 0, (
        f"current_start={current_start} not aligned to tokens_per_frame={tokens_per_frame}"
    )
    start_frame = current_start // tokens_per_frame

    roped_q = causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
    roped_k = causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)

    # Write rotated K and raw V into the per-layer ring buffer. Wan attention is
    # global (window_size=(-1,-1)), so K order within the attended window doesn't
    # matter — ring eviction is safe as long as evicted slots are outside the
    # sliding-window read range.
    cache_size = kv_cache["k"].shape[1]
    ws = current_start % cache_size
    n = current_end - current_start
    if ws + n <= cache_size:
        kv_cache["k"][:, ws:ws + n] = roped_k
        kv_cache["v"][:, ws:ws + n] = v
    else:
        n_first = cache_size - ws
        kv_cache["k"][:, ws:] = roped_k[:, :n_first]
        kv_cache["v"][:, ws:] = v[:, :n_first]
        kv_cache["k"][:, :n - n_first] = roped_k[:, n_first:]
        kv_cache["v"][:, :n - n_first] = v[:, n_first:]

    if mode == "infer_static":
        # Static-shape KV via flash_attn_with_kvcache.
        from flash_attn import flash_attn_with_kvcache
        cache_seqlens = torch.tensor([current_end], dtype=torch.int32, device=roped_q.device)
        x = flash_attn_with_kvcache(
            roped_q.to(dtype), kv_cache["k"], kv_cache["v"],
            k=None, v=None, cache_seqlens=cache_seqlens, causal=False,
        ).to(dtype)
    else:
        from hand2world_model.models.attention_utils import attention as wan_attention
        # Sliding-window KV: clamp the attended range to [window_start, current_end).
        window_blocks = getattr(self, "_kv_cache_window", None)
        if window_blocks is not None and window_blocks > 0:
            window_start = max(0, current_end - window_blocks * tokens_per_frame)
        else:
            window_start = 0
        attn_len = current_end - window_start
        assert attn_len <= cache_size, (
            f"sliding window {attn_len} > cache_size {cache_size}; "
            f"_kv_cache_window={window_blocks} must be ≤ cache_size/tpf"
        )
        ring_start = window_start % cache_size
        if ring_start + attn_len <= cache_size:
            attn_k = kv_cache["k"][:, ring_start:ring_start + attn_len]
            attn_v = kv_cache["v"][:, ring_start:ring_start + attn_len]
        else:
            attn_k = torch.cat([kv_cache["k"][:, ring_start:],
                                 kv_cache["k"][:, :attn_len - (cache_size - ring_start)]], dim=1)
            attn_v = torch.cat([kv_cache["v"][:, ring_start:],
                                 kv_cache["v"][:, :attn_len - (cache_size - ring_start)]], dim=1)
        x = wan_attention(
            roped_q.to(dtype), attn_k.to(dtype), attn_v.to(dtype),
            k_lens=None, window_size=self.window_size,
        ).to(dtype)

    x = x.flatten(2)
    x = self.o(x)
    return x


def apply_causal_patch(transformer) -> None:
    """Monkey-patch every `WanSelfAttention` in `transformer.blocks` to use `_patched_self_attn_forward`.

    After this call every self_attn has: `_causal_mode='off'`, `_original_forward` saved, and the
    new forward bound. Safe to call once per model instance.
    """
    for block in transformer.blocks:
        attn = block.self_attn
        if getattr(attn, "_causal_mode", None) is not None:
            continue                                                # already patched
        attn._original_forward = attn.forward
        attn._causal_mode = "off"
        attn._kv_cache = None
        attn._current_start = 0
        attn._current_end = 0
        attn.forward = types.MethodType(_patched_self_attn_forward, attn)


def set_causal_mode_off(transformer) -> None:
    for block in transformer.blocks:
        block.self_attn._causal_mode = "off"


def set_causal_mode_infer(transformer, kv_caches: List[Dict[str, torch.Tensor]],
                          current_start: int, current_end: int,
                          static_shape: bool = False) -> None:
    """Arm AR inference mode. ``kv_caches[i]`` is the cache for block i.
    ``static_shape=True`` selects flash_attn_with_kvcache (fixed input shape for torch.compile)."""
    assert len(kv_caches) == len(transformer.blocks)
    mode = "infer_static" if static_shape else "infer"
    for i, block in enumerate(transformer.blocks):
        attn = block.self_attn
        attn._causal_mode = mode
        attn._kv_cache = kv_caches[i]
        attn._current_start = current_start
        attn._current_end = current_end


def update_causal_window(transformer, current_start: int, current_end: int) -> None:
    """Per-block window update during the AR rollout loop."""
    for block in transformer.blocks:
        block.self_attn._current_start = current_start
        block.self_attn._current_end = current_end


def allocate_kv_caches(num_layers: int, batch_size: int, total_tokens: int,
                       num_heads: int, head_dim: int, dtype: torch.dtype,
                       device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """One KV cache dict per transformer block; shape [B, total_tokens, H, D] for both K and V."""
    shape = [batch_size, total_tokens, num_heads, head_dim]
    return [{"k": torch.zeros(shape, dtype=dtype, device=device),
             "v": torch.zeros(shape, dtype=dtype, device=device)}
            for _ in range(num_layers)]


