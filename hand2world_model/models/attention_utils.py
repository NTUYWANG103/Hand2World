"""Attention dispatch: FA3 → FA2 → SageAttention → SDPA (in priority order)."""
import os
import warnings

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    major, minor = torch.cuda.get_device_capability(0)
    _sm = f"{major}.{minor}"
    if _sm == "8.0":   from sageattention_sm80 import sageattn
    elif _sm == "8.6": from sageattention_sm86 import sageattn
    elif _sm == "8.9": from sageattention_sm89 import sageattn
    elif _sm == "9.0": from sageattention_sm90 import sageattn
    elif major > 9:    from sageattention_sm120 import sageattn
    else:              raise ImportError
    SAGE_ATTENTION_AVAILABLE = True
except Exception:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except ImportError:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False


def _flash_attention(q, k, v, k_lens, window_size):
    """FA3/FA2 varlen attention; ``k_lens=None`` → pack all batch sequences end-to-end."""
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    out_dtype = q.dtype
    q_packed = q.flatten(0, 1)
    if k_lens is None:
        k_packed, v_packed = k.flatten(0, 1), v.flatten(0, 1)
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=q.device)
    else:
        k_packed = torch.cat([u[:n] for u, n in zip(k, k_lens)])
        v_packed = torch.cat([u[:n] for u, n in zip(v, k_lens)])
    q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    # cu_seqlens_{q,k} MUST live on q.device for flash_attn_varlen_func (k_lens may be on CPU).
    cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
    cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)

    if FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q_packed, k=k_packed, v=v_packed, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            seqused_q=None, seqused_k=None, max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=None, causal=False, deterministic=False,
        )
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q_packed, k=k_packed, v=v_packed, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=0., softmax_scale=None,
            causal=False, window_size=window_size, deterministic=False,
        )
    return x.unflatten(0, (b, lq)).type(out_dtype)


def attention(q, k, v, k_lens=None, window_size=(-1, -1)):
    """[B, L, H, D] q/k/v → [B, Lq, H, D]. ``k_lens`` packs ragged K rows; ``window_size``
    applies sliding-window local attention on FA2."""
    attn_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
    if attn_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if k_lens is not None:
            warnings.warn("SageAttention ignores k_lens padding mask.")
        return sageattn(q, k, v, attn_mask=None, tensor_layout="NHD",
                        is_causal=False, dropout_p=0.)
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return _flash_attention(q, k, v, k_lens, window_size)
    # SDPA fallback (no varlen support, no window).
    if k_lens is not None:
        warnings.warn("SDPA fallback ignores k_lens padding mask.")
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None, is_causal=False, dropout_p=0.,
    )
    return out.transpose(1, 2).contiguous()
