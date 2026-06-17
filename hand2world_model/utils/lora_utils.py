"""Direct state-dict LoRA merge for diffusers pipelines."""
import os
from collections import defaultdict

import torch
from safetensors.torch import load_file


def merge_lora(pipeline, lora_path, multiplier, device='cpu', dtype=torch.float32):
    if lora_path is None:
        return pipeline

    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if os.path.isdir(lora_path):
        candidates = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
        if 'lora_diffusion_pytorch_model.safetensors' in candidates:
            lora_path = os.path.join(lora_path, 'lora_diffusion_pytorch_model.safetensors')
        elif len(candidates) == 1:
            lora_path = os.path.join(lora_path, candidates[0])
        else:
            raise ValueError(f"Cannot resolve lora_path directory: {lora_path}, found safetensors: {candidates}")
    state_dict = load_file(lora_path)
    updates = defaultdict(dict)
    control_adapter_state_dict = {}
    patch_embedding_state_dict = {}
    for key, value in state_dict.items():
        # Skip non-LoRA keys (control_adapter / patch_embedding bundled alongside).
        if "lora_A" not in key and "lora_B" not in key and "lora_up" not in key and "lora_down" not in key:
            if key.startswith("control_adapter."):
                control_adapter_state_dict[key] = value
            elif key.startswith("patch_embedding."):
                patch_embedding_state_dict[key] = value
            continue
        if "lora_A" in key or "lora_B" in key:
            key = "lora_unet__" + key
        key = key.replace(".", "_")
        if key.endswith("_lora_up_weight"):
            key = key[:-15] + ".lora_up.weight"
        if key.endswith("_lora_down_weight"):
            key = key[:-17] + ".lora_down.weight"
        if key.endswith("_lora_A_default_weight"):
            key = key[:-21] + ".lora_A.weight"
        if key.endswith("_lora_B_default_weight"):
            key = key[:-21] + ".lora_B.weight"
        if key.endswith("_lora_A_weight"):
            key = key[:-14] + ".lora_A.weight"
        if key.endswith("_lora_B_weight"):
            key = key[:-14] + ".lora_B.weight"
        if key.endswith("_alpha"):
            key = key[:-6] + ".alpha"
        key = key.replace(".lora_A.default.", ".lora_down.")
        key = key.replace(".lora_B.default.", ".lora_up.")
        key = key.replace(".lora_A.", ".lora_down.")
        key = key.replace(".lora_B.", ".lora_up.")
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value
    
    # Load control_adapter / patch_embedding weights if bundled in the LoRA checkpoint.
    for prefix, sd in (("control_adapter.", control_adapter_state_dict),
                        ("patch_embedding.", patch_embedding_state_dict)):
        if not sd:
            continue
        attr = prefix.rstrip(".")
        if hasattr(pipeline.transformer, attr):
            m, u = getattr(pipeline.transformer, attr).load_state_dict(
                {k.replace(prefix, ""): v for k, v in sd.items()}, strict=False,
            )
            print(f"Loaded {len(sd)} {attr} weights (missing={len(m)}, unexpected={len(u)})")

    for layer, elems in updates.items():

        if "lora_te" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
            curr_layer = pipeline.transformer

        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                print(f'Error loading layer in front search: {layer}. Try it in back search.')
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                if "lora_te" in layer:
                    layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                    curr_layer = pipeline.text_encoder
                else:
                    layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
                    curr_layer = pipeline.transformer

                len_layer_infos = len(layer_infos)
                start_index     = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx        = len_layer_infos

                error_flag      = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            print(f'Error loading layer in back search: {layer}')
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue

        origin_dtype = curr_layer.weight.data.dtype
        origin_device = curr_layer.weight.data.device

        curr_layer = curr_layer.to(device, dtype)
        weight_up = elems['lora_up.weight'].to(device, dtype)
        weight_down = elems['lora_down.weight'].to(device, dtype)
        
        if 'alpha' in elems:
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(
                weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
        curr_layer = curr_layer.to(origin_device, origin_dtype)

    return pipeline
