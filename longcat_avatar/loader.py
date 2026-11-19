"""Load a LongCat-Video-Avatar checkpoint into a native ``MODEL`` (mirrors ``comfy.sd.load_diffusion_model_state_dict``).

The stock loader has no config for this model, so the config comes from the weights' shapes; the rest is the stock
path: dtype choice, mixed-precision ops for quantized files, a dynamic-VRAM patcher, ``assign`` loading and
``cached_patcher_init`` for Select Model Device.
"""
import logging
import os

import comfy.model_management
import comfy.model_patcher
import comfy.storage
import comfy.utils

from .model_base import LongCatAvatarConfig, unet_config_from_state_dict

# the avatar-only weights, and every key unet_config_from_state_dict reads
KEYS = ("audio_proj.proj1_vf.weight", "blocks.0.audio_cross_attn.kv_linear.weight", "x_embedder.proj.weight", "blocks.0.attn.q_norm.weight",
        "y_embedder.y_proj.0.weight", "t_embedder.mlp.0.weight", "audio_proj.proj1.weight", "audio_proj.norm.weight", "audio_proj.proj3.weight")


def load_longcat_avatar(path, disable_dynamic=False):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    missing = [k for k in KEYS if k not in sd]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is not a LongCat-Video-Avatar checkpoint (no {missing[0]}; run tools/convert.py on the release's base_model folder)")

    model_config = LongCatAvatarConfig(unet_config_from_state_dict(sd))
    model_config.quant_config = comfy.utils.detect_layer_quantization(sd, "")

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    weight_dtype = None if model_config.quant_config is not None else comfy.utils.weight_dtype(sd)
    unet_dtype = comfy.model_management.unet_dtype(model_params=comfy.utils.calculate_parameters(sd),
                                                   supported_dtypes=model_config.supported_inference_dtypes, weight_dtype=weight_dtype)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(None if model_config.quant_config is not None else unet_dtype,
                                                                load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)

    model = model_config.get_model(sd, "")
    # attribute access, not a from-import: main.py rebinds CoreModelPatcher to the dynamic-VRAM patcher at startup
    patcher_class = comfy.model_patcher.ModelPatcher if disable_dynamic else comfy.model_patcher.CoreModelPatcher
    patcher = patcher_class(model, load_device=load_device, offload_device=offload_device, fast_disk=comfy.storage.state_dict_fast_disk(sd))
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    model.load_model_weights(sd, "", assign=patcher.is_dynamic())
    patcher.cached_patcher_init = (load_longcat_avatar, (path,))
    logging.info("LongCat Avatar: %s, %s", path, "quantized" if model_config.quant_config is not None else unet_dtype)
    return patcher
