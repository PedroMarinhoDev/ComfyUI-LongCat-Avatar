"""LongCat-Video-Avatar as a native ComfyUI ``BaseModel``.

* **Sampling.** Rectified flow, ``x_t = (1 - sigma) * x0 + sigma * noise``, timestep ``1000 * sigma``. The network
  predicts ``x0 - noise`` and the DiT returns it negated, so this is ``ModelType.FLOW`` unchanged.
* **Latents.** Wan 2.1 VAE latents, ``(B, 16, T, H/8, W/8)``.
* **Clean frames.** Conditioning frames (the start image, or the last frames of the previous segment) travel as
  ``longcat_cond_latent`` and replace the first frames of the LATENT: the DiT reads them clean at timestep 0, and
  their prediction is pinned to them, so every sampler (and a sampling split in two) ends on them exactly.
* **Audio.** ``longcat_audio`` is the windowed Whisper stack ``(B, frames, 5, 5, 1280)``; zeros mean "no audio".
* **Later segments.** ``longcat_ref_latent`` is the first segment's image latent, prepended inside the DiT at RoPE
  time ``longcat_ref_index``; it is not part of the LATENT.
"""
import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base

from .model import LongCatVideoAvatar

# scheduler/scheduler_config.json of the release
SHIFT = 7.0


class LongCatAvatarConfig(comfy.supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}
    sampling_settings = {"shift": SHIFT, "multiplier": 1000}
    latent_format = comfy.latent_formats.Wan21
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    # measured (ck attention): a forward peaks at 1.08x the area term, 3.0x with the clean-frame cache; memory_required counts 2 items
    memory_usage_factor = 1.5

    def get_model(self, state_dict, prefix="", device=None):
        return LongCatAvatarModel(self, device=device)


def unet_config_from_state_dict(sd):
    hidden_size = sd["x_embedder.proj.weight"].shape[0]
    return {
        "in_channels": sd["x_embedder.proj.weight"].shape[1],
        "hidden_size": hidden_size,
        "depth": len({k.split(".")[1] for k in sd if k.startswith("blocks.")}),
        "num_heads": hidden_size // sd["blocks.0.attn.q_norm.weight"].shape[0],
        "caption_channels": sd["y_embedder.y_proj.0.weight"].shape[1],
        "adaln_tembed_dim": sd["t_embedder.mlp.0.weight"].shape[0],
        "frequency_embedding_size": sd["t_embedder.mlp.0.weight"].shape[1],
        "intermediate_dim": sd["audio_proj.proj1.weight"].shape[0],
        "output_dim": sd["audio_proj.norm.weight"].shape[0],
        "context_tokens": sd["audio_proj.proj3.weight"].shape[0] // sd["audio_proj.norm.weight"].shape[0],
    }


class LongCatAvatarModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=LongCatVideoAvatar)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        audio = kwargs.get("longcat_audio", None)
        if audio is not None:
            out["audio_embeds"] = comfy.conds.CONDRegular(audio)
        cond_latent = kwargs.get("longcat_cond_latent", None)
        if cond_latent is not None:
            out["cond_latent"] = comfy.conds.CONDRegular(self.process_latent_in(cond_latent))
        ref_latent = kwargs.get("longcat_ref_latent", None)
        if ref_latent is not None:
            out["ref_latent"] = comfy.conds.CONDRegular(self.process_latent_in(ref_latent))
            out["ref_index"] = comfy.conds.CONDConstant(kwargs.get("longcat_ref_index", 10))
        return out

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        denoised = super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        cond_latent = kwargs.get("cond_latent", None)
        if cond_latent is not None:
            denoised[:, :, :cond_latent.shape[2]] = cond_latent
        return denoised
