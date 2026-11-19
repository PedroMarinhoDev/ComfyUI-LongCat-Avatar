"""The release's distilled sampling as one GUIDER: the DMD and OPSD LoRAs mixed by step (``--mix_steps``, ``--mix_alpha``).

The release runs the first ``mix_steps`` steps with ``mix_alpha`` DMD + ``1 - mix_alpha`` OPSD and the rest with OPSD
alone (longcat_video_dit_avatar.set_lora_mix_step), cfg 1. Here each phase is a clone of the model with ComfyUI's own
LoRA patches, which dynamic VRAM streams with the weights, and the guider samples the first ``mix_steps`` sigmas on
the first clone and the rest on the second: what two chained SamplerCustomAdvanced do, in one sampler node.

Each phase's sampling call gets its own ``CleanKV``, so the clean frames run through the DiT once per phase.
"""
import torch
from tqdm import tqdm

import comfy.patcher_extension
import comfy.samplers
import comfy.sd
import comfy.utils

from .model import CleanKV


def clean_kv_sampling(executor, guider, sigmas, extra_args, *args, **kwargs):
    # extra_args' model_options is this sampling call's own copy: the store goes with it when the call returns
    extra_args["model_options"]["transformer_options"]["longcat_clean_kv"] = CleanKV()
    return executor(guider, sigmas, extra_args, *args, **kwargs)


def patched(model, loras):
    model = model.clone()
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "longcat_clean_kv", clean_kv_sampling)
    for lora, strength in loras:
        if strength != 0.0:
            model = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)[0]
    return model


class DistillGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, dmd_path, opsd_path, mix_steps, mix_alpha):
        dmd, opsd = comfy.utils.load_torch_file(dmd_path, safe_load=True), comfy.utils.load_torch_file(opsd_path, safe_load=True)
        super().__init__(patched(model, [(dmd, mix_alpha), (opsd, 1.0 - mix_alpha)]))
        self.second = comfy.samplers.CFGGuider(patched(model, [(opsd, 1.0)]))
        self.mix_steps = mix_steps

    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})
        self.second.inner_set_conds({"positive": positive})

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        steps = sigmas.shape[-1] - 1
        split = min(self.mix_steps, steps)
        # one progress bar over all the steps: the phases count on from each other
        bar = None if disable_pbar else tqdm(total=steps, desc="LongCat Avatar", unit="step", dynamic_ncols=True)

        def counted(offset):
            def step_callback(step, x0, x, total_steps):
                if bar is not None:
                    bar.update(1)
                if callback is not None:
                    callback(offset + step, x0, x, steps)
            return step_callback

        try:
            if split > 0:
                latent_image = super().sample(noise, latent_image, sampler, sigmas[:split + 1], denoise_mask, counted(0), True, seed)
                # the second phase continues from the first's output without new noise, as DisableNoise does
                noise = torch.zeros_like(noise)
            if split < steps:
                latent_image = self.second.sample(noise, latent_image, sampler, sigmas[split:], denoise_mask, counted(split), True, seed)
        finally:
            if bar is not None:
                bar.close()
        return latent_image
