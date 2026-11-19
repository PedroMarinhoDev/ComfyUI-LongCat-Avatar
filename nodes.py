"""LongCat-Video-Avatar 1.6 nodes.

A video is a run of 93-frame segments: the first starts from the image, each next one continues from the last 13
frames of the one before, with the speech that follows. Two nodes make the whole video and encode it as it goes; they
differ only in the guider, conds and sigmas of a segment:

* LongCat Avatar Video: distilled, 8 steps with the DMD and OPSD LoRAs, cfg 1.
* LongCat Avatar Video Base: the base model, 50 steps with text and audio guidance.
"""
import logging
import math
import time

import torch
from tqdm import tqdm
from typing_extensions import override

import comfy.audio
import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import node_helpers
from comfy_api.latest import ComfyExtension, InputImpl, io
from comfy_extras.nodes_custom_sampler import Guider_DualCFG

from .longcat_avatar.distill import DistillGuider
from .longcat_avatar.loader import load_longcat_avatar
from .longcat_avatar.video_io import ENCODERS, Writer, audio_samples

AUDIO_FEATURES = io.Custom("LONGCAT_AVATAR_AUDIO")

# the release's segment: 93 frames, each next one continuing from the last 13
LENGTH = 93
COND_FRAMES = 13
REF_INDEX = 10

FPS = 25
WHISPER_RATE = 16000
WHISPER_CHUNK = 30 * WHISPER_RATE
WHISPER_FPS = 50
# silence appended to the speech, so a last segment that runs past it reads silence (the release pads with zeros too)
TAIL_SECONDS = 4.0
# the release's vocal separator (audio-separator 0.30.2, normalization_threshold 0.9) scales speech that peaks above 0.9
# down to 0.9 and leaves quieter speech as it is
PEAK = 0.9


class LongCatAvatarModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCatAvatarModelLoader",
            display_name="Load LongCat Avatar Model",
            category="LongCat Avatar",
            inputs=[io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models"))],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name):
        return io.NodeOutput(load_longcat_avatar(folder_paths.get_full_path_or_raise("diffusion_models", unet_name)))


class LongCatAvatarAudioEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCatAvatarAudioEncode",
            display_name="LongCat Avatar Audio Encode",
            category="LongCat Avatar",
            description="Whisper large-v3 features of the speech at 25 fps, any length. Feed clean vocals: the release separates them from music first.",
            inputs=[io.AudioEncoder.Input("audio_encoder", tooltip="whisper_large_v3 from Load Audio Encoder."), io.Audio.Input("audio")],
            outputs=[AUDIO_FEATURES.Output(display_name="audio_features"),
                     io.Int.Output(display_name="segments", tooltip="Segments of 93 frames, each continuing from 13, that cover the speech: "
                                                                    "the loop count for Start Loop.")],
        )

    @classmethod
    def execute(cls, audio_encoder, audio):
        wave = speech(audio)
        speech_frames = int(wave.shape[-1] / WHISPER_RATE * FPS)
        wave = torch.nn.functional.pad(wave, (0, int(TAIL_SECONDS * WHISPER_RATE)))
        frames = int(wave.shape[-1] / WHISPER_RATE * FPS)

        comfy.model_management.load_model_gpu(audio_encoder.patcher)
        groups = []
        # Whisper reads 30 s windows: one mel and one encoder pass per window, like the release
        for start in range(0, wave.shape[-1], WHISPER_CHUNK):
            _, layers = audio_encoder.model(wave[..., start:start + WHISPER_CHUNK].to(audio_encoder.load_device))
            layers = torch.stack(layers, dim=2).float()
            # embeddings + 32 layers -> means of layers 0-7, 8-15, 16-23, 24-31 and the final layer-normed output
            groups.append(torch.stack([layers[:, :, i:i + 8].mean(dim=2) for i in range(0, 32, 8)] + [layers[:, :, 32]], dim=2))
        features = torch.cat(groups, dim=1)[0, :2 * frames]
        features = torch.nn.functional.interpolate(features.flatten(1).T.unsqueeze(0), size=frames, mode="linear", align_corners=True)
        features = features[0].T.unflatten(1, (5, -1)).to(comfy.model_management.intermediate_device())
        segments = segment_count(speech_frames)
        logging.info("LongCat Avatar: %.1f s of speech, %d frames at %d fps: %d segment%s", speech_frames / FPS, speech_frames, FPS,
                     segments, "" if segments == 1 else "s")
        return io.NodeOutput({"features": features, "segments": segments}, segments)


def speech(audio):
    """The first clip of an AUDIO as [1, 1, samples] at 16 kHz, mono, as the release's separator returns it: peak
    limited to 0.9 over all channels (never amplified), then the channels' mean, then resampled."""
    wave = audio["waveform"][:1].float()
    peak = wave.abs().max()
    if peak > PEAK:
        wave = wave * (PEAK / peak)
    return comfy.audio.resample(wave.mean(dim=1, keepdim=True), audio["sample_rate"], WHISPER_RATE)


def segment_count(frames, length=LENGTH, cond_frames=COND_FRAMES):
    return 1 + math.ceil(max(0, frames - length) / (length - cond_frames))


def resize(images, width, height):
    # the release resizes to cover the frame and center-crops
    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)[..., :3]


def segment_conditioning(vae, audio_features, width, height, length, segment, num_cond_frames, ref_img_index, start_image=None, previous_frames=None,
                         ref_latent=None):
    """Conditioning values, the empty latent and the count of repeated frames for segment ``segment``.

    The first segment starts clean from start_image (or from nothing). A later one starts from the last
    num_cond_frames of previous_frames, keeps an identity reference (ref_latent if given, else the start image, else
    previous_frames' first frame, encoded) and reads the speech from where those frames are.
    """
    latent = torch.zeros([1, 16, (length - 1) // 4 + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
    values = {}
    trim = 0
    if previous_frames is not None:
        trim = min(num_cond_frames, previous_frames.shape[0], length)
        values["longcat_cond_latent"] = vae.encode(resize(previous_frames[-trim:], width, height))
        if ref_latent is None:
            reference = start_image if start_image is not None else previous_frames
            ref_latent = vae.encode(resize(reference[:1], width, height))
        values["longcat_ref_latent"] = ref_latent
        values["longcat_ref_index"] = ref_img_index
    elif start_image is not None:
        values["longcat_cond_latent"] = vae.encode(resize(start_image[:1], width, height))

    # each frame reads a 5-frame window of features centred on it
    features = audio_features["features"]
    start = segment * (length - num_cond_frames)
    index = (torch.arange(start, start + length).unsqueeze(1) + torch.arange(-2, 3)).clamp(0, features.shape[0] - 1)
    values["longcat_audio"] = features[index].unsqueeze(0)
    return values, latent, trim


def release_sigmas(model, steps, schedule):
    # pipeline_longcat_video_avatar.get_timesteps_sigmas (avatar-v1.5 branch, which 1.6 runs), then the scheduler's shift
    if schedule == "distilled":
        table = torch.linspace(0, 1, 1000).flip(0)
        index = 1000 - (torch.arange(1, steps + 1, dtype=torch.float32) * (1000 // steps)).round().long()
        sigmas = table[index].flip(0)
    else:
        sigmas = torch.linspace(1, 0.001, steps)
    sigmas = comfy.model_sampling.time_snr_shift(model.get_model_object("model_sampling").shift, sigmas)
    if schedule == "base continuation":
        # generate_avc enhance_hf: keep the steps above 0.5, then 10 even steps from 0.5 down to 0.05
        sigmas = torch.cat([sigmas[sigmas > 0.5], torch.linspace(0.5, 0, 11)[:-1]])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def decode(vae, samples):
    images = vae.decode(samples)
    return images.reshape(-1, *images.shape[-3:])


ENCODER_TOOLTIP = ("hevc nvenc / h264 nvenc: NVIDIA's video encoder, beside the CUDA cores. h264: x264 on the CPU. "
                   "auto takes the first that runs, in that order.")
REF_INDEX_TOOLTIP = "Time position of the identity reference's latent in continued segments; it keeps the identity steady."


class LongCatAvatarVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        loras = folder_paths.get_filename_list("loras")
        default = lambda name: next((l for l in loras if name in l), loras[0] if loras else None)
        return io.Schema(
            node_id="LongCatAvatarVideo",
            display_name="LongCat Avatar Video (distilled, 8 steps)",
            category="LongCat Avatar",
            description="The whole audio-driven video in one node, sampled the release's distilled way: 8 steps, DMD then "
                        "OPSD LoRA, cfg 1. As many 93-frame segments as the speech needs, each continuing from the last 13 "
                        "frames and encoded into the video as soon as it is decoded.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive", tooltip="The prompt (UMT5 / Wan 2.1 text encoder)."),
                io.Vae.Input("vae"),
                AUDIO_FEATURES.Input("audio_features"),
                io.Audio.Input("audio", tooltip="The speech, muxed into the video."),
                io.Combo.Input("dmd_lora", options=loras, default=default("dmd_lora")),
                io.Combo.Input("opsd_lora", options=loras, default=default("opsd_lora")),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Image.Input("start_image", optional=True, tooltip="The person. Without it the video is made from the prompt and audio alone."),
                io.Int.Input("max_segments", default=0, min=0, max=100000, advanced=True, tooltip="Stop after this many segments; 0 covers the whole speech."),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler", advanced=True),
                io.Int.Input("mix_steps", default=4, min=0, max=8, advanced=True, tooltip="Steps that use mix_alpha DMD + (1 - mix_alpha) OPSD; the rest use OPSD."),
                io.Float.Input("mix_alpha", default=1.0, min=0.0, max=1.0, step=0.05, advanced=True),
                io.Int.Input("ref_img_index", default=REF_INDEX, min=-1000, max=1000, advanced=True, tooltip=REF_INDEX_TOOLTIP),
                io.Combo.Input("encoder", options=["auto"] + list(ENCODERS), default="auto", advanced=True, tooltip=ENCODER_TOOLTIP),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, model, positive, vae, audio_features, audio, dmd_lora, opsd_lora, seed, width, height, max_segments, sampler_name,
                mix_steps, mix_alpha, ref_img_index, encoder, start_image=None):
        guider = DistillGuider(model, folder_paths.get_full_path_or_raise("loras", dmd_lora),
                               folder_paths.get_full_path_or_raise("loras", opsd_lora), mix_steps, mix_alpha)
        sigmas = release_sigmas(model, 8, "distilled")

        def set_conds(values):
            guider.set_conds(node_helpers.conditioning_set_values(positive, values))

        return io.NodeOutput(sample_video(guider, set_conds, (sigmas, sigmas), vae, audio_features, audio, seed, width, height, start_image,
                                        max_segments, sampler_name, ref_img_index, encoder))


class LongCatAvatarVideoBase(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LongCatAvatarVideoBase",
            display_name="LongCat Avatar Video (base, 50 steps, text + audio guidance)",
            category="LongCat Avatar",
            description="The whole audio-driven video in one node with the base model alone, no LoRAs: the release's 50 steps "
                        "with text and audio guidance, three model passes per step, so many times slower than distilled. "
                        "As many 93-frame segments as the speech needs, each continuing from the last 13 frames.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive", tooltip="The prompt (UMT5 / Wan 2.1 text encoder)."),
                io.Conditioning.Input("negative", tooltip="The negative prompt."),
                io.Vae.Input("vae"),
                AUDIO_FEATURES.Input("audio_features"),
                io.Audio.Input("audio", tooltip="The speech, muxed into the video."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Image.Input("start_image", optional=True, tooltip="The person. Without it the video is made from the prompt and audio alone."),
                io.Int.Input("steps", default=50, min=1, max=1000, tooltip="Continued segments keep the steps above sigma 0.5 and end on 10 even ones, like the release."),
                io.Float.Input("text_guidance", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="Pulls toward the prompt: the release's text_guidance_scale."),
                io.Float.Input("audio_guidance", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="Pulls toward the speech: the release's audio_guidance_scale."),
                io.Int.Input("max_segments", default=0, min=0, max=100000, advanced=True, tooltip="Stop after this many segments; 0 covers the whole speech."),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler", advanced=True),
                io.Int.Input("ref_img_index", default=REF_INDEX, min=-1000, max=1000, advanced=True, tooltip=REF_INDEX_TOOLTIP),
                io.Combo.Input("encoder", options=["auto"] + list(ENCODERS), default="auto", advanced=True, tooltip=ENCODER_TOOLTIP),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, model, positive, negative, vae, audio_features, audio, seed, width, height, steps, text_guidance, audio_guidance,
                max_segments, sampler_name, ref_img_index, encoder, start_image=None):
        # the release's guidance: uncond + text * (cond - uncond_text) + audio * (uncond_text - uncond), which is Dual CFG
        # (regular) with cond1 the prompt, cond2 the negative with audio and negative the negative without
        guider = Guider_DualCFG(model)
        guider.set_cfg(text_guidance, audio_guidance)
        sigmas = (release_sigmas(model, steps, "base"), release_sigmas(model, steps, "base continuation"))

        def set_conds(values):
            guider.set_conds(*base_conds(positive, negative, values))

        return io.NodeOutput(sample_video(guider, set_conds, sigmas, vae, audio_features, audio, seed, width, height, start_image,
                                        max_segments, sampler_name, ref_img_index, encoder))


def base_conds(positive, negative, values):
    """The base mode's three conds for a segment: prompt with audio, negative with audio, negative with silent audio."""
    silent = {**values, "longcat_audio": torch.zeros_like(values["longcat_audio"])}
    return (node_helpers.conditioning_set_values(positive, values), node_helpers.conditioning_set_values(negative, values),
            node_helpers.conditioning_set_values(negative, silent))


def sample_video(guider, set_conds, sigmas, vae, audio_features, audio, seed, width, height, start_image, max_segments, sampler_name,
               ref_img_index, encoder):
    """Samples, decodes and encodes every segment; the VIDEO. set_conds(values) gives the guider a segment's conditioning
    values; sigmas is (first segment's, continued segments')."""
    segments = audio_features["segments"]
    if max_segments:
        segments = min(segments, max_segments)
    sampler = comfy.samplers.sampler_object(sampler_name)
    steps = [sigmas[min(segment, 1)].shape[-1] - 1 for segment in range(segments)]
    total = sum(steps)
    logging.info("LongCat Avatar: %d segment%s at %dx%d, %d steps for the first, %d for each next, %d in all", segments,
                 "" if segments == 1 else "s", width, height, steps[0], sigmas[1].shape[-1] - 1, total)
    writer = Writer(width, height, FPS, encoder, audio_samples(audio))
    logging.info("LongCat Avatar: encoding %dx%d with %s", width, height, writer.encoder)
    # one bar over every step of the video; the GUI's progress and previews count the same way
    preview = latent_preview.prepare_callback(guider.model_patcher, total)
    bar = tqdm(total=total, desc="LongCat Avatar", unit="step", dynamic_ncols=True, disable=not comfy.utils.PROGRESS_BAR_ENABLED)
    started = time.perf_counter()
    previous = ref_latent = None
    try:
        for segment in range(segments):
            bar.set_postfix_str(f"segment {segment + 1}/{segments}")
            values, latent, trim = segment_conditioning(vae, audio_features, width, height, LENGTH, segment, COND_FRAMES, ref_img_index,
                                                        start_image, previous, ref_latent)
            set_conds(values)

            def callback(step, x0, x, total_steps, offset=sum(steps[:segment])):
                bar.update(1)
                preview(offset + step, x0, x, total)

            noise = comfy.sample.prepare_noise(latent, seed + segment)
            samples = guider.sample(noise, latent, sampler, sigmas[min(segment, 1)], callback=callback, disable_pbar=True, seed=seed + segment)
            samples = samples.to(comfy.model_management.intermediate_device())
            if start_image is None and ref_latent is None:
                # the release's reference without an image: the first segment's first latent, no VAE round trip
                ref_latent = samples[:, :, :1].clone()
            images = decode(vae, samples)
            writer.put(images[trim:])
            previous = images[-COND_FRAMES:]
        path = writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        bar.close()
    elapsed = time.perf_counter() - started
    logging.info("LongCat Avatar: %d frames (%.1f s of video) in %.0f s, %.0f s per segment", writer.frames, writer.frames / FPS,
                 elapsed, elapsed / segments)
    return InputImpl.VideoFromFile(path)


class LongCatAvatarExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LongCatAvatarModelLoader, LongCatAvatarAudioEncode, LongCatAvatarVideo, LongCatAvatarVideoBase]


async def comfy_entrypoint() -> LongCatAvatarExtension:
    return LongCatAvatarExtension()
