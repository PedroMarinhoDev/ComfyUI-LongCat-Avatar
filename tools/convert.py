"""Repackage the meituan-longcat/LongCat-Video-Avatar-1.6 release as single ComfyUI files.

    python tools/convert.py <release folder> <ComfyUI models folder>

* base_model/ (6 shards)          -> diffusion_models/longcat-video-avatar/longcat_video_avatar_1.6_bf16.safetensors (keys unchanged)
* loraA/ (DMD), loraB/ (OPSD)     -> loras/longcat-video-avatar/longcat_video_avatar_1.6_{dmd,opsd}_lora_bf16.safetensors
* whisper-large-v3/               -> audio_encoders/whisper_large_v3_encoder_fp16.safetensors (encoder only)

The release's LoRAs split fused projections (attn.qkv, cross_attn.kv_linear) into per-part up matrices over
chunks of one down matrix. As a standard LoRA that is one down matrix and a block-diagonal up matrix; alpha is set
so alpha / rank equals the stored alpha_scale. They are stored in bf16, the dtype the release applies them in.
Existing outputs are skipped; each file is written to <name>.part and renamed when complete.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def write(path, tensors, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path + ".part", metadata=metadata)
    os.replace(path + ".part", path)
    print(f"wrote {path} ({len(tensors)} tensors, {os.path.getsize(path) / 2**30:.2f} GiB)")


def convert_dit(release, out):
    folder = os.path.join(release, "base_model")
    weight_map = json.load(open(os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(folder, shard), "pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(torch.bfloat16)
    assert set(tensors) == set(weight_map), "shards do not match the index"
    write(out, tensors)


def convert_lora(path, out):
    tensors = {}
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        names = sorted({k.split(".lora_down.weight")[0] for k in keys if k.endswith(".lora_down.weight")})
        for name in names:
            module = name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            down = f.get_tensor(name + ".lora_down.weight").float()
            parts = sorted(k for k in keys if k.startswith(name + ".lora_up.blocks."))
            if parts:
                ups = [f.get_tensor(k).float() for k in parts]
                up = torch.block_diag(*ups)
            else:
                up = f.get_tensor(name + ".lora_up.weight").float()
            alpha = f.get_tensor(name + ".alpha_scale").float() * down.shape[0]
            prefix = "diffusion_model." + module
            tensors[prefix + ".lora_down.weight"] = down.to(torch.bfloat16)
            tensors[prefix + ".lora_up.weight"] = up.to(torch.bfloat16)
            tensors[prefix + ".alpha"] = alpha
    write(out, tensors)


def convert_whisper(folder, out):
    with safe_open(os.path.join(folder, "model.safetensors"), "pt") as f:
        tensors = {k: f.get_tensor(k).to(torch.float16) for k in f.keys() if k.startswith("model.encoder.")}
    write(out, tensors)


def main(release, models):
    jobs = [
        (convert_dit, release, os.path.join(models, "diffusion_models", "longcat-video-avatar", "longcat_video_avatar_1.6_bf16.safetensors")),
        (convert_lora, os.path.join(release, "loraA", "lora.safetensors"),
         os.path.join(models, "loras", "longcat-video-avatar", "longcat_video_avatar_1.6_dmd_lora_bf16.safetensors")),
        (convert_lora, os.path.join(release, "loraB", "lora.safetensors"),
         os.path.join(models, "loras", "longcat-video-avatar", "longcat_video_avatar_1.6_opsd_lora_bf16.safetensors")),
        (convert_whisper, os.path.join(release, "whisper-large-v3"), os.path.join(models, "audio_encoders", "whisper_large_v3_encoder_fp16.safetensors")),
    ]
    for fn, src, out in jobs:
        if os.path.exists(out):
            print(f"exists, skipped: {out}")
            continue
        fn(src, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
