"""Plain round-to-nearest quantized copies of the converted DiT, in ComfyUI's native formats (no GPTQ, no compression).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/AI/ComfyUI/.venv/bin/python tools/quantize.py <bf16 file> {int8_convrot,w6a8,w4a8}

* int8_convrot: every block linear int8 ConvRot (rowwise scales, 256-wide Hadamard groups).
* w6a8 / w4a8: the block linears in w6a8_int8 / asym_w4a8_int8 (16-wide groups), except the adaLN modulation
  layers, which stay int8 ConvRot; their outputs scale and gate every residual.
* Embedders, the audio projection and the final layer stay bf16 in all three.

Quantization goes through the Comfy Kitchen layouts' own quantize() and QuantizedTensor.state_dict, the same code
ComfyUI uses to requantize and save these formats. Writes <name>.part, then renames. Prints each layer's output
error on random inputs through the quantized matmul the model runs (rotation included).
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.expanduser("~/AI/ComfyUI"))

import comfy.quant_ops  # noqa: E402,F401  (registers the layouts)
from comfy.quant_ops import QuantizedTensor  # noqa: E402

FORMATS = {
    "int8_convrot": ("int8_tensorwise", "TensorWiseINT8Layout", {"per_channel": True, "convrot": True, "convrot_groupsize": 256}),
    "w6a8": ("w6a8_int8", "AsymW4A8Int8Layout", {"bits": 6, "group_size": 16, "convrot_groupsize": 256}),
    "w4a8": ("asym_w4a8_int8", "AsymW4A8Int8Layout", {"bits": 4, "group_size": 16, "convrot_groupsize": 256}),
}


def layer_format(key, target):
    if not key.startswith("blocks.") or not key.endswith(".weight") or "norm" in key:
        return None
    if "adaLN_modulation" in key:
        return "int8_convrot"
    return target


def quantize(weight, fmt):
    quant_format, layout, kwargs = FORMATS[fmt]
    qt = QuantizedTensor.from_float(weight, layout, **kwargs)
    conf = {"format": quant_format}
    if quant_format == "int8_tensorwise":
        conf.update(convrot=True, convrot_groupsize=kwargs["convrot_groupsize"])
    else:
        conf.update(group_size=kwargs["group_size"], convrot_groupsize=kwargs["convrot_groupsize"])
    return qt, conf


def main(src, target):
    name = os.path.basename(src).replace("_bf16", f"_{target}")
    out = os.path.join(os.path.dirname(src), name)
    if os.path.exists(out):
        print(f"exists, skipped: {out}")
        return
    device = torch.device("cuda")
    tensors, errors, start = {}, {}, time.time()
    gen = torch.Generator(device=device).manual_seed(0)
    with safe_open(src, "pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            fmt = layer_format(key, target)
            if fmt is None:
                tensors[key] = tensor
                continue
            weight = tensor.to(device)
            qt, conf = quantize(weight, fmt)
            x = torch.randn(256, weight.shape[1], device=device, dtype=torch.bfloat16, generator=gen)
            ref = x @ weight.T
            errors[key] = ((torch.nn.functional.linear(x, qt).float() - ref.float()).norm() / ref.float().norm()).item()
            prefix = key[:-len("weight")]
            for k, v in qt.state_dict(prefix + "weight").items():
                tensors[k] = v.cpu()
            tensors[prefix + "comfy_quant"] = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
    save_file(tensors, out + ".part")
    os.replace(out + ".part", out)

    kinds = {}
    for key, err in errors.items():
        kind = ".".join(p for p in key.split(".")[2:-1])
        kinds.setdefault(kind, []).append(err)
    print(f"wrote {out} ({os.path.getsize(out) / 2**30:.2f} GiB, {time.time() - start:.0f} s, {len(errors)} quantized layers)")
    for kind, errs in sorted(kinds.items()):
        print(f"  {kind:32s} output error mean {sum(errs) / len(errs) * 100:.2f}%  max {max(errs) * 100:.2f}%")
    all_errs = list(errors.values())
    print(f"  all layers mean {sum(all_errs) / len(all_errs) * 100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
