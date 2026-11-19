"""release_sigmas against the release's get_timesteps_sigmas + FlowMatchEulerDiscreteScheduler (shift 7), on CPU.

    cd tests && ~/AI/ComfyUI/.venv/bin/python -m pytest -q
"""
import importlib.util
import os
import sys
import types

import pytest
import torch

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(PACK, "ref", "LongCat-Video")


def load_nodes():
    spec = importlib.util.spec_from_file_location("longcat_pack", os.path.join(PACK, "__init__.py"), submodule_search_locations=[PACK])
    module = importlib.util.module_from_spec(spec)
    sys.modules["longcat_pack"] = module
    spec.loader.exec_module(module)
    return importlib.import_module("longcat_pack.nodes")


def reference_sigmas(steps, distilled, continuation=False):
    sys.path.insert(0, REF)
    import diffusers  # noqa: F401  (before the stubs: it probes for ftfy)
    for name in ("pyloudnorm", "loguru", "ftfy"):
        sys.modules.setdefault(name, types.ModuleType(name))
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    pipe = types.SimpleNamespace(model_type="avatar-v1.5", num_timesteps=1000, num_distill_sample_steps=8)
    sigmas = LongCatVideoAvatarPipeline.get_timesteps_sigmas(pipe, steps, use_distill=distilled)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    scheduler.set_timesteps(len(sigmas), sigmas=sigmas)
    if continuation:
        # generate_avc enhance_hf
        tail = [torch.tensor(t) for t in torch.linspace(500, 0, 11)[:-1].tolist()]
        timesteps = torch.cat([t.unsqueeze(0) for t in scheduler.timesteps if t > 500] + [t.unsqueeze(0) for t in tail])
        return torch.cat([timesteps / 1000, torch.zeros(1)])
    return scheduler.sigmas


@pytest.mark.skipif(not os.path.isdir(REF), reason="reference clone missing (ref/LongCat-Video)")
@pytest.mark.parametrize("steps,schedule", [(8, "distilled"), (50, "base"), (50, "base continuation")])
def test_scheduler_matches_release(steps, schedule):
    nodes = load_nodes()
    model = types.SimpleNamespace(get_model_object=lambda name: types.SimpleNamespace(shift=7.0))
    got = nodes.release_sigmas(model, steps, schedule)
    expected = reference_sigmas(steps, schedule == "distilled", schedule == "base continuation")
    assert got.shape == expected.shape
    assert torch.allclose(got, expected.float(), atol=1e-6), (got, expected)
