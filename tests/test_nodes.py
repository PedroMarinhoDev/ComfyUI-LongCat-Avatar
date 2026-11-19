"""The nodes' parts that run without the model, on CPU: segment counts, audio windows, the reference latent, the
speech's peak, the base mode's conds and guidance, the shared segment loop with a stub guider and VAE, the loader's
check, and the shipped workflows against the node schemas.

    cd tests && ~/AI/ComfyUI/.venv/bin/python -m pytest -q test_nodes.py
"""
import json
import os
import types

import av
import pytest
import safetensors.torch
import torch

import comfy.samplers
import folder_paths
from test_scheduler import PACK, load_nodes

nodes = load_nodes()
from longcat_pack.longcat_avatar import loader  # noqa: E402  (load_nodes imports the pack as longcat_pack)


class StubVAE:
    def __init__(self):
        self.encoded = []

    def encode(self, images):
        self.encoded.append(images.shape[0])
        return torch.full([1, 16, 1 + (images.shape[0] - 1) // 4, images.shape[1] // 8, images.shape[2] // 8], float(len(self.encoded)))

    def decode(self, samples):
        frames = 1 + (samples.shape[2] - 1) * 4
        return torch.rand(1, frames, samples.shape[3] * 8, samples.shape[4] * 8, 3)


def features(frames, segments=0):
    # frame t's features are all t, so the window shows which frames it read
    return {"features": torch.arange(frames, dtype=torch.float32).view(frames, 1, 1).expand(frames, 5, 2), "segments": segments}


@pytest.mark.parametrize("frames, segments", [(0, 1), (10, 1), (93, 1), (94, 2), (173, 2), (174, 3)])
def test_segment_count(frames, segments):
    assert nodes.segment_count(frames) == segments


def windows(values):
    return values["longcat_audio"][0, :, :, 0, 0]


def test_audio_window_first_segment():
    values, latent, trim = nodes.segment_conditioning(StubVAE(), features(200), 16, 16, 93, 0, 13, 10)
    assert windows(values).shape == (93, 5)
    assert windows(values)[0].tolist() == [0, 0, 0, 1, 2]
    assert windows(values)[1].tolist() == [0, 0, 1, 2, 3]
    assert windows(values)[92].tolist() == [90, 91, 92, 93, 94]
    assert trim == 0 and latent.shape == (1, 16, 24, 2, 2)


def test_audio_window_later_segment():
    previous = torch.rand(93, 16, 16, 3)
    values, _, trim = nodes.segment_conditioning(StubVAE(), features(170), 16, 16, 93, 1, 13, 10, previous_frames=previous)
    assert trim == 13
    assert windows(values)[0].tolist() == [78, 79, 80, 81, 82]
    # frames past the features read the last one
    assert windows(values)[88].tolist() == [166, 167, 168, 169, 169]
    assert windows(values)[92].tolist() == [169, 169, 169, 169, 169]


def test_reference_latent():
    previous = torch.rand(93, 16, 16, 3)
    vae = StubVAE()
    values, _, _ = nodes.segment_conditioning(vae, features(200), 16, 16, 93, 1, 13, 10, previous_frames=previous)
    assert vae.encoded == [13, 1] and values["longcat_ref_latent"].eq(2).all() and values["longcat_ref_index"] == 10
    # a given reference latent is used as it is, without encoding anything for it
    vae = StubVAE()
    ref = torch.randn(1, 16, 1, 2, 2)
    values, _, _ = nodes.segment_conditioning(vae, features(200), 16, 16, 93, 1, 13, 10, previous_frames=previous, ref_latent=ref)
    assert vae.encoded == [13] and values["longcat_ref_latent"] is ref


def test_speech_peak():
    loud = torch.zeros(1, 2, 16000)
    loud[0, 1, 100] = -1.8
    loud[0, 0, 200] = 0.5
    wave = nodes.speech({"waveform": loud, "sample_rate": 16000})
    # the peak over both channels becomes 0.9, then the channels are averaged
    assert wave.shape == (1, 1, 16000)
    assert wave[0, 0, 100].item() == pytest.approx(-0.45) and wave[0, 0, 200].item() == pytest.approx(0.125)
    quiet = torch.zeros(1, 1, 16000)
    quiet[0, 0, 100] = 0.3
    assert nodes.speech({"waveform": quiet, "sample_rate": 16000})[0, 0, 100].item() == pytest.approx(0.3)


def test_base_conds():
    positive, negative = [[torch.randn(1, 4, 8), {}]], [[torch.randn(1, 4, 8), {}]]
    audio, cond = torch.randn(1, 93, 5, 5, 2), torch.randn(1, 16, 4, 2, 2)
    values = {"longcat_audio": audio, "longcat_cond_latent": cond, "longcat_ref_index": 10}
    cond1, cond2, uncond = nodes.base_conds(positive, negative, values)
    assert cond1[0][0] is positive[0][0] and cond2[0][0] is negative[0][0] and uncond[0][0] is negative[0][0]
    assert cond1[0][1]["longcat_audio"] is audio and cond2[0][1]["longcat_audio"] is audio
    assert uncond[0][1]["longcat_audio"].eq(0).all() and uncond[0][1]["longcat_audio"].shape == audio.shape
    for c in (cond1, cond2, uncond):
        assert c[0][1]["longcat_cond_latent"] is cond and c[0][1]["longcat_ref_index"] == 10
    assert values["longcat_audio"] is audio


def test_base_guidance_is_the_release_formula(monkeypatch):
    # Guider_DualCFG fed base_conds' order: uncond + text * (cond - uncond_text) + audio * (uncond_text - uncond)
    preds = {"cond": torch.tensor(5.0), "uncond_text": torch.tensor(2.0), "uncond": torch.tensor(1.0)}
    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", lambda model, conds, *args: [preds[c] for c in conds])
    guider = nodes.Guider_DualCFG(type("Patcher", (), {"model_options": {}})())
    guider.set_cfg(4.0, 3.0)
    guider.conds = {"positive": "cond", "middle": "uncond_text", "negative": "uncond"}
    guider.inner_model = None
    got = guider.predict_noise(torch.zeros(1), torch.zeros(1))
    assert got.item() == pytest.approx(1.0 + 4.0 * (5.0 - 2.0) + 3.0 * (2.0 - 1.0))


class StubGuider:
    model_patcher = None

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def sample(self, noise, latent, sampler, sigmas, callback=None, disable_pbar=False, seed=None):
        segment = len(self.calls)
        if segment == self.fail_at:
            raise RuntimeError("stop")
        self.calls.append((sigmas.shape[-1] - 1, seed, self.values))
        for step in range(sigmas.shape[-1] - 1):
            callback(step, None, None, sigmas.shape[-1] - 1)
        return torch.full_like(latent, float(segment + 1))


def run_video(monkeypatch, tmp_path, guider, start_image=None, max_segments=0, sigmas=(torch.linspace(1, 0, 6), torch.linspace(1, 0, 3))):
    folder_paths.set_temp_directory(str(tmp_path))
    previews = []
    monkeypatch.setattr(nodes.latent_preview, "prepare_callback", lambda patcher, total: lambda step, x0, x, total_: previews.append((step, total_)))

    def set_conds(values):
        guider.values = values

    audio = {"waveform": torch.zeros(1, 1, 16000 * 12), "sample_rate": 16000}
    video = nodes.sample_video(guider, set_conds, sigmas, StubVAE(), features(400, segments=3), audio, 7, 64, 64, start_image, max_segments,
                               "euler", 10, "h264")
    return video, previews


def test_video_segments(monkeypatch, tmp_path):
    guider = StubGuider()
    video, previews = run_video(monkeypatch, tmp_path, guider)
    # 5 steps for the first segment, 2 for each next; previews count on over the whole video
    assert [c[0] for c in guider.calls] == [5, 2, 2] and [c[1] for c in guider.calls] == [7, 8, 9]
    assert previews == [(step, 9) for step in range(9)]
    # without a start image the reference is the first segment's first latent (the stub fills segment 1's with 1)
    assert "longcat_ref_latent" not in guider.calls[0][2]
    for _, _, values in guider.calls[1:]:
        assert values["longcat_ref_latent"].shape == (1, 16, 1, 8, 8) and values["longcat_ref_latent"].eq(1).all()
    with av.open(video.get_stream_source()) as container:
        assert sum(1 for _ in container.decode(video=0)) == 93 + 80 + 80


def test_video_start_image_reference(monkeypatch, tmp_path):
    guider = StubGuider()
    run_video(monkeypatch, tmp_path, guider, start_image=torch.rand(1, 64, 64, 3))
    # the start image is encoded for the reference, as before
    assert guider.calls[1][2]["longcat_ref_latent"].ne(1).all()


def test_video_error_cleans_up(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="stop"):
        run_video(monkeypatch, tmp_path, StubGuider(fail_at=1))
    assert os.listdir(tmp_path / "longcat_avatar") == []


def test_base_smoke_test_settings(monkeypatch, tmp_path):
    # a quick check of the base node: 2 of the 3 segments at 2 steps runs the first and the continued schedule once each
    model = types.SimpleNamespace(get_model_object=lambda name: types.SimpleNamespace(shift=7.0))
    sigmas = (nodes.release_sigmas(model, 2, "base"), nodes.release_sigmas(model, 2, "base continuation"))
    # the continued schedule keeps the steps above 0.5 (here one) and always ends on the release's 10 even ones
    assert sigmas[0].shape[-1] == 3 and sigmas[1].shape[-1] == 12 and sigmas[1][-2].item() == pytest.approx(0.05)
    guider = StubGuider()
    video, previews = run_video(monkeypatch, tmp_path, guider, max_segments=2, sigmas=sigmas)
    assert [c[0] for c in guider.calls] == [2, 11] and previews[-1] == (12, 13)
    with av.open(video.get_stream_source()) as container:
        assert sum(1 for _ in container.decode(video=0)) == 93 + 80
    inputs = {i.id: i for i in nodes.LongCatAvatarVideoBase.define_schema().inputs}
    assert inputs["steps"].min == 1 and inputs["max_segments"].min == 0


def test_loader_rejects_other_checkpoint(tmp_path):
    sd = {k: torch.zeros(4, 4) for k in loader.KEYS if k != "audio_proj.norm.weight"}
    path = tmp_path / "wan.safetensors"
    safetensors.torch.save_file(sd, str(path))
    with pytest.raises(ValueError, match="wan.safetensors is not a LongCat-Video-Avatar checkpoint.*audio_proj.norm.weight"):
        loader.load_longcat_avatar(str(path))


PACK_NODES = {cls.define_schema().node_id: cls for cls in
              (nodes.LongCatAvatarModelLoader, nodes.LongCatAvatarAudioEncode, nodes.LongCatAvatarVideo, nodes.LongCatAvatarVideoBase)}
WIDGETS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def test_node_list():
    import asyncio
    listed = asyncio.run(asyncio.run(nodes.comfy_entrypoint()).get_node_list())
    assert {cls.define_schema().node_id for cls in listed} == set(PACK_NODES)


@pytest.mark.parametrize("name", ["longcat_avatar_video", "longcat_avatar_video_base"])
def test_workflow_matches_schemas(name):
    """The pack's nodes in a shipped workflow: links on inputs of the right type, every required input linked, and
    one widget value per widget input in schema order (seed adds its control value)."""
    with open(os.path.join(PACK, "example_workflows", f"{name}.json")) as f:
        workflow = json.load(f)
    links = {link[0]: link for link in workflow["links"]}
    found = 0
    for node in workflow["nodes"]:
        if node["type"] not in PACK_NODES:
            continue
        found += 1
        inputs = PACK_NODES[node["type"]].define_schema().inputs
        linked = {i["name"]: links[i["link"]][5] for i in node.get("inputs", []) if i.get("link") is not None}
        for i in inputs:
            kind = i.get_io_type()
            if i.id in linked:
                assert linked[i.id] == kind, (node["type"], i.id)
            elif kind not in WIDGETS:
                assert i.optional, f"{node['type']}: {i.id} is not connected"
        values = list(node.get("widgets_values") or [])
        for i in inputs:
            if i.get_io_type() not in WIDGETS:
                continue
            value = values.pop(0)
            if i.get_io_type() == "INT":
                assert isinstance(value, int) and i.min <= value <= i.max, (node["type"], i.id, value)
            if getattr(i, "control_after_generate", False):
                assert values.pop(0) in ("fixed", "increment", "decrement", "randomize")
        assert values == [], (node["type"], values)
    assert found >= 3
