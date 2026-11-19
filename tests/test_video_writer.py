"""The streaming video writer on CPU with x264: a write with audio, and use after close or abort.

    cd tests && ~/AI/ComfyUI/.venv/bin/python -m pytest -q test_video_writer.py
"""
import os

import av
import numpy as np
import pytest
import torch

import folder_paths
from test_scheduler import load_nodes

load_nodes()
from longcat_pack.longcat_avatar import video_io  # noqa: E402  (load_nodes imports the pack as longcat_pack)


@pytest.fixture(autouse=True)
def temp_directory(tmp_path):
    folder_paths.set_temp_directory(str(tmp_path))


def frames(n):
    return torch.rand(n, 64, 64, 3)


def writer(audio=None):
    return video_io.Writer(64, 64, 25, "h264", audio)


def test_segments_with_audio():
    # 3.2 s of speech at 25 fps: 80 frames, so the third 30-frame segment is cut short
    w = writer((np.zeros((1, 16000 * 32 // 10), dtype=np.float32), 16000))
    for _ in range(3):
        w.put(frames(30))
    path = w.close()
    with av.open(path) as container:
        assert sum(1 for _ in container.decode(video=0)) == 80
        assert len(container.streams.audio) == 1
    assert not w.thread.is_alive()


def test_wrong_size_raises():
    w = writer()
    with pytest.raises(ValueError, match="32x32"):
        w.put(torch.rand(4, 32, 32, 3))
    w.abort()


def test_put_after_close_raises():
    w = writer()
    w.put(frames(8))
    path = w.close()
    with pytest.raises(RuntimeError, match="closed"):
        w.put(frames(8))
    with pytest.raises(RuntimeError, match="closed"):
        w.close()
    # an abort after a finished close keeps the file
    w.abort()
    assert os.path.exists(path)


def test_put_after_abort_raises():
    w = writer()
    w.put(frames(8))
    w.abort()
    with pytest.raises(RuntimeError, match="closed"):
        w.put(frames(8))
    assert not w.thread.is_alive() and not os.path.exists(w.path)
