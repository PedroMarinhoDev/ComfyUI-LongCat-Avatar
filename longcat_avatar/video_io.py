"""Streaming video output: each segment is encoded as soon as it is decoded, so a video's length costs disk, not RAM.

Adapted from ComfyUI-ReCaVSR's recavsr/video_io.py. Frames become uint8 and are copied into one of two CPU slots
(pinned with CUDA); a thread encodes each slot while the next segment computes. NVENC runs on the GPU's video
encoder, beside the CUDA cores; x264 on the CPU.
"""
import math
import os
import queue
import threading
import uuid
from fractions import Fraction

import av
import numpy as np
import torch

import folder_paths

# name: (PyAV codec, options), in fallback order
ENCODERS = {
    "hevc nvenc": ("hevc_nvenc", {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": "19", "b": "0"}),
    "h264 nvenc": ("h264_nvenc", {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": "17", "b": "0"}),
    "h264": ("libx264", {"crf": "14", "preset": "medium"}),
}


def pick_encoder(choice, width, height, fps):
    """The first encoder that opens at this size, from the choice down the fallback order (auto starts at the top):
    NVENC is missing without an NVIDIA GPU or in a PyAV build without it, and its H.264 stops at 4096 pixels a side."""
    names = list(ENCODERS)
    for name in names[0 if choice == "auto" else names.index(choice):]:
        codec, options = ENCODERS[name]
        try:
            context = av.CodecContext.create(codec, "w")
            context.width, context.height, context.pix_fmt = width, height, "yuv420p"
            context.framerate, context.time_base = fps, 1 / fps
            context.options = dict(options)
            context.open()
            del context
            return name
        except (av.FFmpegError, ValueError):
            continue
    raise RuntimeError(f"no video encoder opens at {width}x{height}")


class Writer:
    """Encodes IMAGE frames [n, H, W, 3] in [0, 1] into an mp4 in ComfyUI's temp folder, with the speech muxed in; the
    video ends where the speech does."""

    def __init__(self, width, height, fps, encoder, audio=None, chunk=96, slots=2):
        fps = Fraction(fps).limit_denominator(1001)
        encoder = pick_encoder(encoder, width, height, fps)
        codec, options = ENCODERS[encoder]
        folder = os.path.join(folder_paths.get_temp_directory(), "longcat_avatar")
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, f"{uuid.uuid4().hex}.mp4")
        self.encoder, self.width, self.height, self.fps, self.frames = encoder, width, height, fps, 0
        self.container = av.open(self.path, "w")
        self.stream = self.container.add_stream(codec, rate=fps, options=dict(options))
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"
        # audio = (samples [C, S] float32, rate); every stream before the first packet, the header is written then.
        # The video ends with it: frames past the audio's last one are dropped.
        self.audio = audio
        self.max_frames = None
        if audio is not None:
            self.max_frames = math.ceil(audio[0].shape[1] / audio[1] * fps)
            self.audio_stream = self.container.add_stream("aac", rate=audio[1], layout="stereo" if audio[0].shape[0] > 1 else "mono")
        # open the encoder now: NVENC takes its VRAM at open, and by the first frame the model may have filled the card
        self.container.start_encoding()
        self.chunk = chunk
        self.slots = [torch.empty((chunk, height, width, 3), dtype=torch.uint8, pin_memory=torch.cuda.is_available()) for _ in range(slots)]
        self.free, self.work = queue.Queue(), queue.Queue()
        for i in range(slots):
            self.free.put(i)
        self.error = None
        # closed: close() or abort() ran, no more frames; finished: the file is complete and abort() keeps it
        self.closed = self.finished = False
        self.thread = threading.Thread(target=self._encode, daemon=True)
        self.thread.start()

    def put(self, images):
        """Queue frames for encoding; returns once they are copied, while the encoder works on."""
        if self.closed:
            raise RuntimeError("the video writer is already closed")
        if images.shape[1:3] != (self.height, self.width):
            raise ValueError(f"frames are {images.shape[2]}x{images.shape[1]}, the video is {self.width}x{self.height}")
        if self.max_frames is not None:
            images = images[:max(0, self.max_frames - self.frames)]
        for start in range(0, images.shape[0], self.chunk):
            if self.error:
                raise self.error
            frames = images[start:start + self.chunk].clamp(0, 1).mul(255).round().to(torch.uint8)
            slot = self.free.get()
            n = frames.shape[0]
            self.slots[slot][:n].copy_(frames)
            self.work.put((slot, n))
            self.frames += n

    def _encode(self):
        while (item := self.work.get()) is not None:
            slot, n = item
            try:
                if self.error is None:
                    for frame in self.slots[slot][:n].numpy():
                        self.container.mux(self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            except Exception as e:  # raised again in the node's thread by put() or close()
                self.error = e
            self.free.put(slot)

    def abort(self):
        """Stop the encoder thread and drop the unfinished file, after an error or a cancel."""
        if self.finished:
            return
        self.closed = True
        if self.thread.is_alive():
            self.error = self.error or RuntimeError("aborted")
            self.work.put(None)
            self.thread.join()
        try:
            self.container.close()
        except av.FFmpegError:
            pass
        if os.path.exists(self.path):
            os.remove(self.path)

    def close(self):
        """Finish the file and return its path."""
        if self.closed:
            raise RuntimeError("the video writer is already closed")
        self.closed = True
        self.work.put(None)
        self.thread.join()
        if self.error:
            raise self.error
        self.container.mux(self.stream.encode())
        if self.audio is not None:
            samples, rate = self.audio
            samples = samples[:2, :round(self.frames / self.fps * rate)]
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, dtype=np.float32), format="fltp", layout=self.audio_stream.layout.name)
            frame.sample_rate = rate
            self.container.mux(self.audio_stream.encode(frame))
            self.container.mux(self.audio_stream.encode())
        self.container.close()
        self.finished = True
        return self.path


def audio_samples(audio):
    """ComfyUI AUDIO -> (samples [C, S] float32, rate) for the writer."""
    return audio["waveform"][0].float().cpu().numpy(), audio["sample_rate"]
