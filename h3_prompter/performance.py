"""Bounded preprocessing reuse and wall-clock stage timings for the compact V2V node."""
import hashlib
import json
import os
import re
import time


def file_stamp(path):
    if path is None:
        return None
    # llama.cpp split GGUFs load every shard, so invalidate on changes to any part.
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path, re.I)
    paths = [path]
    if match:
        paths = [path[:match.start()] + f"-{i:05d}-of-{int(match[2]):05d}.gguf"
                 for i in range(1, int(match[2]) + 1)]
    return tuple((os.path.realpath(p), os.stat(p).st_mtime_ns, os.stat(p).st_size) for p in paths)


def request_key(*values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def image_key(image):
    # Hash all pixels of the two Qwen input images, not a sparse fingerprint.
    # ComfyUI IMAGE tensors are float32; float() also supports bfloat16 sources.
    array = image.detach().float().cpu().contiguous().numpy()
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return str(image.dtype), tuple(array.shape), digest


class SingleEntryCache:
    """One result per node; caches neither diffusion weights nor GPU tensors."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.key, self.value = None, None

    def get(self, key):
        return self.value if key == self.key else None

    def put(self, key, value):
        self.key, self.value = key, value


class StageTimer:
    """Host wall time; no CUDA synchronization is injected into generation."""
    def __init__(self, log):
        self.log = log
        self.start = self.last = time.perf_counter()
        self.stages = []

    def mark(self, name):
        now = time.perf_counter()
        elapsed = now - self.last
        self.stages.append((name, elapsed))
        self.last = now
        self.log(f"V2V timing: {name}={elapsed:.2f}s")

    def finish(self):
        self.log("V2V timing total: " + f"{time.perf_counter() - self.start:.2f}s; " +
                 "; ".join(f"{name}={seconds:.2f}s" for name, seconds in self.stages) +
                 " (wall time; asynchronous GPU work may cross stage boundaries)")


def take_frames(frames, indices):
    """Use a view for a contiguous clip instead of copying the whole IMAGE batch."""
    if indices and all(index == indices[0] + offset for offset, index in enumerate(indices)):
        return frames[indices[0]:indices[-1] + 1]
    import torch
    return frames[torch.tensor(indices, device=frames.device, dtype=torch.long)]
