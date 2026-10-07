"""BerniniR long video (beyond the 81 frames of the Wan base): one node that runs the whole clip.

source clip -> target fps (default 16, Wan's native rate) -> resize/crop to the canvas
  -> overlapping segments of `segment_frames` (default 81) frames
  -> for each segment: BerniniR Source Media + BerniniR Sampler + BerniniR Decode (the nodes of ComfyUI-BerniniR,
     called directly, so the model / text conditioning are loaded once and reused for every segment)
  -> optional chaining so the edit looks the same in every segment
  -> per-segment colour match + crossfade in the overlaps -> one video (+ the matching audio)

Needs the ComfyUI-BerniniR custom node pack (BR_MODEL / BR_VAE / BR_COND types).
"""

from __future__ import annotations

import math

from .h3_prompter import llama_client as lc
from .h3_prompter import v2v

CATEGORY = "BerniniR/Long video"
CHAIN_MODES = [
    "edited overlap as source (chain)",
    "previous frame as reference",
    "independent",
]


def _br(name):
    """A node class of ComfyUI-BerniniR from ComfyUI's global registry."""
    try:
        import nodes as comfy_nodes  # type: ignore

        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    except Exception:  # noqa: BLE001
        cls = None
    if cls is None:
        raise RuntimeError(f"ComfyUI-BerniniR is not installed or not loaded ({name} missing).")
    return cls


def _args(o):
    return o.args if hasattr(o, "args") else o


def canvas_wan(src_w: int, src_h: int, short_edge: int) -> tuple[int, int]:
    """Aspect of the source, short edge `short_edge`, area capped like 848x480 scaled, multiples of 16."""
    ratio = src_w / src_h
    cap = (short_edge * short_edge) * (848 / 480)
    w, h = (short_edge * ratio, short_edge) if ratio >= 1 else (short_edge, short_edge / ratio)
    if w * h > cap:
        s = math.sqrt(cap / (w * h))
        w, h = w * s, h * s
    return max(16, int(round(w / 16)) * 16), max(16, int(round(h / 16)) * 16)


def plan_segments(n: int, seg: int, overlap: int) -> list[int]:
    """Start indices of fixed-length segments covering n frames with AT LEAST `overlap` shared frames; the
    segments are spread evenly (equal overlaps) and the last one ends exactly on the last frame."""
    if n <= seg:
        return [0]
    k = max(2, math.ceil((n - overlap) / max(1, seg - overlap)))
    return [round(i * (n - seg) / (k - 1)) for i in range(k)]


def resample(frames, src_fps: float, dst_fps: float, start_s: float, max_s: float):
    """Nearest-frame resample of [N,H,W,C] to dst_fps between start_s and start_s+max_s."""
    import torch

    total = frames.shape[0] / src_fps
    end = min(total, start_s + max_s)
    n = max(1, int(math.floor((end - start_s) * dst_fps + 1e-6)))
    idx = torch.clamp(torch.round((start_s + torch.arange(n) / dst_fps) * src_fps).long(), 0, frames.shape[0] - 1)
    return frames[idx]


def _lab_match(seg, ref_seg, ref_target):
    """Shift `seg` so the overlap frames `ref_seg` (part of seg) get the colour statistics of `ref_target`."""
    import torch

    a = v2v._srgb_to_lab(ref_seg.float())
    b = v2v._srgb_to_lab(ref_target.float())
    mu_a, sd_a = a.mean(dim=(0, 1, 2)), a.std(dim=(0, 1, 2)).clamp(min=1e-3)
    mu_b, sd_b = b.mean(dim=(0, 1, 2)), b.std(dim=(0, 1, 2)).clamp(min=1e-3)
    ratio = (sd_b / sd_a).clamp(0.8, 1.25)
    out = []
    for chunk in torch.split(seg, 16, dim=0):
        lab = v2v._srgb_to_lab(chunk.float())
        out.append(v2v._lab_to_srgb((lab - mu_a) * ratio + mu_b).to(seg.dtype))
    return torch.cat(out, dim=0)


class BerniniRLongVideo:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "frame_count", "segments")
    DESCRIPTION = (
        "Bernini-R video edit longer than 81 frames: the clip is resampled to `fps` (16 = Wan native), split into "
        "overlapping segments, each segment runs through BerniniR Source Media / Sampler / Decode with the same model, "
        "conditioning and seed, then the segments are colour-matched and crossfaded into one video. 15 s @16 fps = "
        "240 frames = 4 segments of 81 (overlap 16). Load the model WITHOUT offload on a 96 GB card so the experts "
        "are not swapped for every segment."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BR_MODEL",),
                "vae": ("BR_VAE",),
                "cond": ("BR_COND", {"tooltip": "From BerniniR Text Encode (encoded once, used for every segment)."}),
                "source_video": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "fps": ("FLOAT", {"default": 16.0, "min": 4.0, "max": 30.0, "step": 1.0,
                                  "tooltip": "Generation fps. 16 = Wan native (81 frames = 5 s). Interpolate to "
                                             "24/30 fps afterwards (RIFE / GIMM-VFI) if needed."}),
                "max_seconds": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "segment_frames": ("INT", {"default": 81, "min": 17, "max": 129, "step": 4,
                                           "tooltip": "Frames per segment (4k+1). 81 = what Bernini-R was validated on."}),
                "overlap_frames": ("INT", {"default": 16, "min": 0, "max": 48, "step": 1,
                                           "tooltip": "Frames shared by neighbouring segments (crossfade + chaining)."}),
                "chain_mode": (CHAIN_MODES, {
                    "default": CHAIN_MODES[0],
                    "tooltip": "How a segment learns the look of the previous one. 'edited overlap as source': the "
                               "overlap frames of the source are replaced by the already edited frames. 'previous "
                               "frame as reference': the first edited frame of the overlap is added as a reference "
                               "image. 'independent': same prompt/seed/refs only.",
                }),
                "resolution": (["480p", "576p", "720p"], {"default": "480p",
                                                         "tooltip": "Short edge; aspect follows the source. Bernini-R "
                                                                    "is tested at 480p."}),
                "guidance_mode": (["auto", "rv2v", "v2v", "v2v_chain", "t2v", "r2v_apg", "v2v_apg", "t2v_apg"],
                                  {"default": "auto"}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "omega_V": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 20.0, "step": 0.05}),
                "omega_I": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.05}),
                "omega_TI": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.05}),
                "omega_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "norm_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 500.0, "step": 1.0}),
                "momentum": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s) for rv2v (same for every segment)."}),
                "source_audio": ("AUDIO", {"tooltip": "Audio of the source; comes out trimmed to the result."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "match_color_between_segments": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, model, vae, cond, source_video, source_fps, fps, max_seconds, segment_frames, overlap_frames,
            chain_mode, resolution, guidance_mode, steps, omega_V, omega_I, omega_TI, omega_scale, eta,
            norm_threshold, momentum, seed, reference_images=None, source_audio=None, start_seconds=0.0,
            match_color_between_segments=True):
        import torch

        Media, Sampler, Decode = _br("BerniniRSourceMedia"), _br("BerniniRSampler"), _br("BerniniRDecode")
        seg_len = max(5, (int(segment_frames) // 4) * 4 + 1)
        overlap = max(0, min(int(overlap_frames), seg_len - 5))

        # ---- source at the generation fps and canvas ------------------------------------------------
        src = resample(source_video[..., :3], source_fps, fps, start_seconds, max_seconds)
        short = int(resolution.rstrip("p"))
        width, height = canvas_wan(src.shape[2], src.shape[1], short)
        src = v2v.resize_frames(src, width, height).cpu()
        n = src.shape[0]
        if n < seg_len:  # short clip: one segment, snapped to 4k+1
            seg_len = max(5, ((n - 1) // 4) * 4 + 1)
            src = src[:seg_len]
            n = seg_len
        starts = plan_segments(n, seg_len, overlap)
        lc.log(f"BerniniR long: {n} frames @ {fps:g} fps ({n / fps:.2f}s) at {width}x{height}; "
               f"{len(starts)} segment(s) of {seg_len} at {starts}; chain = {chain_mode}")
        refs = reference_images[..., :3] if reference_images is not None else None  # passed as given

        out = torch.zeros((n, height, width, 3), dtype=torch.float32)
        done_until = 0  # frames [0, done_until) are final
        prev_end = 0
        for k, s in enumerate(starts):
            e = s + seg_len
            seg_src = src[s:e].clone()
            anchor = None
            ov = max(0, prev_end - s) if k > 0 else 0  # real overlap with what is already in `out`
            if k > 0 and ov > 0:
                if chain_mode.startswith("edited overlap"):
                    seg_src[:ov] = out[s:s + ov]
                elif chain_mode.startswith("previous frame"):
                    anchor = out[s:s + 1]
            lc.log(f"BerniniR long: segment {k + 1}/{len(starts)} frames {s}-{e - 1}"
                   + (f" (overlap {ov})" if ov else ""))
            br_src = dict(_args(Media().encode(vae, source_video=seg_src, reference_images=refs))[0])
            if anchor is not None:  # extra reference stream after the user's references
                extra = _args(Media().encode(vae, reference_images=anchor))[0]
                br_src["image_latents"] = list(br_src.get("image_latents", [])) + list(extra.get("image_latents", []))
            lat = _args(Sampler().sample(model, cond, guidance_mode, width, height, seg_len, steps, omega_V, omega_I,
                                         omega_TI, omega_scale, eta, norm_threshold, momentum, seed, src=br_src))[0]
            frames = _args(Decode().decode(vae, lat))[0][:seg_len].float().cpu()
            if frames.shape[1:3] != (height, width):
                frames = v2v.resize_frames(frames, width, height).cpu()
            if frames.shape[0] < seg_len:  # decoder returned fewer frames: hold the last one
                frames = torch.cat([frames, frames[-1:].expand(seg_len - frames.shape[0], -1, -1, -1)], dim=0)

            if k > 0 and ov > 0:
                if match_color_between_segments:
                    frames = _lab_match(frames, frames[:ov], out[s:s + ov])
                w = torch.linspace(0.0, 1.0, ov + 2)[1:-1].view(ov, 1, 1, 1)  # crossfade weights (0,1) exclusive
                out[s:s + ov] = out[s:s + ov] * (1 - w) + frames[:ov] * w
                out[s + ov:e] = frames[ov:]
            else:
                out[s:e] = frames
            prev_end = e
            done_until = e
            del br_src, lat
            try:
                import comfy.model_management as mm  # type: ignore

                mm.soft_empty_cache()
            except Exception:  # noqa: BLE001
                pass
        assert done_until == n

        audio = None
        if source_audio is not None:
            wf, sr = source_audio["waveform"], int(source_audio["sample_rate"])
            a = int(round(start_seconds * sr))
            want = int(round(n / fps * sr))
            wf = wf[..., a:a + want]
            if wf.shape[-1] < want:
                wf = torch.cat([wf, torch.zeros(*wf.shape[:-1], want - wf.shape[-1], dtype=wf.dtype)], dim=-1)
            audio = {"waveform": wf.contiguous(), "sample_rate": sr}
        info = ", ".join(f"{s}-{s + seg_len - 1}" for s in starts)
        return (out.clamp(0, 1), audio, float(fps), int(n), info)


NODE_CLASS_MAPPINGS = {"BerniniRLongVideo": BerniniRLongVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"BerniniRLongVideo": "BerniniR · Long Video (auto segments, 15 s+)"}

