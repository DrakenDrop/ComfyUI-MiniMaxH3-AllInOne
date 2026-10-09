"""Canvas and timeline rules for the compact generate nodes; no GPU imports."""
import math

RESOLUTIONS = ["360p", "480p", "768p (native)"]
ASPECTS = ["same as reference", "9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "custom"]

# Official H3 pixel buckets: 480p renders 832x480 / 480x832 / 640x640, native 768p
# renders 1344x768 / 768x1344 / 992x992. Each preset keeps a constant pixel budget
# for EVERY aspect ratio. 360p keeps its historical 640x352 budget.
AREA_BUDGETS = {"360p": 640 * 352, "480p": 832 * 480, "768p (native)": 768 * 1344}
_NATIVE_SQUARE = 992  # official square bucket; the raw sqrt rounds to 1024 instead


def canvas(resolution, aspect_ratio, reference_width, reference_height, custom_aspect="16:9"):
    if resolution not in RESOLUTIONS:
        raise ValueError("Unknown H3 resolution preset")
    if aspect_ratio == "same as reference":
        w, h = float(reference_width), float(reference_height)
    else:
        text = custom_aspect if aspect_ratio == "custom" else aspect_ratio
        try:
            w, h = map(float, text.strip().split(":"))
        except (ValueError, AttributeError):
            raise ValueError("Custom aspect ratio must be width:height, for example 5:4") from None
    if not all(math.isfinite(x) and x > 0 for x in (w, h)) or not 0.1 <= w / h <= 10:
        raise ValueError("Aspect ratio must be positive and between 1:10 and 10:1")
    # Every preset fixes a pixel BUDGET, not a short edge, so any aspect ratio
    # (including same-as-reference) gets the same pixel count as the official H3
    # buckets: 480p = 832*480 (~0.4 MP, buckets 832x480 / 480x832 / 640x640),
    # native 768p = 768*1344 (buckets 1344x768 / 768x1344 / 992x992).
    budget = AREA_BUDGETS[resolution]
    ratio = w / h
    if resolution == "768p (native)" and abs(ratio - 1.0) < 1e-9:
        return _NATIVE_SQUARE, _NATIVE_SQUARE
    nw, nh = math.sqrt(budget * ratio), math.sqrt(budget / ratio)
    return max(32, round(nw / 32) * 32), max(32, round(nh / 32) * 32)


def _aligned_frames(seconds):
    """Official duration expression, within this node's 362-frame limit."""
    base = max(5, round(seconds * 24))
    return min(362, base + (5 - (base % 17)) % 17)


def video_timeline(frame_count, fps, start_seconds=0.0, max_seconds=15.08):
    """Align up like the official workflow; repeat the tail only if source ends."""
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source video must contain frames and a positive finite FPS")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and non-negative")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    available = frame_count / fps - start_seconds
    duration = min(available, max_seconds)
    if math.floor(duration * 24 + 1e-6) < 5:
        raise ValueError("Selected source segment needs at least 5 frames at 24 FPS")
    n = _aligned_frames(duration)
    indices = [min(frame_count - 1, math.floor(start_seconds * fps + j * fps / 24 + 1e-7)) for j in range(n)]
    return n, indices


def generation_frames(seconds):
    if not math.isfinite(seconds) or not 5 / 24 <= seconds <= 362 / 24:
        raise ValueError("H3 duration must be between 0.21 and 15.08 seconds")
    return _aligned_frames(seconds)
