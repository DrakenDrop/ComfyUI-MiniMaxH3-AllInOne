"""Canvas and timeline rules for the compact generate nodes; no GPU imports."""
import math

RESOLUTIONS = ["360p", "480p", "768p (native)"]
ASPECTS = ["same as reference", "9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "custom"]


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
    short = {"360p": 352, "480p": 480, "768p (native)": 768}[resolution]
    ratio = w / h
    nw, nh = (short * ratio, short) if ratio >= 1 else (short, short / ratio)
    # Native H3 uses the official 768*1344 area cap. Low presets keep their short edge.
    if resolution == "768p (native)" and nw * nh > 768 * 1344:
        scale = math.sqrt(768 * 1344 / (nw * nh))
        nw, nh = nw * scale, nh * scale
    return max(32, round(nw / 32) * 32), max(32, round(nh / 32) * 32)


def video_timeline(frame_count, fps, start_seconds=0.0, max_seconds=15.1):
    """Snap DOWN: never invent/hold frames to claim an exact motion match."""
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source video must contain frames and a positive finite FPS")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and non-negative")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    available = frame_count / fps - start_seconds
    n = min(362, math.floor(min(available, max_seconds) * 24 + 1e-6))
    if n < 5:
        raise ValueError("Selected source segment needs at least 5 frames at 24 FPS")
    n = 5 + 17 * ((n - 5) // 17)
    indices = [min(frame_count - 1, int((start_seconds + j / 24) * fps)) for j in range(n)]
    return n, indices


def generation_frames(seconds):
    if not math.isfinite(seconds) or not 5 / 24 <= seconds <= 362 / 24:
        raise ValueError("H3 duration must be between 0.21 and 15.08 seconds")
    return min(362, 5 + 17 * math.ceil((seconds * 24 - 5) / 17))
