"""Run: python -m unittest discover -s tests -v (no ComfyUI/GPU needed)."""
import importlib.util
import math
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load("test_canvas", "h3_prompter/canvas.py")
formats = load("test_prompt_format", "h3_prompter/prompt_format.py")
video_only = load("test_video_only", "h3_prompter/video_only.py")
PROMPT = "\n\n".join(f"{f}:\n{body}" for f, body in zip(formats.FIELDS, (
    "<Subject 1> is the person from <Picture 1>.", "[reference generation] A person walks.",
    "<Subject 1>: fully_preserved - identity.", "Live action. [Shot 1] The person walks.", "Room tone.", "N/A")))


class CanvasTests(unittest.TestCase):
    def test_presets_match_official_h3_buckets(self):
        self.assertEqual(geometry.canvas("360p", "16:9", 1, 1), (640, 352))
        self.assertEqual(geometry.canvas("480p", "16:9", 1, 1), (832, 480))
        self.assertEqual(geometry.canvas("480p", "9:16", 1, 1), (480, 832))
        self.assertEqual(geometry.canvas("480p", "1:1", 1, 1), (640, 640))
        self.assertEqual(geometry.canvas("768p (native)", "16:9", 1, 1), (1344, 768))
        self.assertEqual(geometry.canvas("768p (native)", "9:16", 1, 1), (768, 1344))
        self.assertEqual(geometry.canvas("768p (native)", "1:1", 1, 1), (992, 992))
        for res in geometry.RESOLUTIONS:
            for aspect in geometry.ASPECTS:
                w, h = geometry.canvas(res, aspect, 1920, 1080, "5:4")
                self.assertEqual(w % 32, 0)
                self.assertEqual(h % 32, 0)

    def test_every_preset_keeps_a_constant_pixel_budget(self):
        for res, budget in geometry.AREA_BUDGETS.items():
            for ratio in ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9",
                          "5:4", "10:1", "1:10"):
                with self.subTest(preset=res, ratio=ratio):
                    w, h = geometry.canvas(res, "custom", 1, 1, ratio)
                    self.assertAlmostEqual(w * h / budget, 1.0, delta=0.12,
                                           msg=f"{res} {ratio} -> {w}x{h}")
                    num, den = (float(x) for x in ratio.split(":"))
                    # the 32px grid limits ratio fidelity when the short side is small
                    self.assertAlmostEqual(w / h, num / den, delta=0.15 * num / den)

    def test_same_as_reference_and_custom(self):
        self.assertEqual(geometry.canvas("480p", "same as reference", 800, 800), (640, 640))
        self.assertEqual(geometry.canvas("480p", "custom", 1, 1, "5:4"), (704, 576))
        for ratio in ("", "16/9", "0:1", "nan:1", "-1:2", "1000:1"):
            with self.assertRaises(ValueError):
                geometry.canvas("480p", "custom", 1, 1, ratio)

    def test_timeline_uses_official_alignment_without_retiming(self):
        for fps in (24, 30, 60, 23.976):
            n, indices = geometry.video_timeline(1000, fps, 2, 10)
            self.assertEqual(n, 243)
            for j, index in enumerate(indices):
                self.assertLessEqual(abs(index / fps - (2 + j / 24)), 1 / fps + 1e-7)
        n, idx = geometry.video_timeline(900, 30, 2, 5)
        self.assertEqual(n, 124)
        self.assertEqual(idx[0], 60)
        self.assertEqual(idx[-1], 213)

    def test_timeline_repeats_only_missing_tail_frames(self):
        n, idx = geometry.video_timeline(240, 24, 0, 10)
        self.assertEqual(n, 243)
        self.assertEqual(idx[:240], list(range(240)))
        self.assertEqual(idx[240:], [239] * 3)
        n, idx = geometry.video_timeline(120, 24, 0, 10)
        self.assertEqual(n, 124)  # source shorter than requested: no long freeze
        self.assertEqual(idx[:120], list(range(120)))
        self.assertEqual(idx[120:], [119] * 4)
        n, idx = geometry.video_timeline(288, 24, 2, 10)
        self.assertEqual(idx[:240], list(range(48, 288)))
        self.assertEqual(idx[240:], [287] * 3)

    def test_official_duration_rounding_at_boundaries(self):
        for seconds, expected in ((5, 124), (10, 243), (15, 362),
                                  (22 / 24, 22), (22.4 / 24, 22), (22.6 / 24, 39)):
            self.assertEqual(geometry.generation_frames(seconds), expected)
            self.assertEqual(geometry.video_timeline(1000, 24, 0, seconds)[0], expected)
        self.assertEqual(geometry.video_timeline(10000, 24, 0, 60)[0], 362)

    def test_timeline_rejects_invalid_segments(self):
        for count, fps, start, seconds in ((3, 24, 0, 10), (20, float("nan"), 0, 10),
                                           (240, 24, 10, 5), (240, 24, -1, 5),
                                           (240, 24, 0, 0)):
            with self.assertRaises(ValueError):
                geometry.video_timeline(count, fps, start, seconds)


class PromptTests(unittest.TestCase):
    def test_first_frame_applies_to_override_and_is_idempotent(self):
        result = formats.apply_policy(PROMPT, first_frame=True)
        self.assertIn("[keyframe completion + reference generation]", result)
        self.assertIn("<Picture 1> is the first frame", result)
        self.assertIn("[Shot 1] The shot begins from <Picture 1>.", result)
        self.assertEqual(result, formats.apply_policy(result, first_frame=True))

    def test_no_first_frame_by_default(self):
        self.assertNotIn("first frame", formats.apply_policy(PROMPT))

    def test_silent_and_invalid_audio_refs(self):
        result = formats.apply_policy(PROMPT, silent=True)
        self.assertIn("overall_soundscape:\nN/A", result)
        self.assertIn("non_diegetic_music:\nN/A", result)
        with self.assertRaises(ValueError):
            formats.apply_policy(PROMPT.replace("Room tone.", "<Audio 1>"), silent=True)
        # heading-less text is passed through instead of failing the run
        self.assertEqual(formats.apply_policy("truncated result"), "truncated result")
        with self.assertRaises(ValueError):
            formats.apply_policy("truncated result", strict=True)


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)
    def __getitem__(self, key):
        return FakeTensor((*self.shape[:-1], 0))


class VideoOnlyTests(unittest.TestCase):
    def test_no_audio_tokens_and_no_payload_mutation(self):
        torch = types.ModuleType("torch")
        torch.zeros_like = lambda x: FakeTensor(x.shape)
        old = sys.modules.get("torch")
        sys.modules["torch"] = torch
        try:
            video, audio = FakeTensor((1, 24, 2, 4, 4)), FakeTensor((1, 32, 2, 40))
            payload = {"layout": "original", "refs": []}
            options = {"key": "original"}
            def executor(x, t, context, **kw):
                self.assertEqual(x[1].shape[-1], 0)
                self.assertNotIn("layout", kw["minimax_payload"])
                kw["transformer_options"]["key"] = "changed"
                return [video, x[1]]
            out = video_only._video_only_forward(executor, [video, audio], 1, None,
                    transformer_options=options, minimax_payload=payload)
            self.assertEqual(out[1].shape, audio.shape)
            self.assertEqual(payload["layout"], "original")
            self.assertEqual(options["key"], "original")
            with self.assertRaises(ValueError):
                video_only._video_only_forward(executor, [video, audio], 1, None,
                    minimax_payload={"refs": [{"kind": "audio", "audio_latent": audio}]})
        finally:
            if old is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old


if __name__ == "__main__":
    unittest.main()
