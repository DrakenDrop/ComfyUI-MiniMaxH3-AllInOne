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
    def test_presets_portrait_and_landscape(self):
        self.assertEqual(geometry.canvas("360p", "16:9", 1, 1), (640, 352))
        self.assertEqual(geometry.canvas("480p", "9:16", 1, 1), (480, 864))
        self.assertEqual(geometry.canvas("768p (native)", "16:9", 1, 1), (1344, 768))
        for res in geometry.RESOLUTIONS:
            for aspect in geometry.ASPECTS:
                w, h = geometry.canvas(res, aspect, 1920, 1080, "5:4")
                self.assertEqual(w % 32, 0)
                self.assertEqual(h % 32, 0)

    def test_same_as_reference_and_custom(self):
        self.assertEqual(geometry.canvas("480p", "same as reference", 800, 800), (480, 480))
        self.assertEqual(geometry.canvas("480p", "custom", 1, 1, "5:4"), (608, 480))
        for ratio in ("", "16/9", "0:1", "nan:1", "-1:2", "1000:1"):
            with self.assertRaises(ValueError):
                geometry.canvas("480p", "custom", 1, 1, ratio)

    def test_timeline_preserves_time_and_never_pads(self):
        for fps, count in ((24, 120), (30, 150), (60, 300), (23.976, 240)):
            n, indices = geometry.video_timeline(count, fps)
            self.assertEqual(n % 17, 5)
            self.assertLessEqual(n / 24, count / fps)
            self.assertEqual(indices[0], 0)
            for j, index in enumerate(indices):
                self.assertLessEqual(abs(index / fps - j / 24), 1 / fps + 1e-7)
        n, idx = geometry.video_timeline(900, 30, 2, 5)
        self.assertEqual(n, 107)
        self.assertEqual(idx[0], 60)
        self.assertLess(idx[-1], 210)
        self.assertEqual(geometry.generation_frames(5), 124)
        with self.assertRaises(ValueError):
            geometry.video_timeline(3, 24)
        with self.assertRaises(ValueError):
            geometry.video_timeline(20, float("nan"))


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
        with self.assertRaises(ValueError):
            formats.apply_policy("truncated result")


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
