"""CPU tests for direct character-reference routing and unchecked H3 prompting."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes_character_swap.py"

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "h3_prompter" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

geometry, perf, fmt = load("canvas.py"), load("performance.py"), load("prompt_format.py")

class CharacterSwapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("vision.gguf", "mmproj.gguf"):
            (self.root / name).write_bytes(b"weights")
        self.source = torch.arange(300 * 2 * 3 * 3, dtype=torch.float32).reshape(300, 2, 3, 3)
        self.reference = torch.ones(2, 4, 3, 3)
        self.reference[1] = 0
        self.request = Mock(return_value=(json.dumps({"minimax_prompt": "Swap without required tags or sections."}), None, None))
        self.resolve = Mock(return_value=(str(self.root / "vision.gguf"), str(self.root / "mmproj.gguf")))
        self.server = types.SimpleNamespace(ensure=Mock(return_value="http://local"), stop=Mock())
        self.lc = types.SimpleNamespace(log=Mock(), ensure_server=Mock())
        self.conditioning = Mock(return_value=("positive", "latent"))
        self.control = Mock(return_value=("controlled",))
        self.wrapper = Mock(return_value="strict")
        self.unload = Mock()
        comfy = types.ModuleType("comfy")
        comfy.samplers = types.SimpleNamespace(SCHEDULER_NAMES=["simple"])
        comfy.model_management = types.SimpleNamespace(unload_all_models=self.unload)
        h3 = types.SimpleNamespace(MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=self.conditioning),
                                  MiniMaxH3FunControlNetApply=types.SimpleNamespace(execute=self.control))
        modules = {"comfy": comfy, "comfy.samplers": comfy.samplers,
                   "comfy.model_management": comfy.model_management,
                   "comfy_extras": types.SimpleNamespace(nodes_minimax_h3=h3)}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3CharacterSwap")
        system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in n.targets))
        ns = dict(_Pipeline=object, CATEGORY="test", NONE="(none)", os=os,
            SYSTEM_PROMPT=system, FIELDS=fmt.FIELDS, geometry=geometry, perf=perf,
            lc=self.lc, local_models=types.SimpleNamespace(resolve=self.resolve), managed_server=self.server,
            prompt_field_unchecked=fmt.prompt_field_unchecked, CFG={"server_url": "http://local"},
            prompter=types.SimpleNamespace(THINKING=["off", "low", "medium", "xhigh"],
                MiniMaxH3R2VPrompter=types.SimpleNamespace(_run=self.request)),
            media=types.SimpleNamespace(sample_indices=lambda n, m: list(range(0, n, max(1, n // m)))[:m],
                image_batch_to_pil=lambda t: [t], pil_to_data_url=lambda im, size: str((size, perf.image_key(im)))),
            args=lambda out: out, patch_video_only=self.wrapper,
            v2v=types.SimpleNamespace(resize_frames=lambda im, *args: im),
            common_inputs=lambda: {}, extra_inputs=lambda: {}, choices=lambda *args, **kw: ["fun"])
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), ns)
        self.node = ns["MiniMaxH3CharacterSwap"]()
        self.node._base_models = Mock(return_value=("model", "clip", "vae"))
        self.node._load = Mock(return_value="patch")
        self.node._sample = Mock(return_value=(self.source, "sampled"))

    def enhance(self, **changes):
        kw = dict(source=self.source[:243], reference=self.reference[:1], instruction="swap character",
                  frames=243, llm_model="vision", mmproj="auto", seed=1,
                  minimax_prompt_style="official", minimax_thinking="medium", keep_models_loaded=True)
        kw.update(changes)
        return self.node._enhance_character(**kw)

    def generate(self, **changes):
        kw = dict(ref_video=self.source, ref_image=self.reference, fun_controlnet="fun", start_seconds=2,
                  max_seconds=10, instruction="swap", resolution="480p", aspect_ratio="same as reference",
                  custom_aspect="16:9", seed=1, steps=2, sampler_name="res_multistep", scheduler="simple",
                  prompt_override="Use the target character.", h3_sampling_mode="native AV (discard audio)")
        kw.update(changes)
        return self.node.generate(**kw)

    def test_direct_reference_and_same_video_for_funcontrolnet(self):
        result = self.generate()
        args = self.conditioning.call_args.kwargs
        self.assertTrue(torch.equal(args["ref_images"]["ref_image_0"], self.reference[:1]))
        video = args["ref_videos"]["ref_video_0"]
        self.assertTrue(torch.equal(video, self.source[48:291]))
        self.assertIs(self.control.call_args.kwargs["control_video"], video)
        self.assertNotIn("mask", self.control.call_args.kwargs)
        self.assertEqual(args["length"], 243)
        self.assertEqual(self.node._sample.call_args.args[0], "controlled")
        self.assertFalse(self.node._sample.call_args.kwargs["silent"])
        self.assertEqual(len(result[0]), 243)
        self.assertEqual(result[1], "Use the target character.")
        self.wrapper.assert_not_called()
        self.assertIsNone(args["audio_vae"])
        self.assertIsNone(args["ref_audios"])
        self.assertIsNone(args["ref_video_audios"])
        self.request.assert_not_called()
        self.resolve.assert_not_called()

    def test_strict_video_only_keeps_references_and_control(self):
        self.generate(h3_sampling_mode="strict video-only (experimental)")
        self.wrapper.assert_called_once_with("controlled")
        self.assertTrue(self.node._sample.call_args.kwargs["silent"])
        self.assertEqual(self.node._sample.call_args.args[0], "strict")

    def test_schema_has_official_input_names_and_no_image_editor(self):
        schema = self.node.INPUT_TYPES()
        self.assertEqual(schema["required"]["ref_video"][0], "IMAGE")
        self.assertEqual(schema["required"]["ref_image"][0], "IMAGE")
        self.assertIn("fun_controlnet", schema["required"])
        self.assertFalse(any(k.startswith("qwen_") for group in schema.values() for k in group))
        self.assertNotIn("motion_control", schema["optional"])
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE", "STRING"))

    def test_unchecked_prompt_and_correct_asset_roles(self):
        prompt = self.enhance()
        self.assertEqual(prompt, "Swap without required tags or sections.")
        self.assertEqual(self.request.call_count, 1)
        args = self.request.call_args.args
        self.assertEqual(args[4], "medium")
        self.assertIn("directly connected to H3", args[2]["content"])
        parts = args[3]["content"]
        self.assertEqual(sum(p["type"] == "image_url" for p in parts), 9)
        self.assertIn("<Picture 1>", parts[-2]["text"])
        self.assertFalse(any("<image1>" in p.get("text", "") for p in parts))
        self.unload.assert_not_called()
        self.server.stop.assert_called_once()

    def test_prompt_cache_invalidates_reference_and_thinking(self):
        self.enhance()
        self.enhance()
        self.assertEqual(self.request.call_count, 1)
        self.reference[0, 0, 0, 0] = 0.2
        self.enhance()
        self.assertEqual(self.request.call_count, 2)
        self.enhance(minimax_thinking="off")
        self.assertEqual(self.request.call_count, 3)

    def test_override_needs_no_enhancer_and_is_verbatim(self):
        raw = "  Raw prompt <Picture 9> as first frame.  "
        self.assertEqual(self.enhance(prompt_override=raw), raw)
        self.resolve.assert_not_called()
        self.request.assert_not_called()

    def test_external_server_is_not_cached_or_stopped(self):
        self.resolve.return_value = (None, None)
        self.enhance(keep_models_loaded=False)
        self.enhance(keep_models_loaded=False)
        self.assertEqual(self.request.call_count, 2)
        self.assertEqual(self.unload.call_count, 2)
        self.server.stop.assert_not_called()

    def test_invalid_inputs_fail_before_generation(self):
        with self.assertRaisesRegex(ValueError, "ref_video"):
            self.generate(ref_video=self.source[:0])
        with self.assertRaisesRegex(ValueError, "ref_image"):
            self.generate(ref_image=self.reference[:0])
        self.node._sample.assert_not_called()

if __name__ == "__main__":
    unittest.main()
