"""CPU regression tests for preprocessing reuse; no model/GPU speed claims."""
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


def load(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "h3_prompter" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


perf = load("performance.py")
fmt = load("prompt_format.py")
prompts = load("prompts.py")


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("model", "encoder", "vae", "lora", "llm.gguf", "mmproj.gguf"):
            (self.root / name).write_bytes(b"weights")
        self.source = torch.zeros(5, 2, 3, 3)
        self.reference = torch.ones(1, 2, 3, 3)
        self.qwen = Mock(side_effect=lambda *args: torch.full((1, 2, 3, 3), 0.25))
        self.loader = Mock(side_effect=lambda kind, name: name)
        loader = self.loader
        class Pipeline:
            def _load(self, kind, name):
                return loader(kind, name)
        self.unload = Mock()
        management = types.ModuleType("comfy.model_management")
        management.unload_all_models = self.unload
        comfy = types.ModuleType("comfy")
        comfy.model_management = management
        self.lc = types.SimpleNamespace(log=Mock(), ensure_server=Mock())
        self.server = types.SimpleNamespace(ensure=Mock(return_value="http://local"), stop=Mock())
        self.resolve = Mock(return_value=(str(self.root / "llm.gguf"), str(self.root / "mmproj.gguf")))
        response = json.dumps({"qwen_prompt": "Edit <image1> using <image2>.",
                               "minimax_prompt": "Edit <Video 1> using the outfit in <Picture 1>."})
        self.request = Mock(return_value=(response, None, None))
        media = types.SimpleNamespace(
            sample_indices=lambda count, maximum: list(range(count)),
            image_batch_to_pil=lambda image: [image],
            pil_to_data_url=lambda image, size: str((size, perf.image_key(image))))
        modules = {
            "comfy": comfy, "comfy.model_management": management,
            "folder_paths": types.SimpleNamespace(get_full_path_or_raise=lambda kind, name: str(self.root / name)),
            "nodes": types.SimpleNamespace(LoraLoaderModelOnly=lambda: types.SimpleNamespace(
                load_lora_model_only=Mock(return_value=("patched",)))),
            "speed_pkg.nodes_h3qwen": types.SimpleNamespace(
                MiniMaxH3QwenKeyframeEdit=types.SimpleNamespace(_qwen_edit=self.qwen)),
            "speed_pkg.h3_prompter": types.SimpleNamespace(media=media, llama_client=self.lc, prompts=prompts),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        tree = ast.parse((ROOT / "nodes_compact.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3V2VGenerate")
        namespace = dict(__name__="speed_pkg.nodes_compact", __package__="speed_pkg",
                         _Pipeline=Pipeline, CATEGORY="test", NONE="(none)", os=os, perf=perf,
                         CFG={"server_url": "http://local"}, local_models=types.SimpleNamespace(resolve=self.resolve),
                         managed_server=self.server, parse_edit_response_with_repair=fmt.parse_edit_response_with_repair,
                         prompter=types.SimpleNamespace(MiniMaxH3R2VPrompter=types.SimpleNamespace(_run=self.request)))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "nodes_compact.py", "exec"), namespace)
        self.node = namespace["MiniMaxH3V2VGenerate"]()

    def edit(self, **changes):
        kwargs = dict(source=self.source, reference=self.reference, prompt="edit", seed=1,
                      model_name="model", encoder_name="encoder", vae_name="vae",
                      lora_name="(none)", lora_strength=1.0, steps=6, resolution=1024, reuse=True)
        kwargs.update(changes)
        return self.node._cached_qwen_edit(**kwargs)

    def enhance(self, **changes):
        kwargs = dict(source=self.source, reference=self.reference, instruction="change clothes", frames=5,
                      llm_model="llm", mmproj="auto", seed=1, edit_mode="custom", minimax_prompt_style="simple",
                      unload_llm_after_prompt=True, reuse_preprocessing=True)
        kwargs.update(changes)
        return self.node._enhance_edit(**kwargs)

    def test_qwen_hit_skips_loaders_and_returns_isolated_cpu_copy(self):
        first = self.edit()
        first.fill_(0.9)
        second = self.edit(source=self.source.clone(), reference=self.reference.clone())
        self.assertTrue(torch.all(second == 0.25))
        second.fill_(0.1)
        self.assertTrue(torch.all(self.edit() == 0.25))
        self.assertEqual(self.qwen.call_count, 1)
        self.assertEqual(self.loader.call_count, 3)
        self.assertEqual(second.device.type, "cpu")

    def test_qwen_invalidates_pixels_parameters_and_weights(self):
        self.edit()
        self.reference[0, 1, 2, 1] = 0.99
        self.edit()
        self.assertEqual(self.qwen.call_count, 2)
        (self.root / "model").write_bytes(b"new model weights")
        self.edit()
        self.assertEqual(self.qwen.call_count, 3)
        for changes in ({"seed": 2}, {"prompt": "different"}, {"steps": 7}, {"resolution": 768},
                        {"lora_name": "lora"}, {"lora_name": "lora", "lora_strength": 0.5}):
            before = self.qwen.call_count
            self.edit(**changes)
            self.assertEqual(self.qwen.call_count, before + 1)

    def test_disable_clears_qwen_cache(self):
        self.edit()
        self.edit(reuse=False)
        self.edit()
        self.assertEqual(self.qwen.call_count, 3)

    def test_enhancer_hit_skips_unload_server_start_and_inference(self):
        original = self.enhance()
        self.assertEqual(self.enhance(steps=2, scheduler="simple", control_strength=0.8), original)
        self.assertEqual(self.request.call_count, 1)
        self.unload.assert_called_once()
        self.server.ensure.assert_called_once()
        self.assertEqual(self.server.stop.call_count, 2)

    def test_enhancer_invalidates_actual_request_and_model(self):
        self.enhance()
        for changes in ({"seed": 2}, {"instruction": "change person"}, {"context_size": 65536},
                        {"additional_system_prompt": "retain background"}):
            before = self.request.call_count
            self.enhance(**changes)
            self.assertEqual(self.request.call_count, before + 1)
        self.enhance()
        before = self.request.call_count
        self.source[4, 0, 0, 0] = 0.4
        self.enhance()
        self.assertEqual(self.request.call_count, before + 1)
        (self.root / "mmproj.gguf").write_bytes(b"new projector")
        self.enhance()
        self.assertEqual(self.request.call_count, before + 2)

    def test_external_and_disabled_enhancer_always_query(self):
        self.enhance()
        self.enhance(reuse_preprocessing=False)
        self.enhance()
        self.assertEqual(self.request.call_count, 3)
        self.resolve.return_value = (None, None)
        self.enhance()
        self.enhance()
        self.assertEqual(self.request.call_count, 5)

    def test_frame_view_matches_indexed_selection_without_copy(self):
        frames = torch.arange(60).reshape(5, 2, 2, 3)
        result = perf.take_frames(frames, [1, 2, 3])
        self.assertTrue(torch.equal(result, frames[torch.tensor([1, 2, 3])]))
        self.assertEqual(result.untyped_storage().data_ptr(), frames.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(perf.take_frames(frames, [1, 3, 3]), frames[torch.tensor([1, 3, 3])]))


    def test_qwen_without_reference_uses_one_image_and_separate_cache(self):
        self.edit()
        self.edit(reference=None)
        self.assertEqual(len(self.qwen.call_args.args[4]), 1)
        self.assertEqual(self.qwen.call_count, 2)
        self.edit(reference=None)
        self.assertEqual(self.qwen.call_count, 2)
        self.edit()
        self.assertEqual(self.qwen.call_count, 3)
        self.assertEqual(len(self.qwen.call_args.args[4]), 2)

    def test_enhancer_without_reference_both_styles_and_validation(self):
        qwen = "Edit <image1>: make the dress red."
        mini = "Edit <Video 1> using the edited appearance in <Picture 1>."
        for style in ("simple", "official"):
            with self.subTest(style=style):
                value = mini if style == "simple" else dict(zip(fmt.FIELDS, (
                    "<Video 1> is the source. <Picture 1> is the edited appearance.",
                    "[video editing] Make the dress red.", "Preserve source motion.",
                    mini, "N/A", "N/A")))
                response = json.dumps({"qwen_prompt": qwen, "minimax_prompt": value})
                self.request.return_value = (response, None, None)
                actual, _ = self.enhance(reference=None, instruction="make her dress red",
                                         minimax_prompt_style=style)
                self.assertEqual(actual, qwen)
                parts = self.request.call_args.args[3]["content"]
                image_labels = [p.get("text", "") for p in parts if p.get("type") == "text"]
                self.assertFalse(any("<image2>" in text for text in image_labels))
                self.assertEqual(sum(p.get("type") == "image_url" for p in parts), len(self.source) + 1)
                with self.assertRaisesRegex(ValueError, "not connected"):
                    fmt.parse_edit_response(response.replace(qwen, "Edit <image1> using <image2>."),
                                            style=style, has_reference=False)
                anchored = json.loads(response)
                if style == "simple":
                    anchored["minimax_prompt"] += " The shot begins from <Picture 1>."
                else:
                    anchored["minimax_prompt"]["detailed_description"] += " The shot begins from <Picture 1>."
                repaired, _ = fmt.parse_edit_response_with_repair(
                    json.dumps(anchored), style, lambda reason: response, has_reference=False)
                self.assertEqual(repaired, qwen)

    def test_split_gguf_stamp_tracks_all_shards(self):
        first = self.root / "model-00001-of-00002.gguf"
        second = self.root / "model-00002-of-00002.gguf"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        before = perf.file_stamp(str(first))
        second.write_bytes(b"changed")
        self.assertNotEqual(perf.file_stamp(str(first)), before)


if __name__ == "__main__":
    unittest.main()
