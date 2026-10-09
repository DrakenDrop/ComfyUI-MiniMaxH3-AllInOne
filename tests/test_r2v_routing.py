"""CPU reference-routing tests; model calls are mocked, not GPU quality checks."""
import ast
import importlib.util
from itertools import product
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes_compact.py"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "h3_prompter" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry, fmt = load("canvas.py"), load("prompt_format.py")
PROMPT = "\n\n".join(f"{field}:\n{body}" for field, body in zip(fmt.FIELDS, (
    "<Subject 1> is the person from <Picture 1>.", "[reference generation] A person walks.",
    "<Subject 1>: fully_preserved - identity.", "Live action. [Shot 1] The person walks.",
    "Room tone.", "N/A")))


class R2VRoutingTests(unittest.TestCase):
    def setUp(self):
        self.images = [torch.full((2, 4, 6, 3), float(i)) for i in range(3)]
        self.audios = [dict(waveform=torch.arange(200).reshape(1, 1, 200) + i * 1000,
                            sample_rate=24, metadata=f"audio {i}") for i in range(2)]
        self.generated_audio = {"waveform": torch.ones(1, 1, 124), "sample_rate": 24}
        self.enhancer = Mock(return_value=(PROMPT,))
        self.conditioning = Mock(return_value=("positive", "latent"))
        self.guide = Mock(return_value=("guided",))
        self.decode_audio = Mock(return_value=(self.generated_audio,))
        self.create_video = Mock(return_value=("video",))
        self.resize = Mock(side_effect=lambda image, *args: image)
        self.unload, self.stop = Mock(), Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management = types.SimpleNamespace(unload_all_models=self.unload)
        h3 = types.SimpleNamespace(
            MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=self.conditioning),
            MiniMaxH3AddGuide=types.SimpleNamespace(execute=self.guide))
        modules = {
            "comfy": comfy, "comfy.model_management": comfy.model_management,
            "nodes": types.SimpleNamespace(VAELoader=lambda: types.SimpleNamespace(
                load_vae=lambda name: ("audio-vae",))),
            "comfy_extras": types.SimpleNamespace(nodes_minimax_h3=h3,
                nodes_audio=types.SimpleNamespace(VAEDecodeAudio=types.SimpleNamespace(execute=self.decode_audio)),
                nodes_video=types.SimpleNamespace(CreateVideo=types.SimpleNamespace(execute=self.create_video))),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Execute the actual pipeline classes without importing ComfyUI/model dependencies.
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name in ("_Pipeline", "MiniMaxH3R2VGenerate")]
        ns = dict(CATEGORY="test", NONE="(none)", CFG={}, geometry=geometry,
            apply_policy=fmt.apply_policy, args=lambda output: output,
            common_inputs=lambda: {}, extra_inputs=lambda: {}, choices=lambda *a, **kw: [],
            prompter=types.SimpleNamespace(THINKING=["off", "low", "medium", "xhigh"],
                MiniMaxH3R2VPrompter=lambda: types.SimpleNamespace(generate=self.enhancer),
                prompts=types.SimpleNamespace(PROMPT_FLAVORS=("standard", "vivid", "spicy"))),
            local_models=types.SimpleNamespace(SERVER_DEFAULT="server default"),
            managed_server=types.SimpleNamespace(stop=self.stop),
            v2v=types.SimpleNamespace(resize_frames=self.resize))
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec"), ns)
        self.node = ns["MiniMaxH3R2VGenerate"]()
        self.node._base_models = Mock(return_value=("model", "clip", "video-vae"))
        self.node._sample = Mock(return_value=("frames", "sampled"))

    def generate(self, **changes):
        kw = dict(ref_image=self.images[0], ref_audio=self.audios[0], h3_audio_vae="audio",
            duration_seconds=5, audio_mode="generate from reference", instruction="Make a conversation.",
            resolution="480p", aspect_ratio="same as reference", custom_aspect="16:9",
            seed=1, steps=2, sampler_name="res_multistep", scheduler="beta",
            llm_model="vision", mmproj="auto")
        kw.update(changes)
        return self.node.generate(**kw)

    def assert_references(self, image_indices, two_audio=False):
        enhanced = self.enhancer.call_args.kwargs
        conditioned = self.conditioning.call_args.kwargs
        self.assertEqual([k for k in enhanced if k.startswith("image_")],
                         [f"image_{i}" for i in range(1, len(image_indices) + 1)])
        self.assertEqual(list(conditioned["ref_images"]),
                         [f"ref_image_{i}" for i in range(len(image_indices))])
        for i, source in enumerate(image_indices):
            self.assertTrue(torch.equal(enhanced[f"image_{i + 1}"], self.images[source][:1]))
            self.assertIs(enhanced[f"image_{i + 1}"], conditioned["ref_images"][f"ref_image_{i}"])
        count = 2 if two_audio else 1
        self.assertEqual([k for k in enhanced if k.startswith("audio_")],
                         [f"audio_{i}" for i in range(1, count + 1)])
        self.assertEqual(list(conditioned["ref_audios"]), [f"ref_audio_{i}" for i in range(count)])
        for i in range(count):
            self.assertIs(enhanced[f"audio_{i + 1}"], self.audios[i])
            self.assertIs(conditioned["ref_audios"][f"ref_audio_{i}"], self.audios[i])

        # Check labels emitted by the real enhancer asset collector as well as its inputs.
        tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3R2VPrompter")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name in ("_collect_assets", "_audio_line")]
        ns = dict(MAX_PICTURES=3, MAX_VIDEOS=1, MAX_AUDIOS=2,
            media=types.SimpleNamespace(image_batch_to_pil=lambda image: [types.SimpleNamespace(size=(6, 4))],
                pil_to_data_url=lambda *args: "data:image", audio_duration=lambda audio: 1),
            lc=types.SimpleNamespace(log=Mock()))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "nodes.py"), "exec"), ns)
        _, pictures, _, audios, _ = ns["MiniMaxH3R2VPrompter"]()._collect_assets(dict(enhanced), 124, 1, 512, 512)
        self.assertEqual([line.split(":")[0] for line in pictures],
                         [f"<Picture {i}>" for i in range(1, len(image_indices) + 1)])
        self.assertEqual([line.split(":")[0] for line in audios],
                         [f"<Audio {i}>" for i in range(1, count + 1)])

    def test_schema_preserves_required_inputs_and_adds_optional_sockets(self):
        schema = self.node.INPUT_TYPES()
        for name, kind in (("ref_image", "IMAGE"), ("ref_audio", "AUDIO")):
            self.assertEqual(schema["required"][name][0], kind)
        for name, kind in (("ref_image_2", "IMAGE"), ("ref_image_3", "IMAGE"), ("ref_audio_2", "AUDIO")):
            self.assertNotIn(name, schema["required"])
            self.assertEqual(schema["optional"][name][0], kind)

    def test_old_workflow_omitting_new_inputs_keeps_single_references(self):
        result = self.generate()
        self.assert_references([0])
        self.guide.assert_not_called()
        self.assertIs(result[2], self.generated_audio)
        self.assertEqual(result[4:], (*geometry.canvas("480p", "same as reference", 6, 4, "16:9"), 124))

    def test_all_optional_combinations_are_numbered_consecutively(self):
        for second, third, audio in product((False, True), repeat=3):
            with self.subTest(second=second, third=third, audio=audio):
                self.generate(ref_image_2=self.images[1] if second else None,
                              ref_image_3=self.images[2] if third else None,
                              ref_audio_2=self.audios[1] if audio else None)
                self.assert_references([0] + ([1] if second else []) + ([2] if third else []), audio)

    def test_generate_audio_preserves_explicit_subject_mapping(self):
        instruction = "Use <Audio 2> for the person in <Picture 1> and <Audio 1> for the person in <Picture 3>."
        result = self.generate(instruction=instruction, ref_image_2=self.images[1],
                               ref_image_3=self.images[2], ref_audio_2=self.audios[1])
        actual = self.enhancer.call_args.kwargs["instruction"]
        self.assertTrue(actual.startswith(instruction))
        self.assertIn("<Audio 1> and <Audio 2>", actual)
        self.assertIn("reference numbers do not imply pairings", actual)
        self.decode_audio.assert_called_once_with(samples="sampled", vae="audio-vae")
        self.assertIs(result[2], self.generated_audio)
        self.assertIs(self.create_video.call_args.kwargs["audio"], result[2])

    def test_first_frame_guides_only_first_image_with_all_references(self):
        result = self.generate(ref_image_2=self.images[1], ref_image_3=self.images[2],
                               ref_audio_2=self.audios[1], ref_image_1_as_first_frame=True)
        self.assert_references([0, 1, 2], True)
        self.guide.assert_called_once()
        self.assertEqual(self.guide.call_args.kwargs["frame_idx"], 0)
        self.assertTrue(torch.equal(self.guide.call_args.kwargs["image"], self.images[0][:1]))
        self.assertEqual(self.node._sample.call_args.args[1], "guided")
        self.assertEqual(self.enhancer.call_args.kwargs["frame_anchor"], "reference 1 = first frame")
        self.assertIn("begins from <Picture 1>", result[3])

    def test_reuse_copies_only_first_audio_without_mutating_sources(self):
        for length in (20, 200):
            with self.subTest(length=length):
                self.audios[0]["waveform"] = torch.arange(length).reshape(1, 1, length)
                original = self.audios[0]["waveform"].clone()
                second = self.audios[1]["waveform"].clone()
                result = self.generate(audio_mode="reuse reference exactly", ref_audio_2=self.audios[1])
                self.assert_references([0], True)
                self.decode_audio.assert_not_called()
                self.assertTrue(torch.equal(result[2]["waveform"], original[..., :124]))
                self.assertEqual(result[2]["sample_rate"], 24)
                self.assertEqual(result[2]["metadata"], "audio 0")
                self.assertTrue(torch.equal(self.audios[0]["waveform"], original))
                self.assertTrue(torch.equal(self.audios[1]["waveform"], second))
                self.assertIs(self.create_video.call_args.kwargs["audio"], result[2])
                self.assertIn("copies only <Audio 1>", self.enhancer.call_args.kwargs["instruction"])

    def test_override_skips_enhancer_but_conditions_all_references_in_both_styles(self):
        for style, prompt in (("official", PROMPT), ("simple", "A person walks.")):
            with self.subTest(style=style):
                result = self.generate(ref_image_2=self.images[1], ref_image_3=self.images[2],
                    ref_audio_2=self.audios[1], prompt_override=prompt, minimax_prompt_style=style,
                    ref_image_1_as_first_frame=True)
                self.enhancer.assert_not_called()
                self.unload.assert_not_called()
                self.stop.assert_not_called()
                self.assertEqual(len(self.conditioning.call_args.kwargs["ref_images"]), 3)
                self.assertEqual(len(self.conditioning.call_args.kwargs["ref_audios"]), 2)
                self.assertIn("begins from <Picture 1>", result[3])

    def test_shared_prompt_single_reference_call_remains_compatible(self):
        self.node._prompt(ref_image=self.images[0][:1], audio=self.audios[0], frames=124,
                          instruction="Walk", llm_model="vision", mmproj="auto", seed=1)
        kw = self.enhancer.call_args.kwargs
        self.assertTrue(torch.equal(kw["image_1"], self.images[0][:1]))
        self.assertIs(kw["audio_1"], self.audios[0])
        self.assertNotIn("image_2", kw)
        self.assertNotIn("audio_2", kw)

    def test_requested_duration_is_conveyed_without_fixed_action_templates(self):
        for style in ("official", "simple"):
            plans = []
            for seconds, frames in ((5, 124), (10, 243), (15, 362)):
                with self.subTest(style=style, seconds=seconds):
                    self.generate(duration_seconds=seconds, minimax_prompt_style=style,
                        ref_image_2=self.images[1], ref_image_3=self.images[2], ref_audio_2=self.audios[1])
                    kw = self.enhancer.call_args.kwargs
                    self.assertEqual(kw["duration_seconds"], seconds)
                    self.assertEqual(kw["frame_count"], frames)
                    self.assertEqual(kw["length"], "standard")
                    self.assertTrue(kw["r2v_duration_context"])
                    self.assertIn(f"This video is for {seconds} seconds.", kw["extra_rules"])
                    self.assertIn(f"appropriate for a {seconds}-second video", kw["extra_rules"])
                    self.assertIn("Choose the shots, actions, pacing and narrative", kw["extra_rules"])
                    self.assertIn("subject/audio mapping", kw["extra_rules"])
                    self.assertNotIn("planning_duration_s", kw)
                    self.assertEqual(self.conditioning.call_args.kwargs["length"], frames)
                    self.assert_references([0, 1, 2], True)
                    plans.append(kw["extra_rules"].replace(str(seconds), "DURATION"))
            # Only the numeric duration changes; no duration-specific action/shot template.
            self.assertEqual(len(set(plans)), 1)

    def test_duration_context_does_not_add_output_rejection_or_rewriting(self):
        self.enhancer.return_value = (PROMPT.replace("The person walks.",
            "The person walks. [Shot 2] At 00:05.000, the camera cuts."),)
        result = self.generate(duration_seconds=5)
        self.assertEqual(result[3], self.enhancer.return_value[0])
        self.conditioning.assert_called_once()
        self.node._sample.assert_called_once()
        self.stop.assert_called_once()

    def test_duration_does_not_rewrite_or_validate_user_override(self):
        for style in ("official", "simple"):
            override = PROMPT if style == "official" else "Keep <Picture 1> and <Audio 1>."
            override += " [Shot 2] At 00:20.000, the camera cuts."
            result = self.generate(duration_seconds=5, prompt_override=override, minimax_prompt_style=style)
            self.assertEqual(result[3], override)
            self.enhancer.assert_not_called()

    def test_prompt_flavor_reaches_enhancer_in_both_styles(self):
        schema = self.node.INPUT_TYPES()["optional"]
        self.assertEqual(schema["minimax_prompt_flavor"][0], ["standard", "vivid", "spicy"])
        self.assertEqual(schema["minimax_prompt_flavor"][1]["default"], "standard")
        for style, flavor in (("official", "vivid"), ("official", "spicy"), ("simple", "spicy")):
            with self.subTest(style=style, flavor=flavor):
                self.generate(minimax_prompt_style=style, minimax_prompt_flavor=flavor)
                self.assertEqual(self.enhancer.call_args.kwargs["prompt_flavor"], flavor)
                self.assert_references([0])
        self.generate()
        self.assertEqual(self.enhancer.call_args.kwargs["prompt_flavor"], "standard")


if __name__ == "__main__":
    unittest.main()
