"""CPU routing tests; these do not assess generated image quality."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "nodes_compact.py"

spec = importlib.util.spec_from_file_location("perf", SOURCE.parent / "h3_prompter/performance.py")
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)

class Tensor:
    def __init__(self, label, count=22, width=768, height=1344):
        self.label, self.shape = label, (count, height, width, 3)
        self.ndim, self.device = 4, "cpu"
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, key):
        if isinstance(key, slice):
            return Tensor(self.label, len(range(self.shape[0])[key]), self.shape[2], self.shape[1])
        return self
    def __repr__(self):
        return self.label

class V2VRoutingTests(unittest.TestCase):
    def run_mode(self, mode, has_reference=True, motion_control="rgb source (legacy)",
                 use_qwen=True, use_control=True):
        source, target = Tensor("source"), Tensor("target", 1)
        edited = Tensor("qwen", 1, 1024, 1024)
        conditioning = Mock(return_value=("positive", "latent"))
        control = Mock(return_value=("controlled-model",))
        qwen = Mock(return_value=edited)
        wrapper = Mock(return_value="strict-model")
        pose = types.SimpleNamespace(require_dwpose=Mock(), extract_pose=Mock(return_value=Tensor("pose")))
        h3 = types.SimpleNamespace(
            MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=conditioning),
            MiniMaxH3FunControlNetApply=types.SimpleNamespace(execute=control))
        comfy = types.ModuleType("comfy")
        comfy.samplers = types.SimpleNamespace(SCHEDULER_NAMES=["simple"])
        modules = {
            "comfy": comfy, "comfy.samplers": comfy.samplers,
            "torch": types.SimpleNamespace(tensor=lambda value, **kw: value, long=object()),
            "nodes": types.SimpleNamespace(),
            "folder_paths": types.SimpleNamespace(),
            "comfy_extras": types.SimpleNamespace(nodes_minimax_h3=h3),
            "routing_pkg.nodes_h3qwen": types.SimpleNamespace(
                MiniMaxH3QwenKeyframeEdit=types.SimpleNamespace(_qwen_edit=qwen)),
            "routing_pkg.h3_prompter": types.SimpleNamespace(llama_client=types.SimpleNamespace(log=lambda msg: None)),
        }
        class Pipeline:
            def _load(self, kind, name):
                return kind + ":" + name
            def _base_models(self, **kw):
                return "h3-model", "h3-clip", "h3-vae"
            def _sample(self, *args, **kw):
                self.sample_args, self.sample_kw = args, kw
                return source, "sampled"
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3V2VGenerate")
        namespace = dict(__name__="routing_pkg.nodes_compact", __package__="routing_pkg",
                         _Pipeline=Pipeline, CATEGORY="test", NONE="(none)", perf=perf, pose_control=pose,
                         prompter=types.SimpleNamespace(THINKING=["off", "low", "medium", "xhigh"]),
                         common_inputs=lambda: {}, extra_inputs=lambda: {}, choices=lambda *a, **kw: [],
                         args=lambda output: output, patch_video_only=wrapper,
                         geometry=types.SimpleNamespace(
                             video_timeline=lambda *a: (22, list(range(22))),
                             canvas=lambda *a: (768, 1344)),
                         v2v=types.SimpleNamespace(resize_frames=lambda image, *a: image))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
        node = namespace["MiniMaxH3V2VGenerate"]()
        node._enhance_edit = Mock(return_value=("qwen prompt", "H3 prompt"))
        with patch.dict(sys.modules, modules):
            schema = node.INPUT_TYPES()
            self.assertNotIn("ref_image", schema["required"])
            self.assertIn("ref_image", schema["optional"])
            self.assertIn("source_video", schema["required"])
            self.assertEqual(schema["optional"]["use_qwen_edit"][1]["default"], True)
            self.assertEqual(schema["optional"]["apply_fun_controlnet"][1]["default"], True)
            result = node.generate(
                source, "fun", "qwen-model", "qwen-clip", "qwen-vae", "custom",
                **({"ref_image": target} if has_reference else {}),
                h3_sampling_mode=mode, motion_control=motion_control, reuse_preprocessing=False, instruction="change clothes", resolution="768p (native)",
                aspect_ratio="same as reference", custom_aspect="16:9", seed=1, steps=2,
                sampler_name="res_multistep", scheduler="simple",
                use_qwen_edit=use_qwen, apply_fun_controlnet=use_control)
        picture = edited if use_qwen else (target if has_reference else source[:1])
        sent_images = conditioning.call_args.kwargs["ref_images"]
        if use_qwen or has_reference:
            self.assertEqual(list(sent_images), ["ref_image_0"])
            # ref_image is sliced [:1] before being sent, so compare by label, not identity
            self.assertEqual(sent_images["ref_image_0"].label, picture.label)
        else:
            self.assertEqual(sent_images, {})
        self.assertIsNone(conditioning.call_args.kwargs["audio_vae"])
        if not use_control:
            # control off: no Fun ControlNet, no pose extraction, source always a <Video 1> reference
            control.assert_not_called()
            pose.require_dwpose.assert_not_called()
            pose.extract_pose.assert_not_called()
            self.assertEqual(conditioning.call_args.kwargs["ref_videos"]["ref_video_0"].label, "source")
        elif motion_control == "pose only (DWPose)":
            self.assertIsNone(conditioning.call_args.kwargs["ref_videos"])
            self.assertEqual(control.call_args.kwargs["control_video"].label, "pose")
            pose.require_dwpose.assert_called_once()
            pose.extract_pose.assert_called_once()
        else:
            self.assertEqual(conditioning.call_args.kwargs["ref_videos"]["ref_video_0"].label, "source")
            self.assertEqual(control.call_args.kwargs["control_video"].label, "source")
            pose.require_dwpose.assert_not_called()
            pose.extract_pose.assert_not_called()
        if use_control:
            self.assertNotIn("mask", control.call_args.kwargs)
            self.assertNotIn("source_video", control.call_args.kwargs)
        self.assertEqual(node._enhance_edit.call_args.kwargs["motion_control"], motion_control)
        self.assertEqual(node._enhance_edit.call_args.kwargs["use_qwen_edit"], use_qwen)
        self.assertEqual(node.sample_args[5:8], (2, "res_multistep", "simple"))
        if use_qwen:
            self.assertIs(result[1], picture)
        else:
            self.assertEqual(result[1].label, picture.label)
        self.assertEqual(result[2:], ("H3 prompt", "qwen prompt"))
        if use_qwen:
            self.assertEqual([t.label for t in qwen.call_args.args[4]], ["source", "target"] if has_reference else ["source"])
        else:
            qwen.assert_not_called()
        return node, wrapper

    def test_native_matches_reference_routing_without_audio_wrapper(self):
        node, wrapper = self.run_mode("native AV (discard audio)")
        wrapper.assert_not_called()
        self.assertFalse(node.sample_kw["silent"])
        self.assertEqual(node.sample_args[0], "controlled-model")

    def test_reference_can_be_omitted_in_both_sampling_modes(self):
        self.run_mode("native AV (discard audio)", has_reference=False)
        self.run_mode("strict video-only (experimental)", has_reference=False)

    def test_pose_only_routes_skeleton_without_rgb_reference_or_mask(self):
        for mode in ("native AV (discard audio)", "strict video-only (experimental)"):
            for reference in (True, False):
                self.run_mode(mode, has_reference=reference, motion_control="pose only (DWPose)")

    def test_strict_remains_explicitly_video_only(self):
        node, wrapper = self.run_mode("strict video-only (experimental)")
        wrapper.assert_called_once_with("controlled-model")
        self.assertTrue(node.sample_kw["silent"])
        self.assertEqual(node.sample_args[0], "strict-model")

    def test_no_qwen_pose_mode_is_motion_transfer_from_reference(self):
        for mode in ("native AV (discard audio)", "strict video-only (experimental)"):
            node, wrapper = self.run_mode(mode, motion_control="pose only (DWPose)", use_qwen=False)
            # ref_image is sent to H3 as-is; skeletons drive the motion; no Qwen load

    def test_no_qwen_pose_without_reference_is_rejected(self):
        pose = types.SimpleNamespace(require_dwpose=Mock(), extract_pose=Mock())
        modules = {
            "comfy": types.ModuleType("comfy"),
            "comfy.samplers": types.SimpleNamespace(SCHEDULER_NAMES=["simple"]),
            "torch": types.SimpleNamespace(tensor=lambda value, **kw: value, long=object()),
            "nodes": types.SimpleNamespace(),
            "folder_paths": types.SimpleNamespace(),
            "comfy_extras": types.SimpleNamespace(nodes_minimax_h3=types.SimpleNamespace()),
            "routing_pkg.h3_prompter": types.SimpleNamespace(llama_client=types.SimpleNamespace(log=lambda msg: None)),
        }
        class Pipeline:
            def _load(self, kind, name):
                return kind
            def _base_models(self, **kw):
                return "m", "c", "v"
            def _sample(self, *a, **kw):
                return None, None
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3V2VGenerate")
        ns = dict(__name__="routing_pkg.nodes_compact", __package__="routing_pkg", _Pipeline=Pipeline,
                  CATEGORY="t", NONE="(none)", perf=perf, pose_control=pose,
                  prompter=types.SimpleNamespace(THINKING=["off"]), common_inputs=lambda: {}, extra_inputs=lambda: {},
                  choices=lambda *a, **kw: [], args=lambda o: o, patch_video_only=Mock(),
                  geometry=types.SimpleNamespace(video_timeline=lambda *a: (22, list(range(22))), canvas=lambda *a: (1, 1)),
                  v2v=types.SimpleNamespace(resize_frames=lambda image, *a: image))
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), ns)
        node = ns["MiniMaxH3V2VGenerate"]()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "ref_image"):
            node.generate(Tensor("source"), "fun", "q", "qc", "qv", "custom",
                          h3_sampling_mode="native AV (discard audio)", motion_control="pose only (DWPose)",
                          reuse_preprocessing=False, instruction="move like this",
                          use_qwen_edit=False, apply_fun_controlnet=True,
                          resolution="768p (native)", aspect_ratio="same as reference", custom_aspect="16:9",
                          seed=1, steps=2, sampler_name="res_multistep", scheduler="simple")

    def test_control_off_sends_source_as_plain_video_reference(self):
        for use_qwen in (True, False):
            for has_reference in (True, False):
                with self.subTest(use_qwen=use_qwen, has_reference=has_reference):
                    self.run_mode("native AV (discard audio)", has_reference=has_reference,
                                  use_qwen=use_qwen, use_control=False)

if __name__ == "__main__":
    unittest.main()
