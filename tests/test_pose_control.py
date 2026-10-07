"""Pose-only integration contracts without downloading or running pose weights."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pose_pkg.pose_control", ROOT / "h3_prompter/pose_control.py")
pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose)


class PoseControlTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.rand(3, 8, 8, 3)
        self.skeleton = torch.zeros_like(self.source)
        self.skeleton[:, :, 4, 0] = 1
        self.log = Mock()
        self.detector = Mock(return_value={"result": (self.skeleton, [{"people": [{}]}] * 3)})
        cls = type("DWPose", (), {"estimate_pose": self.detector})
        self.modules = {
            "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={"DWPreprocessor": cls}),
            "pose_pkg": types.SimpleNamespace(llama_client=types.SimpleNamespace(log=self.log)),
        }

    def test_extracts_all_frames_and_body_hands_face_without_masks(self):
        with patch.dict(sys.modules, self.modules):
            result = pose.extract_pose(self.source, 512)
        self.assertIs(result, self.skeleton)
        kw = self.detector.call_args.kwargs
        self.assertIs(kw["image"], self.source)
        self.assertEqual(kw["detect_body"], "enable")
        self.assertEqual(kw["detect_hand"], "enable")
        self.assertEqual(kw["detect_face"], "enable")
        self.assertNotIn("mask", kw)
        self.assertTrue(kw["bbox_detector"].endswith(".torchscript.pt"))
        self.assertTrue(kw["pose_estimator"].endswith(".torchscript.pt"))

    def test_missing_dependency_fails_without_rgb_fallback(self):
        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={})}):
            with self.assertRaisesRegex(RuntimeError, "comfyui_controlnet_aux"):
                pose.require_dwpose()

    def test_wrong_frame_count_and_empty_detection_fail(self):
        with patch.dict(sys.modules, self.modules):
            self.detector.return_value = {"result": (self.skeleton[:1], [{"people": [{}]}])}
            with self.assertRaisesRegex(ValueError, "timing"):
                pose.extract_pose(self.source)
            self.detector.return_value = {"result": (self.skeleton, [{"people": []}] * 3)}
            with self.assertRaisesRegex(ValueError, "no people"):
                pose.extract_pose(self.source)

    def test_partial_missing_detections_are_reported_without_padding(self):
        self.detector.return_value = {"result": (
            self.skeleton, [{"people": [{}]}, {"people": []}, {"people": [{}]}])}
        with patch.dict(sys.modules, self.modules):
            self.assertIs(pose.extract_pose(self.source), self.skeleton)
        self.assertIn("1/3 frames", self.log.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
