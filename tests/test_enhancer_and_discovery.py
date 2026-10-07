"""CPU-only regression tests for prompt formatting and executable discovery."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "h3_prompter" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fmt = load("format_under_test", "prompt_format.py")
discovery = load("discovery_under_test", "server_discovery.py")
SECTIONS = dict(zip(fmt.FIELDS, (
    "<Video 1> is the source. <Picture 1> is the Qwen-edited appearance reference.",
    "[video editing] Change the outfit.",
    "<Video 1>: fully_preserved - motion.",
    "[Shot 1] The person walks in the target outfit.", "N/A", "N/A")))
PLAIN = "\n\n".join(f"{k}:\n{v}" for k, v in SECTIONS.items())

class PromptRegressionTests(unittest.TestCase):
    def test_official_markdown_and_reordered_headings(self):
        text = "\n\n".join(f"## **{k.replace('_', ' ').title()}:**\n{v}" for k, v in reversed(list(SECTIONS.items())))
        self.assertEqual(fmt.apply_policy(text, silent=True), PLAIN)

    def test_json_sections_and_escaped_newlines(self):
        self.assertEqual(fmt.apply_policy(json.dumps(SECTIONS)), PLAIN)
        self.assertEqual(fmt.apply_policy(PLAIN.replace("\n", "\\n")), PLAIN)

    def test_missing_and_duplicate_sections_fail_with_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "Missing: non_diegetic_music"):
            fmt.apply_policy(PLAIN.rsplit("\n\n", 1)[0])
        with self.assertRaises(ValueError):
            fmt.apply_policy(PLAIN + "\nsummary:\nduplicate")

    def test_dual_response_official_and_simple(self):
        qwen = "Edit <image1> using the outfit in <image2>."
        pair = {"qwen_prompt": qwen, "minimax_prompt": SECTIONS}
        self.assertEqual(fmt.parse_edit_response(json.dumps(pair)), (qwen, PLAIN))
        simple = "Edit <Video 1> using the appearance in <Picture 1>."
        pair["minimax_prompt"] = simple
        q, m = fmt.parse_edit_response(json.dumps(pair), style="simple")
        self.assertEqual(q, qwen)
        self.assertTrue(m.startswith(simple))
        self.assertIn("Silent video", m)
        self.assertNotIn("subject_definitions", m)

    def test_simple_first_frame_and_audio_rejection(self):
        result = fmt.apply_policy("A person walks.", first_frame=True, style="simple")
        self.assertIn("begins from <Picture 1>", result)
        with self.assertRaises(ValueError):
            fmt.apply_policy("Use <Audio 1>.", silent=True, style="simple")
        with self.assertRaises(ValueError):
            fmt.parse_edit_response('{"qwen_prompt": "unfinished')

class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.which = patch.object(discovery.shutil, "which", return_value=None)
        self.which_mock = self.which.start()
        self.addCleanup(self.which.stop)
        self.roots = patch.object(discovery, "_search_roots", return_value=[str(self.root)])
        self.roots.start()
        self.addCleanup(self.roots.stop)

    def binary(self, relative):
        path = self.root / relative / discovery._exe_name()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder; discovery must never run this file")
        path.chmod(0o755)
        return str(path)

    def test_path_and_explicit_override_priority(self):
        on_path = self.binary("path-bin")
        explicit = self.binary("chosen/build/bin/Release")
        self.which_mock.side_effect = lambda value: on_path if value == discovery._exe_name() else None
        self.assertEqual(discovery.discover({}, str(self.root)), on_path)
        self.assertEqual(discovery.discover({"llama_server_path": "chosen"}, str(self.root)), explicit)
        with self.assertRaisesRegex(FileNotFoundError, "auto"):
            discovery.discover({"llama_server_path": "missing.exe"}, str(self.root))

    def test_environment_and_release_archive(self):
        expected = self.binary("llama-b9999-bin-win-cuda/bin")
        self.assertEqual(discovery.discover({"llama_server_path": "auto"}, str(self.root)), expected)
        other = self.binary("my-tools")
        os.environ["LLAMA_SERVER_PATH"] = other
        self.assertEqual(discovery.discover({}, str(self.root)), other)

    def test_build_layout_and_absence(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            discovery.discover({}, str(self.root))
        expected = self.binary("llama.cpp/build/bin")
        self.assertEqual(discovery.discover({}, str(self.root)), expected)

    def test_search_does_not_recurse_arbitrarily(self):
        self.binary("unrelated/deep/llama.cpp/build/bin")
        with self.assertRaises(FileNotFoundError):
            discovery.discover({}, str(self.root))

if __name__ == "__main__":
    unittest.main()
