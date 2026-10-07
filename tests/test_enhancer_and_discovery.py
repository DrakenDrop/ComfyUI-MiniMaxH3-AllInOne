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


    def test_v2v_rejects_reported_shot_anchor_in_both_styles(self):
        qwen = "Edit <image1> using the outfit in <image2>."
        sections = {**SECTIONS, "detailed_description":
                    "Live-action cinematic style. [Shot 1] The shot begins from <Picture 1>, "
                    "showing the woman in a black dress."}
        with self.assertRaises(fmt.V2VAnchorError):
            fmt.parse_edit_response(json.dumps({"qwen_prompt": qwen, "minimax_prompt": sections}))
        for phrase in ("The shot starts from <Picture 1>.",
                       "<Picture 1> is the first frame.",
                       "Use <Picture 1> as the keyframe.",
                       "[keyframe completion + video editing]"):
            with self.subTest(phrase=phrase), self.assertRaises(fmt.V2VAnchorError):
                fmt.parse_edit_response(json.dumps({"qwen_prompt": qwen, "minimax_prompt":
                    "Edit <Video 1> using <Picture 1>. " + phrase}), style="simple")

    def test_v2v_anchor_repair_preserves_original_qwen(self):
        qwen = "Edit <image1> using only clothing from <image2>."
        bad = json.dumps({"qwen_prompt": qwen, "minimax_prompt":
                         "Edit <Video 1>. The shot begins from <Picture 1>."})
        good = json.dumps({"qwen_prompt": "Different edit of <image1> from <image2>.",
                          "minimax_prompt": "Edit <Video 1> using the outfit in <Picture 1>."})
        from unittest.mock import Mock
        repair = Mock(return_value=good)
        q, m = fmt.parse_edit_response_with_repair(bad, "simple", repair)
        self.assertEqual(q, qwen)
        self.assertIn("outfit in <Picture 1>", m)
        repair.assert_called_once()
        repair.reset_mock()
        repair.return_value = bad
        with self.assertRaises(fmt.V2VAnchorError):
            fmt.parse_edit_response_with_repair(bad, "simple", repair)
        repair.assert_called_once()

    def test_valid_v2v_skips_repair_and_r2v_anchor_still_works(self):
        from unittest.mock import Mock
        repair = Mock()
        pair = json.dumps({"qwen_prompt": "Edit <image1> using <image2>.", "minimax_prompt": SECTIONS})
        fmt.parse_edit_response_with_repair(pair, "official", repair)
        repair.assert_not_called()
        anchored = fmt.apply_policy(SECTIONS, first_frame=True)
        self.assertIn("begins from <Picture 1>", anchored)


    def test_missing_h3_labels_repair_in_both_styles_with_optional_reference(self):
        from unittest.mock import Mock
        for style in ("simple", "official"):
            for has_reference in (True, False):
                for missing in ("<Picture 1>", "<Video 1>", "both"):
                    with self.subTest(style=style, has_reference=has_reference, missing=missing):
                        qwen = "Edit <image1> using <image2>." if has_reference else "Make the dress in <image1> red."
                        good_text = "Edit <Video 1> using the appearance in <Picture 1>."
                        labels = ("<Video 1>", "<Picture 1>") if missing == "both" else (missing,)
                        good = good_text if style == "simple" else dict(SECTIONS)
                        bad = good_text if style == "simple" else dict(SECTIONS)
                        for label in labels:
                            if isinstance(bad, str):
                                bad = bad.replace(label, "the supplied asset")
                            else:
                                bad = {k: v.replace(label, "the supplied asset") for k, v in bad.items()}
                        repair = Mock(return_value=json.dumps({"qwen_prompt": "do not use this changed text",
                                                               "minimax_prompt": good}))
                        q, m = fmt.parse_edit_response_with_repair(
                            json.dumps({"qwen_prompt": qwen, "minimax_prompt": bad}),
                            style, repair, has_reference=has_reference)
                        self.assertEqual(q, qwen)
                        self.assertIn("<Video 1>", m)
                        self.assertIn("<Picture 1>", m)
                        self.assertIn("missing", repair.call_args.args[0])
                        repair.assert_called_once()

    def test_h3_tag_spelling_normalizes_without_inventing_assets(self):
        from unittest.mock import Mock
        repair = Mock()
        text = "Edit < video1 > using <PICTURE  1 > for <subject1>."
        _, m = fmt.parse_edit_response_with_repair(json.dumps({
            "qwen_prompt": "Edit <image1> using <image2>.", "minimax_prompt": text}),
            "simple", repair)
        self.assertIn("<Video 1>", m)
        self.assertIn("<Picture 1>", m)
        self.assertIn("<Subject 1>", m)
        repair.assert_not_called()
        for bad in ("Edit the source video using the edited image.",
                    "Edit <Video 1> using <Picture 2>.",
                    "Edit <Video 1> and <Video 2> using <Picture 1>."):
            with self.subTest(bad=bad), self.assertRaises(fmt.V2VPromptError):
                fmt.parse_edit_response(json.dumps({
                    "qwen_prompt": "Edit <image1> using <image2>.", "minimax_prompt": bad}), style="simple")

    def test_h3_format_repair_is_bounded_and_does_not_retry_invalid_qwen(self):
        from unittest.mock import Mock
        good = json.dumps({"qwen_prompt": "Edit <image1> using <image2>.", "minimax_prompt": SECTIONS})
        bad = json.dumps({"qwen_prompt": "Edit <image1> using <image2>.", "minimax_prompt":
                         {k: v for k, v in SECTIONS.items() if k != "summary"}})
        repair = Mock(return_value=good)
        self.assertEqual(fmt.parse_edit_response_with_repair(bad, "official", repair)[1], PLAIN)
        repair.assert_called_once()
        repair.reset_mock()
        repair.return_value = bad
        with self.assertRaisesRegex(fmt.V2VPromptError, "after one repair attempt"):
            fmt.parse_edit_response_with_repair(bad, "official", repair)
        repair.assert_called_once()
        repair.reset_mock()
        with self.assertRaises(ValueError):
            fmt.parse_edit_response_with_repair(
                json.dumps({"qwen_prompt": "", "minimax_prompt": SECTIONS}), "official", repair)
        repair.assert_not_called()

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
