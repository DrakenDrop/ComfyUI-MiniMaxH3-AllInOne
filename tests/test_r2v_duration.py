"""Verify selected duration reaches real LLM messages without fixed action plans."""
import ast
import os
import time
import types
import unittest
from unittest.mock import Mock

import test_r2v_routing as routing

prompts = routing.load("prompts.py")


class DurationTests(unittest.TestCase):
    def enhancer(self):
        tree = ast.parse((routing.ROOT / "nodes.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3R2VPrompter")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate"]
        ns = dict(local_models=types.SimpleNamespace(SERVER_DEFAULT="server", MMPROJ_AUTO="auto",
                      resolve=lambda *a: (None, None)),
            _CFG={"server_url": "http://local"}, H3_MAX_TRAINED_FRAMES=362,
            MAX_VIDEOS=1, FIRST_FIELD="subject_definitions:", SAMPLING={"off": {}, "on": {}},
            media=types.SimpleNamespace(H3_FPS=24), lc=types.SimpleNamespace(ensure_server=Mock(), log=Mock()),
            prompts=prompts, time=time, os=os, _blank_image=lambda: object(),
            _drop_video_parts=lambda parts: parts)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "nodes.py", "exec"), ns)
        node = ns["MiniMaxH3R2VPrompter"]()
        node._collect_assets = Mock(return_value=([], ["<Picture 1>: a person"], [], ["<Audio 2>: voice"], object()))
        node._run = Mock(return_value=(routing.PROMPT, "", {}))
        return node

    def test_real_enhancer_messages_include_selected_duration_in_both_styles(self):
        for simple in (False, True):
            messages = []
            for seconds in (5, 10, 15):
                frames = routing.geometry.generation_frames(seconds)
                # Capture the actual shared pipeline arguments, then run the real message builder.
                fixture = routing.R2VRoutingTests()
                fixture.setUp()
                try:
                    fixture.generate(duration_seconds=seconds, minimax_prompt_style="simple" if simple else "official")
                    kwargs = dict(fixture.enhancer.call_args.kwargs)
                finally:
                    fixture.doCleanups()
                node = self.enhancer()
                result = node.generate(**kwargs)
                system, user = node._run.call_args.args[2:4]
                self.assertIn(f"This video is for {seconds} seconds.", user["content"])
                self.assertIn(f"appropriate for a {seconds}-second video", user["content"])
                self.assertIn(kwargs["extra_rules"], user["content"])
                self.assertNotIn("Internal planning windows", user["content"])
                self.assertNotIn("ONE clear main action", user["content"])
                self.assertIn("<Picture 1>", user["content"])
                self.assertIn("<Audio 2>", user["content"])
                self.assertEqual(result[1:3], (frames, frames / 24))
                if simple:
                    self.assertEqual(system["content"], prompts.SYSTEM_PROMPT_SIMPLE_R2V)
                    self.assertNotIn("Never describe motion", system["content"])
                else:
                    self.assertIn(f"{frames} frames at 24 fps", user["content"])
                    self.assertIn(f"Target video duration: {frames / 24:.2f} seconds", user["content"])
                messages.append(user["content"])
            self.assertEqual(len(set(messages)), 3)

    def test_legacy_simple_enhancement_keeps_original_system(self):
        node = self.enhancer()
        node.generate(instruction="Change the dress.", task="video editing", frame_anchor="none",
            duration_seconds=5, thinking="off", length="standard", allow_invented_dialogue=False,
            max_tokens=3072, seed=1, frame_count=124, prompt_style="simple")
        self.assertEqual(node._run.call_args.args[2]["content"], prompts.SYSTEM_PROMPT_SIMPLE)
