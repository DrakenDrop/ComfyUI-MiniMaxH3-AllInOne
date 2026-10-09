"""FastH3 patch-chain tests plus inherited R2V reference/audio regressions (CPU)."""
import ast
import sys
import types
from unittest.mock import Mock, patch

import test_r2v_routing as routing


class Model:
    def __init__(self, history=()):
        self.history = history


class FastH3Tests(routing.R2VRoutingTests):
    def setUp(self):
        super().setUp()
        base = self.node.__class__
        self.base = base
        base.INPUT_TYPES.__func__.__globals__["common_inputs"] = lambda: {
            "h3_model": (["ref2va"],), "steps": ("INT", {"default": 20}),
            "scheduler": (["simple", "beta"], {"default": "beta"}),
            "sampler_name": (["res_multistep"], {"default": "res_multistep"})}
        source = routing.ROOT / "nodes_fasth3.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        # Execute the module except its relative import, using the actual R2V class.
        ns = dict(MiniMaxH3R2VGenerate=base,
                  args=lambda out: out.args if hasattr(out, "args") else out,
                  choices=lambda *a: ["fasth3", "ref2va"])
        exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n, ast.ImportFrom)],
                               type_ignores=[]), str(source), "exec"), ns)
        self.node.__class__ = ns["MiniMaxH3R2VFastH3Generate"]
        self.registration = ns
        del self.node._base_models
        self.loaded = Model(("load+optional-lora",))
        self.loader = Mock(return_value=(self.loaded, "clip", "video-vae"))
        loader_patch = patch.object(base, "_base_models", self.loader)
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        def output(name):
            return lambda model, **kw: types.SimpleNamespace(args=(Model(model.history + (name,)),))
        self.shift = Mock(side_effect=output("shift"))
        self.backend = Mock(side_effect=output("backend"))
        self.sparse = Mock(side_effect=output("vsa"))
        modules = {
            "comfy_extras.nodes_minimax_h3": types.SimpleNamespace(MiniMaxH3SigmaShift=types.SimpleNamespace(execute=self.shift)),
            "comfy_extras.nodes_model_advanced": types.SimpleNamespace(ModelAttentionBackend=types.SimpleNamespace(execute=self.backend)),
            "comfy_extras.nodes_sparse_attention": types.SimpleNamespace(BlockSparseAttention=types.SimpleNamespace(execute=self.sparse)),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_patch_order_and_exact_template_defaults(self):
        self.generate()
        self.shift.assert_called_once_with(model=self.loaded, shift_video=10.0, shift_audio=3.0)
        self.assertEqual(self.backend.call_args.kwargs["model"].history, ("load+optional-lora", "shift"))
        self.assertEqual(self.backend.call_args.kwargs["attention"], "comfy kitchen attention")
        sparse = self.sparse.call_args.kwargs
        self.assertEqual(sparse["model"].history, ("load+optional-lora", "shift", "backend"))
        self.assertEqual({k: v for k, v in sparse.items() if k != "model"}, {
            "selection": {"selection": "vsa", "keep_percent": 10.0}, "start_percent": 0.2,
            "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False})
        self.assertEqual(self.node._sample.call_args.args[0].history, ("load+optional-lora", "shift", "backend", "vsa"))

    def test_both_guider_and_scheduler_receive_final_patched_model(self):
        del self.node._sample  # Run the real shared sampler, mock only ComfyUI APIs.
        guider = Mock(return_value=("guider",))
        scheduler = Mock(return_value=("sigmas",))
        sample = Mock(return_value=("sampled",))
        custom = types.SimpleNamespace(BasicGuider=types.SimpleNamespace(execute=guider),
            BasicScheduler=types.SimpleNamespace(execute=scheduler), Noise_RandomNoise=Mock(return_value="noise"),
            SamplerCustomAdvanced=types.SimpleNamespace(execute=sample))
        comfy = sys.modules["comfy"]
        comfy.samplers = types.SimpleNamespace(sampler_object=lambda name: name)
        extras = sys.modules["comfy_extras"]
        nodes = sys.modules["nodes"]
        with patch.dict(sys.modules, {"comfy.samplers": comfy.samplers}), \
             patch.object(extras, "nodes_custom_sampler", custom, create=True), \
             patch.object(nodes, "VAEDecode", lambda: types.SimpleNamespace(decode=lambda *a: ("frames",)), create=True):
            self.generate(steps=8, scheduler="simple", ref_image_3=self.images[2], ref_audio_2=self.audios[1])
        model = guider.call_args.args[0]
        self.assertIs(scheduler.call_args.args[0], model)
        self.assertEqual(model.history, ("load+optional-lora", "shift", "backend", "vsa"))
        self.assertEqual(scheduler.call_args.args[1:], ("simple", 8, 1.0))
        self.assertEqual(sample.call_args.args, ("noise", "guider", "res_multistep", "sigmas", "latent"))
        self.assert_references([0, 2], True)

    def test_settings_propagate_and_repeated_runs_do_not_stack_patches(self):
        for shift, keep in ((10, 10), (12, 20)):
            self.generate(shift_video=shift, shift_audio=4, attention_backend="pytorch attention",
                vsa_keep_percent=keep, vsa_start_percent=0.1, vsa_end_percent=0.9,
                vsa_dense_blocks="1,3-5", vsa_min_tokens=4096, vsa_extra_tokens=64, vsa_verbose=True,
                lora_name="test.safetensors", lora_strength=0.4)
            self.assertIs(self.shift.call_args.kwargs["model"], self.loaded)
            self.assertEqual(self.loaded.history, ("load+optional-lora",))
            self.assertEqual(self.shift.call_args.kwargs["shift_video"], shift)
            self.assertEqual(self.shift.call_args.kwargs["shift_audio"], 4)
            self.assertEqual(self.backend.call_args.kwargs["attention"], "pytorch attention")
            self.assertEqual(self.sparse.call_args.kwargs["selection"]["keep_percent"], keep)
            self.assertEqual(self.sparse.call_args.kwargs["dense_blocks"], "1,3-5")
            self.assertEqual(self.sparse.call_args.kwargs["min_tokens"], 4096)
            self.assertEqual(self.sparse.call_args.kwargs["extra_tokens"], 64)
            self.assertEqual(self.sparse.call_args.kwargs["start_percent"], 0.1)
            self.assertEqual(self.sparse.call_args.kwargs["end_percent"], 0.9)
            self.assertTrue(self.sparse.call_args.kwargs["verbose"])
            self.assertEqual(self.loader.call_args.kwargs["lora_name"], "test.safetensors")
            self.assertEqual(self.loader.call_args.kwargs["lora_strength"], 0.4)

    def test_missing_native_nodes_fails_before_loading_without_breaking_schema(self):
        for module in ("nodes_minimax_h3", "nodes_model_advanced", "nodes_sparse_attention"):
            with self.subTest(module=module), patch.dict(sys.modules, {"comfy_extras." + module: None}):
                self.node.INPUT_TYPES()
                with self.assertRaisesRegex(RuntimeError, "Update ComfyUI"):
                    self.generate()
        self.loader.assert_not_called()

    def test_invalid_sparse_range_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "start_percent"):
            self.generate(vsa_start_percent=0.9, vsa_end_percent=0.2)
        self.loader.assert_not_called()

    def test_independent_schema_defaults_and_registration(self):
        original = self.base.INPUT_TYPES()
        fast = self.node.INPUT_TYPES()
        self.assertEqual(original["required"]["steps"][1]["default"], 20)
        self.assertEqual(original["required"]["scheduler"][1]["default"], "beta")
        self.assertNotIn("shift_video", original["optional"])
        self.assertEqual(fast["required"]["steps"][1]["default"], 8)
        self.assertEqual(fast["required"]["scheduler"][1]["default"], "simple")
        self.assertEqual(fast["required"]["h3_model"][0][0], "fasth3")
        self.assertEqual(self.node.RETURN_TYPES, self.base.RETURN_TYPES)
        self.assertIs(self.node.generate.__func__, self.base.generate)
        self.assertIs(self.registration["NODE_CLASS_MAPPINGS"]["MiniMaxH3R2VFastH3Generate"], self.node.__class__)
        self.assertIn("Experimental", self.registration["NODE_DISPLAY_NAME_MAPPINGS"]["MiniMaxH3R2VFastH3Generate"])
