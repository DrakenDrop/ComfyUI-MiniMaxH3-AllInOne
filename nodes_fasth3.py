"""Experimental R2V/V2V with the model patches from the FastH3 I2V template."""
from .nodes_compact import MiniMaxH3R2VGenerate, MiniMaxH3V2VGenerate, args, choices


def _fasth3_precheck(vsa_start_percent=0.2, vsa_end_percent=1.0):
    """Fail before any model loading when native patch nodes are missing or the range is invalid."""
    if not 0.0 <= vsa_start_percent <= vsa_end_percent <= 1.0:
        raise ValueError("VSA requires 0 <= start_percent <= end_percent <= 1.")
    # Lazy imports keep the original nodes usable on older ComfyUI versions.
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift  # noqa: F401
        from comfy_extras.nodes_model_advanced import ModelAttentionBackend  # noqa: F401
        from comfy_extras.nodes_sparse_attention import BlockSparseAttention  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "FastH3 variants require native MiniMaxH3SigmaShift, ModelAttentionBackend and "
            "BlockSparseAttention with VSA. Update ComfyUI and its Comfy Kitchen dependencies."
        ) from exc


def _fasth3_patches(
    model,
    shift_video=10.0,
    shift_audio=3.0,
    attention_backend="comfy kitchen attention",
    vsa_keep_percent=10.0,
    vsa_start_percent=0.2,
    vsa_end_percent=1.0,
    vsa_dense_blocks="",
    vsa_min_tokens=12288,
    vsa_extra_tokens=256,
    vsa_sink_conditioning="exact_kv_and_rows",
    vsa_verbose=False,
):
    _fasth3_precheck(vsa_start_percent, vsa_end_percent)
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    # Native patches clone their input. Never replace the cached loader model,
    # so repeated runs and parameter changes do not stack old VSA wrappers.
    model = args(MiniMaxH3SigmaShift.execute(model=model, shift_video=shift_video, shift_audio=shift_audio))[0]
    model = args(ModelAttentionBackend.execute(model=model, attention=attention_backend))[0]
    model = args(BlockSparseAttention.execute(
        model=model, selection={"selection": "vsa", "keep_percent": vsa_keep_percent},
        start_percent=vsa_start_percent, end_percent=vsa_end_percent,
        dense_blocks=vsa_dense_blocks, min_tokens=vsa_min_tokens, extra_tokens=vsa_extra_tokens,
        sink_conditioning=vsa_sink_conditioning, verbose=vsa_verbose))[0]
    return model


def _fasth3_inputs(schema):
    """Apply the FastH3 schema overrides (model preference, 8-step defaults, patch widgets)."""
    required, optional = schema["required"], schema["optional"]
    required["h3_model"] = (choices("diffusion_models", ["fasth3", "ref2va"]),)
    required["steps"] = ("INT", {"default": 8, "min": 1, "max": 100})
    required["scheduler"] = (required["scheduler"][0], {"default": "simple"})
    optional.update({
        "shift_video": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        "attention_backend": (["comfy kitchen attention", "pytorch attention"], {"default": "comfy kitchen attention",
            "tooltip": "Dense backend when VSA is inactive. Native ComfyUI warns and falls back to PyTorch if Comfy Kitchen attention is unavailable."}),
        "vsa_keep_percent": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 95.0, "step": 0.5}),
        "vsa_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
        "vsa_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "vsa_dense_blocks": ("STRING", {"default": "", "tooltip": "Comma-separated block indices/ranges to keep dense; empty matches the supplied template."}),
        "vsa_min_tokens": ("INT", {"default": 12288, "min": 0, "max": 1000000, "step": 256}),
        "vsa_extra_tokens": ("INT", {"default": 256, "min": 0, "max": 1000000, "step": 64}),
        "vsa_sink_conditioning": (["exact_kv_and_rows"], {"default": "exact_kv_and_rows",
            "tooltip": "Keep conditioning keys, values and query rows exact, as in the supplied workflow."}),
        "vsa_verbose": ("BOOLEAN", {"default": False}),
    })
    return schema


class MiniMaxH3R2VFastH3Generate(MiniMaxH3R2VGenerate):
    DESCRIPTION = (
        "Experimental R2V: keeps three image / two audio references and applies H3 Sigma Shift, "
        "Model Attention Backend, then VSA Sparse Attention before both guider and scheduler. "
        "FastH3 8-Step V2 was not distilled for Ref2VA; reference fidelity, audio and speed are unvalidated. "
        "Requires recent ComfyUI with these native patch nodes and compatible Comfy Kitchen kernels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return _fasth3_inputs(super().INPUT_TYPES())

    def _base_models(self, shift_video=10.0, shift_audio=3.0,
                     attention_backend="comfy kitchen attention", vsa_keep_percent=10.0,
                     vsa_start_percent=0.2, vsa_end_percent=1.0, vsa_dense_blocks="",
                     vsa_min_tokens=12288, vsa_extra_tokens=256,
                     vsa_sink_conditioning="exact_kv_and_rows", vsa_verbose=False, **kw):
        _fasth3_precheck(vsa_start_percent, vsa_end_percent)  # fail before loading any model
        model, clip, vae = super()._base_models(**kw)
        return (_fasth3_patches(model, shift_video=shift_video, shift_audio=shift_audio,
                                attention_backend=attention_backend, vsa_keep_percent=vsa_keep_percent,
                                vsa_start_percent=vsa_start_percent, vsa_end_percent=vsa_end_percent,
                                vsa_dense_blocks=vsa_dense_blocks, vsa_min_tokens=vsa_min_tokens,
                                vsa_extra_tokens=vsa_extra_tokens, vsa_sink_conditioning=vsa_sink_conditioning,
                                vsa_verbose=vsa_verbose), clip, vae)


class MiniMaxH3V2VFastH3Generate(MiniMaxH3V2VGenerate):
    DESCRIPTION = (
        "Experimental silent V2V: inherits the full V2V pipeline (optional Qwen first-frame edit, "
        "DWPose pose control or legacy RGB, the no-Qwen direct modes and the Fun ControlNet toggle) "
        "and applies H3 Sigma Shift, Model Attention Backend, then VSA Sparse Attention to the H3 model "
        "before the Fun ControlNet and scheduler. "
        "FastH3 8-Step V2 was not distilled for these V2V tasks; edit quality, motion fidelity and speed "
        "are unvalidated. Requires recent ComfyUI with these native patch nodes and compatible Comfy Kitchen kernels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return _fasth3_inputs(super().INPUT_TYPES())

    def _base_models(self, shift_video=10.0, shift_audio=3.0,
                     attention_backend="comfy kitchen attention", vsa_keep_percent=10.0,
                     vsa_start_percent=0.2, vsa_end_percent=1.0, vsa_dense_blocks="",
                     vsa_min_tokens=12288, vsa_extra_tokens=256,
                     vsa_sink_conditioning="exact_kv_and_rows", vsa_verbose=False, **kw):
        _fasth3_precheck(vsa_start_percent, vsa_end_percent)  # fail before loading any model
        model, clip, vae = super()._base_models(**kw)
        return (_fasth3_patches(model, shift_video=shift_video, shift_audio=shift_audio,
                                attention_backend=attention_backend, vsa_keep_percent=vsa_keep_percent,
                                vsa_start_percent=vsa_start_percent, vsa_end_percent=vsa_end_percent,
                                vsa_dense_blocks=vsa_dense_blocks, vsa_min_tokens=vsa_min_tokens,
                                vsa_extra_tokens=vsa_extra_tokens, vsa_sink_conditioning=vsa_sink_conditioning,
                                vsa_verbose=vsa_verbose), clip, vae)


NODE_CLASS_MAPPINGS = {"MiniMaxH3R2VFastH3Generate": MiniMaxH3R2VFastH3Generate,
                       "MiniMaxH3V2VFastH3Generate": MiniMaxH3V2VFastH3Generate}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3R2VFastH3Generate": "MiniMax H3 R2V FastH3 Generate (Experimental)",
    "MiniMaxH3V2VFastH3Generate": "MiniMax H3 V2V FastH3 Generate (Experimental)",
}
