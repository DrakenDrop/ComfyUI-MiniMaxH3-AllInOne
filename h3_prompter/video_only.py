"""Experimental H3 video-only denoising using ComfyUI's public wrapper hook.

The sampler retains its AV container for core compatibility. Audio is always a
zero placeholder. Inside the transformer, the target audio has ZERO tokens.
This differs from generating a soundtrack and omitting it at export.
"""


def _video_only_forward(executor, x, timestep, context, transformer_options=None,
                        minimax_payload=None, audio_denoise_mask=None, **kwargs):
    import torch

    if len(x) != 2 or x[1].ndim != 4:
        raise RuntimeError("Video-only mode requires native H3's video/audio pair")
    payload = dict(minimax_payload or {})
    if payload.get("cond_audio_latents") or any(
        r.get("audio_latent") is not None or r.get("kind") in ("audio", "video_audio")
        for r in payload.get("refs", [])
    ):
        raise ValueError("V2V video-only mode does not accept audio conditioning")
    # The native model rebuilds PackedLayout when its signature differs. Do not
    # mutate the shared conditioning payload or another node's model.
    payload.pop("layout", None)
    silent_x = [x[0], x[1][..., :0]]
    out = executor(silent_x, timestep, context,
                   transformer_options=dict(transformer_options or {}),
                   minimax_payload=payload, audio_denoise_mask=None, **kwargs)
    if out[1].shape[-1] != 0:
        raise RuntimeError("H3 returned audio tokens in strict video-only mode")
    return [out[0], torch.zeros_like(x[1])]


def patch_video_only(model):
    from comfy.patcher_extension import WrappersMP

    diffusion = model.model.diffusion_model
    if not all(hasattr(diffusion, attr) for attr in ("audio_patch_proj", "_forward", "final_layer")):
        raise RuntimeError("Strict video-only mode requires the native MiniMax H3 model")
    out = model.clone()
    out.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "minimax_compact_video_only", _video_only_forward)
    return out


class VideoOnlyNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, latent):
        import torch
        import comfy.sample
        from comfy.nested_tensor import NestedTensor

        streams = latent["samples"].tensors
        # Generate randomness only for the video, not a discarded audio stream.
        video = comfy.sample.prepare_noise(streams[0], self.seed, latent.get("batch_index"))
        return NestedTensor((video, torch.zeros_like(streams[1])))
