"""Direct H3 character replacement: reference video + character image, no image editor."""
import os

from . import nodes as prompter
from .nodes_compact import _Pipeline, args, choices, common_inputs, extra_inputs, CFG, CATEGORY, NONE
from .h3_prompter import canvas as geometry, local_models, managed_server, media, performance as perf, v2v
from .h3_prompter import llama_client as lc
from .h3_prompter.prompt_format import FIELDS, prompt_field_unchecked
from .h3_prompter.video_only import patch_video_only

SYSTEM_PROMPT = """Write a MiniMax H3 character-swap prompt from the user instruction and visual references.
<Video 1> is the source video and supplies the performance, motion, gestures,
expressions, timing, camera, composition, environment, background and lighting.
<Picture 1> is the supplied target character image, directly connected to H3.
Replace the requested person in <Video 1> with the target character from <Picture 1>.
Unless the instruction says otherwise, transfer the target character's identity,
face, hair and outfit. Preserve source performance, camera and environment; do not
require preservation of the original person's identity. If several people are
visible, use the user's description to identify who changes and preserve others.
Do not copy the picture's background, pose, lighting or camera into the video.
Adapt the target appearance to the source scene's lighting, exposure and color cast.
The swapped character is present throughout, without an on-screen transformation.
Do not invent gestures, shots, new lighting or additional characters.
Use [video editing]. <Picture 1> is a visual reference, not a pinned frame guide.
There is no image-editing stage. Both references already exist and are shown to you.
Write in English. Request silent video; no dialogue, effects or music.
Return JSON with one field named minimax_prompt. No Markdown fences."""


class MiniMaxH3CharacterSwap(_Pipeline):
    CATEGORY = CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "minimax_prompt")
    DESCRIPTION = "Reference video frames + character image directly into H3, with RGB FunControlNet. Internal model loading, prompting, sampling and VAE decode. No image editor or masks."

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        required = {
            "ref_video": ("IMAGE", {"tooltip": "Source video frames at 24 FPS, matching the official H3 ref_video IMAGE input. Connect Load Video's IMAGE output."}),
            "ref_image": ("IMAGE", {"tooltip": "Target character image. Goes directly to H3 as Picture 1; uses the first image if a batch is connected."}),
            **common_inputs(),
            "fun_controlnet": (choices("model_patches", ["minimax_h3_fun"]), {"tooltip": "H3 FunControlNet weights. Uses the same ref_video frames as RGB motion control."}),
        }
        required["instruction"] = ("STRING", {"multiline": True, "default": "Replace the main person in the video with the character in the reference image. Preserve source motion, camera, background and lighting."})
        required["scheduler"] = (list(comfy.samplers.SCHEDULER_NAMES), {"default": "simple"})
        optional = extra_inputs()
        optional["ref_image_size"] = (["match", "max"], {"default": "max"})
        optional.update({
            "control_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 3, "step": 0.05}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0, "max": 3600}),
            "max_seconds": ("FLOAT", {"default": 15.08, "min": 0.21, "max": 15.08, "tooltip": "Official H3 frame alignment at 24 FPS: 10s becomes 243 frames. Missing alignment frames repeat the source tail."}),
            "h3_sampling_mode": (["strict video-only (experimental)", "native AV (discard audio)"], {"default": "strict video-only (experimental)", "tooltip": "Strict removes audio tokens. Native AV computes audio latents internally but never decodes or returns audio."}),
            "reuse_preprocessing": ("BOOLEAN", {"default": True, "tooltip": "Cache the last identical prompt request for a managed local model."}),
            "minimax_thinking": (list(prompter.THINKING), {"default": "off"}),
            "keep_models_loaded": ("BOOLEAN", {"default": False, "tooltip": "Skip forced ComfyUI model unload before enhancement. ComfyUI can still offload as needed; leave memory available for llama.cpp."}),
        })
        return {"required": required, "optional": optional}

    def _enhance_character(self, source, reference, instruction, frames, **kw):
        override = kw.get("prompt_override", "")
        if override.strip():
            return override
        import comfy.model_management
        model_path, mmproj_path = local_models.resolve(kw["llm_model"], kw["mmproj"], CFG)
        if model_path and mmproj_path is None:
            raise ValueError("Character-swap enhancement needs a vision GGUF with its matching mmproj, or prompt_override.")
        thinking = kw.get("minimax_thinking", "off")
        if thinking not in prompter.THINKING:
            raise ValueError(f"Unknown minimax_thinking: {thinking}")
        style = kw.get("minimax_prompt_style", "official")
        if style not in ("official", "simple"):
            raise ValueError(f"Unknown MiniMax prompt style: {style}")
        system = SYSTEM_PROMPT + ("\nminimax_prompt is an object with these six text fields: " +
            ", ".join(FIELDS) + ". Use N/A for both sound fields." if style == "official" else
            "\nminimax_prompt is a string with 2-4 concise sentences, without section headings.")
        if kw.get("additional_system_prompt", "").strip():
            system += "\nAdditional user preferences:\n" + kw["additional_system_prompt"]
        parts = [{"type": "text", "text": f"Instruction: {instruction}\nTarget length: {frames} frames at 24 FPS."}]
        for i in media.sample_indices(len(source), 8):
            parts.extend([
                {"type": "text", "text": f"<Video 1> source frame at {i / 24:.3f} seconds:"},
                {"type": "image_url", "image_url": {"url": media.pil_to_data_url(media.image_batch_to_pil(source[i:i + 1])[0], 512)}},
            ])
        parts.extend([
            {"type": "text", "text": "<Picture 1>: target character image, directly supplied to H3."},
            {"type": "image_url", "image_url": {"url": media.pil_to_data_url(media.image_batch_to_pil(reference)[0], 768)}},
        ])
        alias = os.path.splitext(os.path.basename(model_path))[0] if model_path else CFG.get("model_alias", "qwen3.8-27b")
        base = dict(model=alias, max_tokens=int(kw.get("max_tokens", 3072)), seed=int(kw["seed"]) % (2**32),
                    temperature=0.2, top_p=0.8, top_k=20, cache_prompt=True, response_format={"type": "json_object"})
        if not hasattr(self, "_character_prompt_cache"):
            self._character_prompt_cache = perf.SingleEntryCache()
        cache = self._character_prompt_cache
        reuse = kw.get("reuse_preprocessing", True) and model_path is not None
        key = None
        if reuse:
            key = perf.request_key("character-swap-unchecked-v1", system, parts, base, thinking,
                                   kw.get("context_size", 32768), CFG, perf.file_stamp(model_path), perf.file_stamp(mmproj_path))
            cached = cache.get(key)
            if cached is not None:
                lc.log("Character swap: enhancer cache hit.")
                if kw.get("unload_llm_after_prompt", True):
                    managed_server.stop(CFG)
                return cached
        else:
            cache.clear()
        if not kw.get("keep_models_loaded", False):
            comfy.model_management.unload_all_models()
        server_url = (kw.get("server_url") or CFG["server_url"]).strip()
        if model_path:
            server_url = managed_server.ensure(model_path, mmproj_path, kw.get("context_size", 32768), CFG)
        else:
            lc.ensure_server(server_url, CFG)
        try:
            lc.log(f"Character swap: enhancing MiniMax prompt; thinking={thinking}.")
            content = prompter.MiniMaxH3R2VPrompter._run(server_url, base,
                {"role": "system", "content": system}, {"role": "user", "content": parts}, thinking,
                float(CFG.get("request_timeout_seconds", 600)), False, None, prefill=None)[0]
            prompt = prompt_field_unchecked(content, "minimax_prompt")
        finally:
            if kw.get("unload_llm_after_prompt", True) and model_path:
                managed_server.stop(CFG)
        if reuse:
            cache.put(key, prompt)
        lc.log("Character swap: enhancer output accepted without prompt validation or correction retry.")
        return prompt

    def generate(self, ref_video, ref_image, fun_controlnet, control_strength=1.0,
                 start_seconds=0.0, max_seconds=15.08,
                 h3_sampling_mode="strict video-only (experimental)", **kw):
        from comfy_extras import nodes_minimax_h3 as h3
        timer = perf.StageTimer(lc.log)
        for name, tensor in (("ref_video", ref_video), ("ref_image", ref_image)):
            if tensor.ndim != 4 or tensor.shape[-1] < 3 or len(tensor) == 0:
                raise ValueError(f"{name} must be a non-empty IMAGE batch [frames, height, width, RGB].")
        if h3_sampling_mode not in ("strict video-only (experimental)", "native AV (discard audio)"):
            raise ValueError(f"Unknown H3 sampling mode: {h3_sampling_mode}")
        strict = h3_sampling_mode == "strict video-only (experimental)"
        original = ref_video[..., :3]
        frames, indices = geometry.video_timeline(len(original), 24.0, start_seconds, max_seconds)
        w, h = geometry.canvas(kw["resolution"], kw["aspect_ratio"], original.shape[2], original.shape[1], kw["custom_aspect"])
        source = v2v.resize_frames(perf.take_frames(original, indices), w, h)
        reference = ref_image[:1, ..., :3]
        tail = max(0, sum(i == len(original) - 1 for i in indices) - 1)
        lc.log(f"Character swap timeline: {frames} frames ({frames / 24:.3f}s); repeated_tail_frames={tail}.")
        timer.mark("prepare_frames")
        instruction = kw.get("instruction", "").strip() or "Replace the main person in <Video 1> with the character in <Picture 1>; preserve source performance and environment."
        prompt = self._enhance_character(source, reference, frames=frames, **{**kw, "instruction": instruction})
        timer.mark("enhancer")
        model, clip, vae = self._base_models(**kw)
        timer.mark("h3_model_load")
        positive, latent = args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=None, prompt=prompt, width=w, height=h, length=frames,
            ref_image_size=kw.get("ref_image_size", "max"), ref_images={"ref_image_0": reference},
            ref_videos={"ref_video_0": source}, ref_video_audios=None, ref_audios=None))[:2]
        timer.mark("h3_conditioning")
        if control_strength != 0:
            model = args(h3.MiniMaxH3FunControlNetApply.execute(model=model,
                model_patch=self._load("patch", fun_controlnet), vae=vae, strength=control_strength,
                start_percent=0.0, end_percent=1.0, control_video=source))[0]
        if strict:
            model = patch_video_only(model)
        timer.mark("control_setup")
        lc.log(f"Character swap: direct image + video references; control={fun_controlnet}@{control_strength}; mode={h3_sampling_mode}.")
        images, _ = self._sample(model, positive, latent, vae, kw["seed"], kw["steps"],
                                 kw["sampler_name"], kw["scheduler"], silent=strict, timer=timer)
        timer.finish()
        return images[:frames], prompt


NODE_CLASS_MAPPINGS = {"MiniMaxH3CharacterSwap": MiniMaxH3CharacterSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3CharacterSwap": "MiniMax H3 Character Swap (Sample + VAE Decode)"}
