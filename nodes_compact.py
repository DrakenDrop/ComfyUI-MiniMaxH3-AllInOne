"""Self-contained H3 R2V / V2V nodes: load, prompt, condition, sample, decode."""
from __future__ import annotations

import os
from . import nodes as prompter
from .h3_prompter import canvas as geometry, local_models, managed_server, v2v
from .h3_prompter.video_only import patch_video_only, VideoOnlyNoise
from .h3_prompter.prompt_format import apply_policy, parse_edit_response

CFG = prompter._CFG
CATEGORY = "MiniMax H3/All in One"
NONE = "(none)"


def args(output):
    return output.args if hasattr(output, "args") else output


def choices(folder, preferred, optional=False):
    import folder_paths
    values = folder_paths.get_filename_list(folder)
    def score(filename):
        return min((i for i, token in enumerate(preferred) if token in filename.lower()), default=len(preferred))
    values = sorted(values, key=lambda f: (score(f), f.lower()))
    return ([NONE] if optional else []) + (values or ["(no models found)"])


def common_inputs():
    import comfy.samplers
    return {
        "h3_model": (choices("diffusion_models", ["ref2va"]),),
        "h3_text_encoder": (choices("text_encoders", ["minimax", "h3"]), {"tooltip": "H3 CLIP/text encoder loaded internally from models/text_encoders."}),
        "h3_video_vae": (choices("vae", ["h3_video", "lynnreal"]), {"tooltip": "H3 video VAE loaded internally from models/vae."}),
        "llm_model": (local_models.model_choices(CFG),),
        "mmproj": (local_models.mmproj_choices(CFG), {"default": "auto"}),
        "instruction": ("STRING", {"multiline": True, "default": ""}),
        "resolution": (geometry.RESOLUTIONS, {"default": "480p"}),
        "aspect_ratio": (geometry.ASPECTS, {"default": "same as reference"}),
        "custom_aspect": ("STRING", {"default": "16:9", "tooltip": "Used only when aspect_ratio=custom; width:height"}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
        "sampler_name": (list(comfy.samplers.SAMPLER_NAMES), {"default": "res_multistep"}),
        "scheduler": (list(comfy.samplers.SCHEDULER_NAMES), {"default": "beta"}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
    }


def extra_inputs():
    return {
        "additional_system_prompt": ("STRING", {"multiline": True, "default": ""}),
        "prompt_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Complete H3 prompt; skips llama.cpp"}),
        "lora_name": (choices("loras", ["ref2v_turbo"], optional=True), {"tooltip": "MiniMax H3 LoRA only. Use qwen_lora_name for the Qwen image edit."}),
        "lora_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05, "tooltip": "Strength of the MiniMax H3 LoRA."}),
        "ref_image_size": (["match", "max"], {"default": "match"}),
        "max_tokens": ("INT", {"default": 3072, "min": 512, "max": 32768}),
        "context_size": ("INT", {"default": 32768, "min": 4096, "max": 262144, "step": 1024}),
        "server_url": ("STRING", {"default": CFG.get("server_url", "http://127.0.0.1:8080")}),
        "unload_llm_after_prompt": ("BOOLEAN", {"default": True}),
        "minimax_prompt_style": (["official", "simple"], {"default": "official", "tooltip": "official: six H3 sections; simple: concise free-form prompt. Qwen keeps its edit prompt."}),
    }


class _Pipeline:
    def __init__(self):
        self._models = {}

    def _load(self, kind, name):
        import folder_paths
        import nodes
        folders = {"model": "diffusion_models", "clip_h3": "text_encoders", "clip_qwen": "text_encoders",
                   "vae": "vae", "patch": "model_patches"}
        path = folder_paths.get_full_path_or_raise(folders[kind], name)
        stamp = (path, os.stat(path).st_mtime_ns, os.stat(path).st_size)
        cached = self._models.get(kind)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        if kind == "model":
            obj = nodes.UNETLoader().load_unet(name, "default")[0]
        elif kind.startswith("clip_"):
            obj = nodes.CLIPLoader().load_clip(name, "minimax" if kind == "clip_h3" else "qwen_image")[0]
        elif kind == "vae":
            obj = nodes.VAELoader().load_vae(name)[0]
        elif kind == "patch":
            from comfy_extras.nodes_model_patch import ModelPatchLoader
            obj = ModelPatchLoader().load_model_patch(name)[0]
        else:
            raise ValueError(f"Unknown model role: {kind}")
        # One cached object per role; replaces changed files instead of accumulating runs.
        self._models[kind] = (stamp, obj)
        return obj

    def _prompt(self, *, ref_image, frames, instruction, llm_model, mmproj, seed,
                video=None, first_frame=None, audio=None, additional_system_prompt="",
                prompt_override="", max_tokens=3072, context_size=32768,
                server_url="http://127.0.0.1:8080", unload_llm_after_prompt=True, minimax_prompt_style="official", **unused):
        if prompt_override.strip():
            prompt = prompt_override.strip()
        else:
            # llama.cpp runs outside ComfyUI's memory manager.
            import comfy.model_management
            comfy.model_management.unload_all_models()
            assets = {"image_1": ref_image}
            if video is not None:
                assets["video_1"] = video
            if first_frame is not None:
                assets["first_frame"] = first_frame
            if audio is not None:
                assets["audio_1"] = audio
            rules = ""
            if video is not None:
                rules = ("VIDEO-ONLY EDIT. Do not write dialogue, vocals, sound effects or music. "
                         "Set overall_soundscape and non_diegetic_music to N/A. Never define <Audio N>. "
                         "Use <Video 1> as the source to edit; preserve its motion, camera, pacing and shots. "
                         "<Picture 1> supplies only the requested target person/clothing. "
                         "The edited first frame supplies the target appearance and source composition.")
            try:
                prompt = prompter.MiniMaxH3R2VPrompter().generate(
                    instruction=instruction, task="video editing" if video is not None else "reference generation",
                    frame_anchor="reference 1 = first frame" if unused.get("ref_image_1_as_first_frame", False) else "none",
                    duration_seconds=frames / 24, thinking="off", length="standard",
                    allow_invented_dialogue=False, max_tokens=max_tokens, seed=int(seed) % (2**32),
                    model=llm_model, mmproj=mmproj, frame_count=frames,
                    additional_system_prompt=additional_system_prompt, extra_rules=rules,
                    context_size=context_size, server_url=server_url, describe_refs=True,
                    prompt_style="simple" if minimax_prompt_style == "simple" else "full (official H3)", **assets)[0]
            finally:
                if unload_llm_after_prompt and llm_model != local_models.SERVER_DEFAULT:
                    managed_server.stop(CFG)
        return apply_policy(prompt, first_frame=bool(unused.get("ref_image_1_as_first_frame", False)),
                            silent=video is not None, style=minimax_prompt_style)

    def _sample(self, model, positive, latent, video_vae, seed, steps, sampler_name, scheduler, silent=False):
        import comfy.samplers
        from comfy_extras import nodes_custom_sampler as cs
        import nodes
        guider = args(cs.BasicGuider.execute(model, positive))[0]
        sigmas = args(cs.BasicScheduler.execute(model, scheduler, steps, 1.0))[0]
        noise = VideoOnlyNoise(seed) if silent else cs.Noise_RandomNoise(seed)
        sampled = args(cs.SamplerCustomAdvanced.execute(
            noise, guider, comfy.samplers.sampler_object(sampler_name), sigmas, latent))[0]
        images = nodes.VAEDecode().decode(video_vae, sampled)[0]
        return images, sampled

    def _base_models(self, h3_model, h3_text_encoder, h3_video_vae, lora_name=NONE, lora_strength=1.0, **unused):
        import nodes
        model = self._load("model", h3_model)
        if lora_name != NONE:
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora_name, lora_strength)[0]
        return model, self._load("clip_h3", h3_text_encoder), self._load("vae", h3_video_vae)


class MiniMaxH3R2VGenerate(_Pipeline):
    CATEGORY = CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "video", "audio", "prompt", "width", "height", "frame_count")
    DESCRIPTION = "Image + audio -> local llama.cpp prompter -> H3 R2V -> sampler -> VAE Decode IMAGE frames, plus VIDEO and AUDIO. All model loaders are internal."

    @classmethod
    def INPUT_TYPES(cls):
        required = {"ref_image": ("IMAGE",), "ref_audio": ("AUDIO",), **common_inputs()}
        required.update({
            "h3_audio_vae": (choices("vae", ["h3_audio"]),),
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.21, "max": 15.08, "step": 0.1}),
            "audio_mode": (["generate from reference", "reuse reference exactly"], {"default": "generate from reference"}),
            "ref_image_1_as_first_frame": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No",
                "tooltip": "Yes: declare Picture 1 as first frame in the prompt and anchor it at frame 0"}),
        })
        return {"required": required, "optional": extra_inputs()}

    def generate(self, ref_image, ref_audio, h3_audio_vae, duration_seconds, audio_mode,
                 ref_image_1_as_first_frame=False, **kw):
        import nodes
        from comfy_extras import nodes_minimax_h3 as h3, nodes_video, nodes_audio
        model, clip, vae = self._base_models(**kw)
        audio_vae = nodes.VAELoader().load_vae(h3_audio_vae)[0]
        frames = geometry.generation_frames(duration_seconds)
        w, h = geometry.canvas(kw["resolution"], kw["aspect_ratio"], ref_image.shape[2], ref_image.shape[1], kw["custom_aspect"])
        instruction = kw["instruction"]
        instruction += ("\nReuse <Audio 1> exactly as the complete target soundtrack." if audio_mode == "reuse reference exactly"
                        else "\nUse <Audio 1> as the audio reference according to the user's instruction.")
        if ref_image_1_as_first_frame:
            instruction += "\n<Picture 1> is the first frame of [Shot 1]; the video begins from <Picture 1>."
        prompt_kw = {**kw, "instruction": instruction, "ref_image_1_as_first_frame": ref_image_1_as_first_frame}
        prompt = self._prompt(ref_image=ref_image[:1], audio=ref_audio, frames=frames, **prompt_kw)
        positive, latent = args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt, width=w, height=h, length=frames,
            ref_image_size=kw.get("ref_image_size", "match"), ref_images={"ref_image_0": ref_image[:1]},
            ref_audios={"ref_audio_0": ref_audio}))[:2]
        if ref_image_1_as_first_frame:
            positive = args(h3.MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=0, vae=vae,
                image=v2v.resize_frames(ref_image[:1], w, h)))[0]
        images, sampled = self._sample(model, positive, latent, vae, kw["seed"], kw["steps"], kw["sampler_name"], kw["scheduler"])
        if audio_mode == "reuse reference exactly":
            audio = {**ref_audio, "waveform": ref_audio["waveform"][..., :round(frames / 24 * ref_audio["sample_rate"])]}
        else:
            audio = args(nodes_audio.VAEDecodeAudio.execute(samples=sampled, vae=audio_vae))[0]
        video = args(nodes_video.CreateVideo.execute(images=images, fps=24.0, audio=audio))[0]
        return images, video, audio, prompt, w, h, frames


class MiniMaxH3V2VGenerate(_Pipeline):
    CATEGORY = CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "qwen_image", "minimax_prompt", "qwen_prompt")
    DESCRIPTION = "Source IMAGE batch (24 FPS) + reference IMAGE -> Qwen Image 2.1 first-frame edit -> Fun ControlNet -> H3 sampler -> VAE Decode IMAGE frames. Also returns the decoded Qwen edit before H3 resizing as qwen_image and both enhanced prompts. Silent IMAGE outputs. Strict video-only is experimental; native AV mode computes audio latents internally and discards them."


    def _enhance_edit(self, source, reference, instruction, frames, **kw):
        """One user instruction -> one vision enhancement response containing both prompts."""
        import comfy.model_management
        from .h3_prompter import media, llama_client as lc, prompts

        model_path, mmproj_path = local_models.resolve(kw["llm_model"], kw["mmproj"], CFG)
        if model_path and mmproj_path is None:
            raise ValueError("V2V prompt enhancement needs a vision GGUF and its matching mmproj.")
        comfy.model_management.unload_all_models()
        if model_path:
            server_url = managed_server.ensure(model_path, mmproj_path, kw.get("context_size", 32768), CFG)
            alias = os.path.splitext(os.path.basename(model_path))[0]
        else:
            server_url = (kw.get("server_url") or CFG["server_url"]).strip()
            lc.ensure_server(server_url, CFG)
            alias = CFG.get("model_alias", "qwen3.8-27b")

        style = kw.get("minimax_prompt_style", "official")
        if style not in ("official", "simple"):
            raise ValueError(f"Unknown MiniMax prompt style: {style}")
        # Editing rules adapted from Qwen's official system_prompt_edit.txt.
        system = (
            "Prepare coordinated prompts for a silent video edit from ONE user instruction. "
            "Return a JSON object with exactly two fields: qwen_prompt and minimax_prompt. "
            "For qwen_prompt use <image1> for the source first frame (the canvas), and <image2> "
            "for the supplied appearance reference. State the requested operation clearly, "
            "identify each image's role, and preserve attributes the user did not request to edit. "
            "Base details on visible evidence; do not invent garment colors, patterns or identity. "
            "For a clothing change transfer clothing only, preserving the source person's face, "
            "body, pose, accessories not targeted by the request, camera, light and background. "
            "For a person change preserve source pose, composition and background while transferring "
            "the requested identity. Write English prose unless the request is Chinese. "
            "Do not include video motion instructions in qwen_prompt. "
            "For minimax_prompt follow the H3 format below. Its asset labels DIFFER from Qwen: "
            "<Video 1> is the source performance; <Picture 1> will be the Qwen-edited image. "
            "The supplied raw appearance reference is used by Qwen only, not supplied directly to H3. "
            "Picture 1 has not been generated yet: describe its intended edited appearance, not invented observations. "
            "Picture 1 is an appearance reference, not a pinned first-frame guide. Do not mention Picture 2. "
            "The target appearance must exist from frame 0 throughout the clip, not transform gradually. "
            "Preserve source timing, actions, camera and shots. Do not invent motion or sound. "
            "Use [video editing] and describe silent video only. "
            "Do not use <Audio N> or Qwen <imageN> labels in minimax_prompt.\n\n"
            + ("H3 formatting rules (apply ONLY to minimax_prompt):\n" + prompts.SYSTEM_PROMPT_R2V
               if style == "official" else
               "For minimax_prompt write 2-4 concise English sentences describing the requested edit, "
               "the roles of <Video 1> and the Qwen-edited <Picture 1>, and preserved motion/camera/timing. "
               "No pinned frame, section headings, retention analysis or shot list.")
        )
        extra = str(kw.get("additional_system_prompt", "")).strip()
        if extra:
            system += "\n\nAdditional user preferences:\n" + extra
        if style == "official":
            system += (
                "\n\nResponse envelope: output only JSON with qwen_prompt (a string) and "
                "minimax_prompt (an object with six non-empty string fields: "
                + ", ".join(prompts.R2V_FIELDS) + "). "
                "Set overall_soundscape and non_diegetic_music to N/A. "
                "qwen_prompt uses both <image1> and <image2>. No Markdown fences.")
        else:
            system += (
                "\n\nResponse envelope: output only JSON with two strings: qwen_prompt and minimax_prompt. "
                "minimax_prompt is concise prose, without the six official section headings. "
                "qwen_prompt uses both <image1> and <image2>. No Markdown fences.")
        parts = [{"type": "text", "text": (
            f"User instruction: {instruction}\nEdit mode: {kw['edit_mode']}\n"
            f"Video: {frames} frames at 24 FPS ({frames / 24:.3f} seconds). "
            "The word 'this' refers to the supplied reference <image2>. "
            "The source frame supplies the person and composition to edit."
        )}]
        # Source motion context is labeled separately from the two Qwen image slots.
        for i in media.sample_indices(len(source), 8):
            parts.extend([
                {"type": "text", "text": f"<Video 1> source motion frame at {i / 24:.3f} seconds:"},
                {"type": "image_url", "image_url": {"url": media.pil_to_data_url(
                    media.image_batch_to_pil(source[i:i + 1])[0], 512)}},
            ])
        for label, tensor in (("<image1>: source first frame, canvas for Qwen.", source[:1]),
                              ("<image2>: target appearance for Qwen; H3 receives only the resulting edit as <Picture 1>.", reference)):
            parts.extend([
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": media.pil_to_data_url(
                    media.image_batch_to_pil(tensor)[0], 768)}},
            ])
        base = dict(model=alias, max_tokens=int(kw.get("max_tokens", 3072)),
                    seed=int(kw["seed"]) % (2**32), temperature=0.2, top_p=0.8,
                    top_k=20, cache_prompt=True, response_format={"type": "json_object"})
        try:
            content, _, _ = prompter.MiniMaxH3R2VPrompter._run(
                server_url, base, {"role": "system", "content": system},
                {"role": "user", "content": parts}, "off",
                float(CFG.get("request_timeout_seconds", 600)), False, None, prefill=None)
        finally:
            if kw.get("unload_llm_after_prompt", True) and model_path:
                managed_server.stop(CFG)
        qwen, minimax = parse_edit_response(content, style=style)
        lc.log("One instruction enhanced into Qwen and MiniMax prompts.")
        return qwen.strip(), minimax

    @classmethod
    def INPUT_TYPES(cls):
        required = {"source_video": ("IMAGE", {"tooltip": "Source video frames as an IMAGE batch at 24 FPS, matching the native H3 ref_video input. Connect a video loader IMAGE output."}), "ref_image": ("IMAGE",), **common_inputs()}
        required.update({
            "fun_controlnet": (choices("model_patches", ["minimax_h3_fun"]),),
            "qwen_model": (choices("diffusion_models", ["qwen_image_2.1", "qwen_image21", "qwen_image_21"]),),
            "qwen_text_encoder": (choices("text_encoders", ["qwen3vl_8b", "qwen3_vl_8b", "qwen_image_2.1", "qwen3_vl", "qwen3vl"]), {"tooltip": "Qwen Image 2.1 CLIP/text encoder loaded internally from models/text_encoders."}),
            "qwen_vae": (choices("vae", ["qwen_image_2.1", "qwen_image21", "qwen_image"]), {"tooltip": "Qwen Image 2.1 VAE loaded internally from models/vae; decodes qwen_image."}),
            "edit_mode": (["change clothes", "change person", "custom"], {"default": "change clothes"}),
        })
        import comfy.samplers
        required["scheduler"] = (list(comfy.samplers.SCHEDULER_NAMES), {"default": "simple"})
        optional = extra_inputs()
        optional.pop("prompt_override", None)
        optional["ref_image_size"] = (["match", "max"], {"default": "max"})
        optional.update({
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0, "max": 3600}),
            "max_seconds": ("FLOAT", {"default": 15.08, "min": 0.21, "max": 15.08}),
            "control_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 3, "step": 0.05}),
            "qwen_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "qwen_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
            "qwen_lora_name": (choices("loras", ["qwen"], optional=True), {"tooltip": "LoRA for the Qwen Image 2.1 diffusion model only, loaded from models/loras. Select (none) to disable."}),
            "qwen_lora_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05, "tooltip": "Strength of the Qwen image-edit LoRA; 0 disables it."}),
            "h3_sampling_mode": (["strict video-only (experimental)", "native AV (discard audio)"], {"default": "strict video-only (experimental)", "tooltip": "Native AV matches the reference sampler but internally denoises audio latents; audio is never decoded or returned. Strict removes audio tokens and can change visual quality."}),
        })
        return {"required": required, "optional": optional}

    def generate(self, source_video, ref_image, fun_controlnet, qwen_model,
                 qwen_text_encoder, qwen_vae, edit_mode, start_seconds=0.0, max_seconds=15.08,
                 control_strength=1.0, qwen_steps=25, qwen_resolution=1024,
                 qwen_lora_name=NONE, qwen_lora_strength=1.0,
                 h3_sampling_mode="strict video-only (experimental)", **kw):
        import torch
        import nodes
        from comfy_extras import nodes_minimax_h3 as h3
        from .nodes_h3qwen import MiniMaxH3QwenKeyframeEdit

        if source_video.ndim != 4 or source_video.shape[-1] < 3 or len(source_video) == 0:
            raise ValueError("source_video must be a non-empty IMAGE batch [frames, height, width, RGB] at 24 FPS")
        if h3_sampling_mode not in ("strict video-only (experimental)", "native AV (discard audio)"):
            raise ValueError(f"Unknown H3 sampling mode: {h3_sampling_mode}")
        strict_video_only = h3_sampling_mode == "strict video-only (experimental)"
        original = source_video[..., :3]
        # IMAGE batches carry no timing metadata. Match native H3 ref_video's 24 FPS contract.
        fps = 24.0
        frames, indices = geometry.video_timeline(len(original), fps, start_seconds, max_seconds)
        w, h = geometry.canvas(kw["resolution"], kw["aspect_ratio"], original.shape[2], original.shape[1], kw["custom_aspect"])
        index_tensor = torch.tensor(indices, device=original.device, dtype=torch.long)
        source = v2v.resize_frames(original[index_tensor], w, h)

        instruction = kw["instruction"].strip()
        if not instruction:
            instruction = ("Replace only the source person's clothing with the outfit in the reference image; preserve their identity."
                           if edit_mode == "change clothes" else
                           "Replace the source person with the person in the reference image; preserve the source performance."
                           if edit_mode == "change person" else "")
        if not instruction:
            raise ValueError("Write an edit instruction for custom edit mode")
        qwen_prompt, prompt = self._enhance_edit(source, ref_image[:1], frames=frames, **{**kw, "instruction": instruction, "edit_mode": edit_mode})
        # Qwen always edits the source first frame using the connected reference.
        qloader = _Pipeline()
        qm = qloader._load("model", qwen_model)
        if qwen_lora_name != NONE and qwen_lora_strength != 0:
            qm = nodes.LoraLoaderModelOnly().load_lora_model_only(
                qm, qwen_lora_name, qwen_lora_strength)[0]
        qc = qloader._load("clip_qwen", qwen_text_encoder)
        qv = qloader._load("vae", qwen_vae)
        edit_text = qwen_prompt
        qwen_images = [source[:1], ref_image[:1]]
        qwen_image = MiniMaxH3QwenKeyframeEdit._qwen_edit(qm, qc, qv, edit_text,
            qwen_images, kw["seed"], qwen_steps, 1.0, "euler", "simple", qwen_resolution)
        del qm, qc, qv, qloader

        model, clip, vae = self._base_models(**kw)
        positive, latent = args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=None, prompt=prompt, width=w, height=h, length=frames,
            ref_image_size=kw.get("ref_image_size", "max"),
            ref_images={"ref_image_0": qwen_image},
            ref_videos={"ref_video_0": source}, ref_video_audios=None, ref_audios=None))[:2]
        patch = self._load("patch", fun_controlnet)
        model = args(h3.MiniMaxH3FunControlNetApply.execute(
            model=model, model_patch=patch, vae=vae, strength=control_strength,
            start_percent=0.0, end_percent=1.0, control_video=source))[0]
        from .h3_prompter import llama_client as lc
        lc.log(f"V2V sampling: {h3_sampling_mode}; scheduler={kw['scheduler']}; steps={kw['steps']}; Qwen steps={qwen_steps}")
        if strict_video_only:
            model = patch_video_only(model)
        images, _ = self._sample(model, positive, latent, vae, kw["seed"], kw["steps"], kw["sampler_name"], kw["scheduler"], silent=strict_video_only)
        images = images[:frames]
        return images, qwen_image, prompt, qwen_prompt


NODE_CLASS_MAPPINGS = {"MiniMaxH3R2VGenerate": MiniMaxH3R2VGenerate, "MiniMaxH3V2VGenerate": MiniMaxH3V2VGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3R2VGenerate": "MiniMax H3 R2V Generate (Sample + VAE Decode)",
                              "MiniMaxH3V2VGenerate": "MiniMax H3 V2V Generate (Sample + VAE Decode, Silent)"}
