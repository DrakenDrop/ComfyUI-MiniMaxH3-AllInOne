"""Self-contained H3 R2V / V2V nodes: load, prompt, condition, sample, decode."""
from __future__ import annotations

import os
from . import nodes as prompter
from .h3_prompter import canvas as geometry, local_models, managed_server, v2v
from .h3_prompter.video_only import patch_video_only, VideoOnlyNoise
from .h3_prompter.prompt_format import apply_policy

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
        "h3_text_encoder": (choices("text_encoders", ["minimax", "h3"]),),
        "h3_video_vae": (choices("vae", ["h3_video", "lynnreal"]),),
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
        "lora_name": (choices("loras", ["ref2v_turbo"], optional=True),),
        "lora_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 2, "step": 0.05}),
        "ref_image_size": (["match", "max"], {"default": "match"}),
        "max_tokens": ("INT", {"default": 3072, "min": 512, "max": 32768}),
        "context_size": ("INT", {"default": 32768, "min": 4096, "max": 262144, "step": 1024}),
        "server_url": ("STRING", {"default": CFG.get("server_url", "http://127.0.0.1:8080")}),
        "unload_llm_after_prompt": ("BOOLEAN", {"default": True}),
    }


class _Pipeline:
    def __init__(self):
        self._models = {}

    def _load(self, kind, name):
        import folder_paths
        import nodes
        folders = {"model": "diffusion_models", "clip_h3": "text_encoders", "clip_qwen": "text_encoders",
                   "vae": "vae", "patch": "model_patches", "pose": "checkpoints"}
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
            obj = nodes.CheckpointLoaderSimple().load_checkpoint(name)
        # One cached object per role; replaces changed files instead of accumulating runs.
        self._models[kind] = (stamp, obj)
        return obj

    def _prompt(self, *, ref_image, frames, instruction, llm_model, mmproj, seed,
                video=None, first_frame=None, audio=None, additional_system_prompt="",
                prompt_override="", max_tokens=3072, context_size=32768,
                server_url="http://127.0.0.1:8080", unload_llm_after_prompt=True, **unused):
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
                    prompt_style="full (official H3)", **assets)[0]
            finally:
                if unload_llm_after_prompt and llm_model != local_models.SERVER_DEFAULT:
                    managed_server.stop(CFG)
        return apply_policy(prompt, first_frame=bool(unused.get("ref_image_1_as_first_frame", False)),
                            silent=video is not None)

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
    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING", "IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "video", "prompt", "edited_reference", "control_pose", "width", "height", "frame_count")
    DESCRIPTION = "Source VIDEO + reference IMAGE -> Qwen Image 2.1 first-frame edit -> pose/Fun ControlNet -> H3 sampler -> VAE Decode IMAGE frames and silent VIDEO. Video-only denoising is experimental."

    @classmethod
    def INPUT_TYPES(cls):
        required = {"source_video": ("VIDEO",), "ref_image": ("IMAGE",), **common_inputs()}
        required.update({
            "fun_controlnet": (choices("model_patches", ["minimax_h3_fun"]),),
            "pose_checkpoint": (choices("checkpoints", ["sdpose"]),),
            "qwen_model": (choices("diffusion_models", ["qwen_image_2.1", "qwen_image21", "qwen_image_21"]),),
            "qwen_text_encoder": (choices("text_encoders", ["qwen3vl_8b", "qwen3_vl_8b", "qwen_image_2.1", "qwen3_vl", "qwen3vl"]),),
            "qwen_vae": (choices("vae", ["qwen_image_2.1", "qwen_image21", "qwen_image"]),),
            "edit_mode": (["change clothes", "change person", "custom"], {"default": "change clothes"}),
        })
        optional = extra_inputs()
        optional.update({
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0, "max": 3600}),
            "max_seconds": ("FLOAT", {"default": 15.08, "min": 0.21, "max": 15.08}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0, "max": 3, "step": 0.05}),
            "qwen_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "qwen_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
            "qwen_prompt": ("STRING", {"multiline": True, "default": ""}),
        })
        return {"required": required, "optional": optional}

    def generate(self, source_video, ref_image, fun_controlnet, pose_checkpoint, qwen_model,
                 qwen_text_encoder, qwen_vae, edit_mode, start_seconds=0.0, max_seconds=15.08,
                 pose_strength=1.0, qwen_steps=25, qwen_resolution=1024, qwen_prompt="",
                 **kw):
        import torch
        import nodes
        from comfy_extras import nodes_minimax_h3 as h3, nodes_video, nodes_sdpose
        from .nodes_h3qwen import MiniMaxH3QwenKeyframeEdit

        components = source_video.get_components()
        original = components.images[..., :3]
        fps = float(components.frame_rate)
        frames, indices = geometry.video_timeline(len(original), fps, start_seconds, max_seconds)
        w, h = geometry.canvas(kw["resolution"], kw["aspect_ratio"], original.shape[2], original.shape[1], kw["custom_aspect"])
        index_tensor = torch.tensor(indices, device=original.device, dtype=torch.long)
        source = v2v.resize_frames(original[index_tensor], w, h)

        pose_model, _, pose_vae = self._load("pose", pose_checkpoint)
        keypoints = args(nodes_sdpose.SDPoseKeypointExtractor.execute(
            model=pose_model, vae=pose_vae, image=source, batch_size=4))[0]
        pose = args(nodes_sdpose.SDPoseDrawKeypoints.execute(
            keypoints=keypoints, draw_body=True, draw_hands=True, draw_face=True, draw_feet=True,
            stick_width=4, face_point_size=3, score_threshold=0.3, draw_head=True))[0]
        if len(pose) != frames:
            raise ValueError("Pose extraction returned a different number of frames")

        instruction = kw["instruction"].strip()
        if not instruction:
            instruction = ("Replace only the source person's clothing with the outfit in the reference image; preserve their identity."
                           if edit_mode == "change clothes" else
                           "Replace the source person with the person in the reference image; preserve the source performance."
                           if edit_mode == "change person" else "")
        if not instruction:
            raise ValueError("Write an edit instruction for custom edit mode")
        # Qwen always edits the source first frame using the connected reference.
        qloader = _Pipeline()
        qm = qloader._load("model", qwen_model)
        qc = qloader._load("clip_qwen", qwen_text_encoder)
        qv = qloader._load("vae", qwen_vae)
        edit_text = qwen_prompt.strip() or (instruction +
            " Image 1 is the source frame and defines pose, camera and background. "
            "Image 2 is the target reference. "
            "Change only the requested person or outfit. Keep the pose, framing and background of Image 1.")
        qwen_images = [source[:1], ref_image[:1]]
        first = MiniMaxH3QwenKeyframeEdit._qwen_edit(qm, qc, qv, edit_text,
            qwen_images, kw["seed"], qwen_steps, 1.0, "euler", "simple", qwen_resolution)
        del qm, qc, qv, qloader
        first = v2v.resize_frames(first, w, h)

        prompt = self._prompt(ref_image=ref_image[:1], frames=frames,
                              video=source, first_frame=first, **{**kw, "instruction": instruction})
        model, clip, vae = self._base_models(**kw)
        positive, latent = args(h3.MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=None, prompt=prompt, width=w, height=h, length=frames,
            ref_image_size=kw.get("ref_image_size", "match"),
            ref_images={"ref_image_0": ref_image[:1], "ref_image_1": first},
            ref_videos={"ref_video_0": source}, ref_video_audios=None, ref_audios=None))[:2]
        positive = args(h3.MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=0, vae=vae, image=first))[0]
        patch = self._load("patch", fun_controlnet)
        model = args(h3.MiniMaxH3FunControlNetApply.execute(
            model=model, model_patch=patch, vae=vae, strength=pose_strength,
            start_percent=0.0, end_percent=1.0, control_video=pose, source_video=source))[0]
        model = patch_video_only(model)
        images, _ = self._sample(model, positive, latent, vae, kw["seed"], kw["steps"], kw["sampler_name"], kw["scheduler"], silent=True)
        images = images[:frames]
        video = args(nodes_video.CreateVideo.execute(images=images, fps=24.0, audio=None))[0]
        return images, video, prompt, first, pose, w, h, frames


NODE_CLASS_MAPPINGS = {"MiniMaxH3R2VGenerate": MiniMaxH3R2VGenerate, "MiniMaxH3V2VGenerate": MiniMaxH3V2VGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3R2VGenerate": "MiniMax H3 R2V Generate (Sample + VAE Decode)",
                              "MiniMaxH3V2VGenerate": "MiniMax H3 V2V Generate (Sample + VAE Decode, Silent)"}
