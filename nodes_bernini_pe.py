"""BerniniR prompt enhancer on the local llama-server (no OpenAI key, no vLLM).

Same behaviour as Bernini's official `--use_pe` (bernini/prompt_enhancer.py): the task-specific template,
3 uniformly sampled source frames and the reference images go to a vision chat model; r2v / r2i / rv2v ask for
JSON {"rewritten_text": ...}. Here the chat model is the local Qwen GGUF + mmproj served by llama-server.
Output -> `prompt` of BerniniR · Text Encode (use the same task_type there).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

from . import nodes as prompter_nodes
from .h3_prompter import bernini_pe_templates as T
from .h3_prompter import llama_client as lc
from .h3_prompter import local_models
from .h3_prompter import managed_server
from .h3_prompter import media

CATEGORY = "BerniniR/Long video"
_CFG = prompter_nodes._CFG
TASKS = ["v2v", "rv2v", "mv2v", "i2i", "t2v", "t2i", "i2v", "r2v", "r2i", "vi2v", "vrc2v", "ads2v"]


def _frames(video, n: int):
    if video is None or video.shape[0] == 0 or n <= 0:
        return []
    total = video.shape[0]
    idx = [total // 2] if n == 1 else [round(i * (total - 1) / (n - 1)) for i in range(n)]
    return [media.image_batch_to_pil(video[i:i + 1])[0] for i in idx]


def _refs(images):
    if images is None:
        return []
    return media.image_batch_to_pil(images)


def build_request(task, prompt, frames, refs):
    """(system, user_text, images, json_mode) exactly like PromptEnhancer.__call__."""
    base = T.SYSTEM_PROMPTS["default"]
    n_ref = len(refs)
    if task == "t2v":
        return T.T2V_A14B_EN_SYS_PROMPT, prompt, [], False
    if task == "t2i":
        return T.T2I_A14B_EN_SYS_PROMPT, prompt, [], False
    if task in ("v2v", "mv2v"):
        return base, T.V2V_TEMPLATE.format(user_prompt=prompt), frames, False
    if task == "i2i":
        return base, T.I2I_TEMPLATE.format(user_prompt=prompt), refs or frames[:1], False
    if task == "i2v":
        imgs = refs if refs else frames[:1]
        return base, T.I2V_TEMPLATE.format(user_prompt=prompt, image_num=len(imgs)), imgs, False
    if task == "ads2v":
        return base, T.ADS2V_TEMPLATE.format(user_prompt=prompt), frames, False
    if task == "vi2v":
        return base, T.VI2V_TEMPLATE.format(user_prompt=prompt, image_num=n_ref), frames + refs, False
    if task == "r2v":
        return base, T.R2V_TEMPLATE.format(image_num=max(n_ref, 1), original_text=prompt), refs, True
    if task == "r2i":
        return base, T.R2I_TEMPLATE.format(image_num=max(n_ref, 1), original_text=prompt), refs, True
    if task in ("rv2v", "vrc2v"):
        return base, T.VR2V_TEMPLATE.format(image_num=max(n_ref, 1), original_text=prompt), frames + refs, True
    return base, prompt, [], False


def _extract(text: str, json_mode: bool) -> str:
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", (text or "").strip()).strip()
    if json_mode:
        try:
            return str(json.loads(t).get("rewritten_text", t)).strip()
        except Exception:  # noqa: BLE001
            m = re.search(r'"rewritten_text"\s*:\s*"((?:[^"\\]|\\.)*)"', t, re.S)
            if m:
                return json.loads(f'"{m.group(1)}"').strip()
    t = re.sub(r"^(here is|here's)[^\n]*\n", "", t, flags=re.I).strip()
    return t.strip().strip('"').strip()


class BerniniRPromptEnhancer:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    DESCRIPTION = ("Bernini's official prompt enhancer (task templates, 3 source frames + reference images) on the "
                   "local llama-server instead of GPT/OpenAI. Connect `prompt` to BerniniR · Text Encode with the same "
                   "task_type.")

    _cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Short edit instruction, any language (e.g. 'ganti dress-nya "
                                                      "jadi hitam')."}),
                "task_type": (TASKS, {"default": "v2v"}),
                "llm_model": (local_models.model_choices(_CFG), {}),
                "mmproj": (local_models.mmproj_choices(_CFG), {"default": local_models.MMPROJ_AUTO}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "source_video": ("IMAGE", {"tooltip": "Source frames (3 are sampled uniformly, like Bernini)."}),
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s) = image0, image1, ..."}),
                "video_frames": ("INT", {"default": 3, "min": 1, "max": 16,
                                         "tooltip": "Frames sampled from the source (Bernini default 3)."}),
                "image_max_side": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 64}),
                "context_size": ("INT", {"default": int(_CFG.get("context_size", 32768)), "min": 4096,
                                         "max": 262144, "step": 1024}),
                "unload_llm_after": ("BOOLEAN", {"default": False,
                                                 "tooltip": "Stop llama-server afterwards (frees VRAM for Bernini)."}),
                "print_to_console": ("BOOLEAN", {"default": True}),
            },
        }

    def run(self, instruction, task_type, llm_model, mmproj, seed, source_video=None, reference_images=None,
            video_frames=3, image_max_side=768, max_tokens=1024, context_size=32768, unload_llm_after=False,
            print_to_console=True):
        prompt = (instruction or "").strip()
        if not prompt:
            raise ValueError("Write an instruction to enhance.")
        frames = _frames(source_video, int(video_frames))
        refs = _refs(reference_images)
        system, user_text, imgs, json_mode = build_request(task_type, prompt, frames, refs)

        model_path, mmproj_path = local_models.resolve(llm_model, mmproj, _CFG)
        if model_path:
            if imgs and mmproj_path is None and mmproj != local_models.MMPROJ_NONE:
                raise RuntimeError(f"mmproj 'auto' found no vision file for {llm_model}: the LLM could not see the "
                                   "frames. Pick the mmproj manually.")
            server_url = managed_server.ensure(model_path, mmproj_path, context_size, _CFG)
            alias = os.path.splitext(os.path.basename(model_path))[0]
            if mmproj_path is None:
                imgs = []
        else:
            server_url = _CFG["server_url"]
            lc.ensure_server(server_url, _CFG)
            alias = _CFG.get("model_alias", "qwen3.8-27b")

        key = (task_type, prompt, alias, seed, json_mode, tuple(hashlib.md5(i.tobytes()).hexdigest() for i in imgs))
        if key in self._cache:
            lc.log("BerniniR PE: inputs unchanged -> cached prompt.")
            out = self._cache[key]
        else:
            content = [{"type": "text", "text": user_text}]
            for i, im in enumerate(imgs):  # same labelling as Bernini's _build_messages
                content.append({"type": "text", "text": f"\n[Image {i}]:"})
                content.append({"type": "image_url", "image_url": {"url": media.pil_to_data_url(im, image_max_side)}})
            base = {"model": alias, "max_tokens": int(max_tokens), "seed": int(seed), "cache_prompt": True,
                    **prompter_nodes.SAMPLING["off"]}
            if json_mode:
                base["response_format"] = {"type": "json_object"}
            lc.log(f"BerniniR PE: task {task_type}, {len(frames)} frame(s) + {len(refs)} reference(s)"
                   + (" (json)" if json_mode else ""))
            t0 = time.time()
            text, _, _ = prompter_nodes.MiniMaxH3R2VPrompter._run(
                server_url, base, {"role": "system", "content": system}, {"role": "user", "content": content},
                "off", float(_CFG.get("request_timeout_seconds", 600)), print_to_console, None, prefill=None)
            out = _extract(text, json_mode) or prompt
            lc.log(f"BerniniR PE: done in {time.time() - t0:.1f}s")
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = out
        if unload_llm_after and model_path:
            managed_server.stop(_CFG)
        return (out,)


NODE_CLASS_MAPPINGS = {"BerniniRPromptEnhancer": BerniniRPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"BerniniRPromptEnhancer": "BerniniR · Prompt Enhancer (local LLM, official templates)"}

