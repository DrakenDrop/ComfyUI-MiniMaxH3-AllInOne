"""Normalize enhanced prompts and apply first-frame/audio policies."""
import json
import re

FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")


def strip_fence(text):
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence) and text.endswith(fence):
        text = re.sub("^" + fence + r"[^\n]*\n", "", text)
        text = text.rsplit(fence, 1)[0].strip()
    return text


def _sections(prompt):
    if isinstance(prompt, str):
        prompt = strip_fence(prompt)
        if prompt.startswith("{"):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                pass
    if isinstance(prompt, dict):
        sections = {str(k).strip().lower().replace(" ", "_"): v for k, v in prompt.items()}
        if set(sections) != set(FIELDS) or not all(isinstance(v, str) and v.strip() for v in sections.values()):
            raise ValueError("Official MiniMax prompt must contain six non-empty text sections: " + ", ".join(FIELDS))
        return {k: sections[k].strip() for k in FIELDS}
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("MiniMax prompt is empty or is not text.")
    if "\n" not in prompt and "\\n" in prompt:
        prompt = prompt.replace("\\n", "\n")
    alternatives = "|".join(f.replace("_", r"[_ ]+") for f in FIELDS)
    heading = re.compile(r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?(" + alternatives +
                         r")(?:\*\*|__)?[ \t]*(?::[ \t]*(?:\*\*|__)?[ \t]*|[ \t]*(?=\n|$))")
    matches = list(heading.finditer(prompt))
    names = [re.sub(r" +", "_", m.group(1).lower()) for m in matches]
    if len(names) != len(FIELDS) or set(names) != set(FIELDS):
        found = ", ".join(names) or "none"
        missing = ", ".join(f for f in FIELDS if f not in names) or "none (duplicate headings)"
        raise ValueError(
            f"Official MiniMax prompt format is incomplete. Found: {found}. Missing: {missing}. "
            "Choose minimax_prompt_style=simple for free-form text, or use official for six sections. "
            "A larger max_tokens helps only if the response was truncated.")
    sections = {name: prompt[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(prompt)].strip()
                for i, (name, m) in enumerate(zip(names, matches))}
    if any(not value for value in sections.values()):
        raise ValueError("Official MiniMax prompt contains an empty section.")
    return sections


def apply_policy(prompt, first_frame=False, silent=False, style="official"):
    if style not in ("simple", "official"):
        raise ValueError(f"Unknown MiniMax prompt style: {style}")
    if style == "simple":
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Simple MiniMax prompt must be non-empty text.")
        text = strip_fence(prompt)
        if silent:
            if re.search(r"<Audio\s+\d+>|<d>", text, re.I):
                raise ValueError("Video-only prompt contains audio/dialogue tags.")
            if "Silent video; no dialogue, sound effects or music." not in text:
                text += "\nSilent video; no dialogue, sound effects or music."
        if first_frame and "The video begins from <Picture 1> as its first frame." not in text:
            text += "\nThe video begins from <Picture 1> as its first frame."
        return text

    sections = _sections(prompt)
    if silent:
        if re.search(r"<Audio\s+\d+>|<d>", "\n".join(sections.values()), re.I):
            raise ValueError("Video-only prompt contains audio/dialogue tags; remove them from system instructions or override")
        sections["overall_soundscape"] = "N/A"
        sections["non_diegetic_music"] = "N/A"
    if first_frame:
        definition = "<Picture 1> is the first frame of [Shot 1], defining its opening composition and appearance."
        retention = "<Picture 1> ([Shot 1] first frame): fully_preserved - the shot begins from this reference image."
        for field, line in (("subject_definitions", definition), ("retention_analysis", retention)):
            if re.search(r"(?m)^<Picture 1>.*$", sections[field]):
                sections[field] = re.sub(r"(?m)^<Picture 1>.*$", line, sections[field])
            else:
                sections[field] += "\n" + line
        if "keyframe completion" not in sections["summary"]:
            sections["summary"] = re.sub(r"^\[([^\]]+)\]", r"[keyframe completion + \1]", sections["summary"], count=1)
        if "begins from <Picture 1>" not in sections["detailed_description"]:
            sections["detailed_description"] = sections["detailed_description"].replace(
                "[Shot 1]", "[Shot 1] The shot begins from <Picture 1>.", 1)
    return "\n\n".join(f"{f}:\n{sections[f]}" for f in FIELDS)


def parse_edit_response(content, style="official"):
    try:
        result = json.loads(strip_fence(content))
    except (TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Enhancer did not return complete JSON. Check its response/log for truncation or formatting errors.") from exc
    if not isinstance(result, dict) or set(result) != {"qwen_prompt", "minimax_prompt"}:
        raise ValueError("Enhancer must return qwen_prompt and minimax_prompt.")
    qwen = result["qwen_prompt"]
    if not isinstance(qwen, str) or not qwen.strip():
        raise ValueError("Qwen prompt must be non-empty text.")
    if "<image1>" not in qwen or "<image2>" not in qwen:
        raise ValueError("Qwen prompt must reference <image1> (source) and <image2> (target).")
    minimax = apply_policy(result["minimax_prompt"], silent=True, style=style)
    if re.search(r"<image\s*\d+>", minimax, re.I):
        raise ValueError("MiniMax prompt contains Qwen image labels.")
    if not all(tag in minimax for tag in ("<Video 1>", "<Picture 1>", "<Picture 2>")):
        raise ValueError("MiniMax prompt must reference source video, appearance reference, and edited first frame.")
    return qwen.strip(), minimax
