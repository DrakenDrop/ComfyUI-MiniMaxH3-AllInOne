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


class V2VPromptError(ValueError):
    """A valid Qwen response contains a MiniMax prompt that needs correction."""


def _normalize_h3_labels(text):
    # Only canonicalize explicit H3 tags. Never infer missing assets or map Qwen labels.
    return re.sub(r"<\s*(picture|video|subject)\s*(\d+)\s*>",
                  lambda m: f"<{m[1].title()} {int(m[2])}>", text, flags=re.I)


def validate_qwen_prompt(qwen, has_reference=True):
    if not isinstance(qwen, str) or not qwen.strip():
        raise ValueError("Qwen prompt must be non-empty text.")
    required = ("<image1>", "<image2>") if has_reference else ("<image1>",)
    if not all(label in qwen for label in required):
        raise ValueError("Qwen prompt must reference " + " and ".join(required) + ".")
    allowed = {"1", "2"} if has_reference else {"1"}
    if any(label not in allowed for label in re.findall(r"<image\s*(\d+)>", qwen, re.I)):
        raise ValueError("Qwen prompt references an image that is not connected.")
    return qwen.strip()


def parse_prompt_field(content, field):
    """Read a single-stage enhancer response without accepting extra prompt fields."""
    try:
        result = json.loads(strip_fence(content))
    except (TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} enhancer did not return complete JSON; check max_tokens and formatting.") from exc
    if not isinstance(result, dict) or set(result) != {field}:
        raise ValueError(f"Enhancer must return exactly one field: {field}.")
    return result[field]


def parse_edit_response(content, style="official", has_reference=True, has_video_reference=True):
    try:
        result = json.loads(strip_fence(content))
    except (TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("Enhancer did not return complete JSON. Check its response/log for truncation or formatting errors.") from exc
    if not isinstance(result, dict) or set(result) != {"qwen_prompt", "minimax_prompt"}:
        raise ValueError("Enhancer must return qwen_prompt and minimax_prompt.")
    qwen = validate_qwen_prompt(result["qwen_prompt"], has_reference)
    try:
        minimax = apply_policy(result["minimax_prompt"], silent=True, style=style)
    except ValueError as exc:
        raise V2VPromptError(str(exc)) from exc
    minimax = _normalize_h3_labels(minimax)
    if re.search(r"<image\s*\d+>", minimax, re.I):
        raise V2VPromptError("MiniMax prompt contains Qwen image labels. Use only the connected H3 asset labels.")
    required_h3 = ("<Video 1>", "<Picture 1>") if has_video_reference else ("<Picture 1>",)
    missing = [tag for tag in required_h3 if tag not in minimax]
    if missing:
        raise V2VPromptError(
            "MiniMax prompt is missing " + ", ".join(missing) + ". "
            + ("<Video 1> is the source video. " if has_video_reference else
               "H3 has no reference video; motion comes from skeleton control. ") +
            "<Picture 1> is the Qwen-edited appearance reference, including when no optional ref_image is connected. Describe its role.")
    if any(n != "1" for n in re.findall(r"<Picture\s+(\d+)>", minimax, re.I)):
        raise V2VPromptError("MiniMax receives only one picture: the Qwen-edited <Picture 1>.")
    if any(n != "1" for n in re.findall(r"<Video\s+(\d+)>", minimax, re.I)):
        raise V2VPromptError("MiniMax receives only one video: source <Video 1>.")
    if not has_video_reference and (re.search(r"<Video\s+\d+>", minimax, re.I) or "[video editing]" in minimax.lower()):
        raise V2VPromptError("Pose-only H3 receives no <Video N> asset. Use <Picture 1>, skeleton control and [reference generation].")
    _check_v2v_anchor(minimax)
    return qwen.strip(), minimax


class V2VAnchorError(V2VPromptError):
    """An enhancer assigned a frame anchor which this V2V pipeline does not supply."""


def _check_v2v_anchor(prompt):
    patterns = (
        r"\b(?:begins?|starts?|ends?|opens?|closes?)\s+(?:exactly\s+)?(?:from|on|with|at)\s+<Picture\s+1>",
        r"<Picture\s+1>[^.!?\n]{0,100}\b(?:first[ -]frame|last[ -]frame|keyframe|frame[ -]0)\b",
        r"\b(?:first[ -]frame|last[ -]frame|keyframe|frame[ -]0|anchor(?:ed)?)\b[^.!?\n]{0,100}<Picture\s+1>",
        r"\bkeyframe completion\b",
    )
    if any(re.search(pattern, prompt, re.I) for pattern in patterns):
        raise V2VAnchorError(
            "MiniMax V2V has no frame guide. Use <Picture 1> only as an appearance "
            "reference; remove first/last-frame, keyframe and shot-begins-from claims.")


def parse_edit_response_with_repair(content, style, repair, has_reference=True, has_video_reference=True):
    """Repair an invalid MiniMax prompt once, preserving the already validated Qwen prompt."""
    try:
        return parse_edit_response(content, style=style, has_reference=has_reference, has_video_reference=has_video_reference)
    except V2VPromptError as exc:
        original_qwen = json.loads(strip_fence(content))["qwen_prompt"].strip()
        corrected = repair(str(exc))
        # The retry edits MiniMax only: validate with the original, already valid Qwen text.
        try:
            repaired = json.loads(strip_fence(corrected))
        except (TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ValueError("MiniMax prompt repair returned incomplete JSON after one retry.") from exc
        if not isinstance(repaired, dict) or set(repaired) != {"qwen_prompt", "minimax_prompt"}:
            raise ValueError("MiniMax prompt repair must return both prompt fields after one retry.")
        repaired["qwen_prompt"] = original_qwen
        try:
            _, minimax = parse_edit_response(json.dumps(repaired), style=style, has_reference=has_reference, has_video_reference=has_video_reference)
        except V2VPromptError as exc:
            raise type(exc)("MiniMax prompt is still invalid after one repair attempt: " + str(exc)) from exc
        return original_qwen, minimax
