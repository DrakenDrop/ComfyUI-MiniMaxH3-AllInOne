"""Validate and apply node policies to the official six-section rewrite."""
import re

FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")


def apply_policy(prompt, first_frame=False, silent=False):
    matches = list(re.finditer(r"(?im)^\s*(" + "|".join(FIELDS) + r")\s*:\s*", prompt))
    if tuple(m.group(1).lower() for m in matches) != FIELDS:
        raise ValueError("Expected the six official H3 sections in order; increase max_tokens or fix prompt_override")
    sections = {m.group(1).lower(): prompt[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(prompt)].strip()
                for i, m in enumerate(matches)}
    if silent:
        if re.search(r"<Audio\s+\d+>|<d>", prompt, re.I):
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
