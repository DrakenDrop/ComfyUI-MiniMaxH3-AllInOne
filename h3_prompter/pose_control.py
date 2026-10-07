"""Internal skeleton extraction for the compact V2V node; no masks or RGB fallback."""


def require_dwpose():
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get("DWPreprocessor")
    if cls is None:
        raise RuntimeError(
            "Pose-only mode needs DWPose from comfyui_controlnet_aux. "
            "Install Fannovel16/comfyui_controlnet_aux with its requirements and restart ComfyUI. "
            "No external pose input is needed. RGB mode remains available explicitly.")
    return cls


def extract_pose(source, resolution=512):
    from . import llama_client as lc
    cls = require_dwpose()
    lc.log(f"V2V pose: extracting body/hands/face for {len(source)} frames at resolution {resolution}.")
    result = cls().estimate_pose(
        image=source, detect_body="enable", detect_hand="enable", detect_face="enable",
        resolution=int(resolution), bbox_detector="yolox_l.torchscript.pt",
        pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", scale_stick_for_xinsr_cn="disable")
    values = result["result"] if isinstance(result, dict) else result
    images, keypoints = values[:2]
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) != len(source):
        raise ValueError("DWPose must return one RGB skeleton frame per source frame; refusing mismatched control timing.")
    if len(keypoints) != len(source):
        raise ValueError("DWPose keypoint count does not match the source video.")
    missing = sum(not frame.get("people") for frame in keypoints)
    if missing == len(source):
        raise ValueError("DWPose detected no people. Check subject visibility or use a different source video.")
    if missing:
        lc.log(f"V2V pose: {missing}/{len(source)} frames have no detected person; those control frames are empty.")
    return images
