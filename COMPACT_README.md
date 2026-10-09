# MiniMax H3 All in One

Integrated R2V and V2V generation nodes with local llama.cpp prompting and native ComfyUI H3 sampling.

## Install

1. Clone `https://github.com/DrakenDrop/ComfyUI-MiniMaxH3-AllInOne.git` into `ComfyUI/custom_nodes/`. Keep only one installation of this node package in `custom_nodes` to avoid duplicate node IDs.
2. Use a current ComfyUI with `TextEncodeQwenImage21`, `MiniMaxH3FunControlNetApply`, and `MiniMaxH3AddGuide` (Qwen Image 2.1 support requires 0.37.0 or newer).
3. Install requirements using the same Python environment as ComfyUI: `python -m pip install -r requirements.txt`.
4. Install/extract llama.cpp `llama-server`. The node detects its executable automatically; `llama_server_path` can stay empty. Put a vision GGUF and its matching mmproj in `ComfyUI/models/LLM/`. Subfolders and split GGUF models are scanned. The model choice `(llama-server yang sudah jalan)` uses an existing server; it does not switch that server's model.
5. The V2V examples require ComfyUI-VideoHelperSuite for the video loader. Restart ComfyUI and open one of the workflows in `example_workflows/`. Select installed model filenames in the main node; placeholder filenames in the workflows are examples.

## H3 Character Swap without Qwen Image

Use **MiniMax H3 Character Swap (Sample + VAE Decode)**, node type `MiniMaxH3CharacterSwap`.
Example: [character_swap.json](example_workflows/character_swap.json).

Connect VHS Load Video's **IMAGE** output to **ref_video** and Load Image's **IMAGE** output to **ref_image**. Like the official H3 reference-video input, `ref_video` is an IMAGE sequence at 24 FPS; it is not a VIDEO object. Connect the generated `images` to VHS Video Combine at 24 FPS.

| Input or output | Routing |
|---|---|
| `ref_video` (IMAGE) | Selected source frames go directly to H3 `ref_video_0` / `<Video 1>` and the same frames go to RGB FunControlNet |
| `ref_image` (IMAGE) | First image goes directly to H3 `ref_image_0` / `<Picture 1>` as the target character |
| `images` (IMAGE output) | Sampled and VAE-decoded video frames |
| `minimax_prompt` (STRING output) | The actual prompt used by H3 |

Select the H3 diffusion model, H3 text encoder, video VAE and H3-compatible FunControlNet weights inside the node. FunControlNet is part of the workflow, defaults to strength 1.0 and runs over the full sampling range; strength 0 disables its application for comparisons. No Qwen image model, Qwen image encoder, Qwen image VAE, DWPose, mask or pinned first-frame guide is used. The H3 text encoder and the optional llama.cpp enhancer may themselves use Qwen-family language models; those are separate from Qwen Image generation.

Describe the character swap in `instruction`; for multiple people, specify which person to replace. By default the enhancer transfers the reference identity, face, hair and outfit while requesting preservation of source performance, camera, background and lighting. Change the instruction if clothing should be preserved. The source video also contains the old character's appearance, so identity transfer and motion/lighting preservation still depend on H3 and control strength; exact movement is not guaranteed.

The node supports the same resolution/aspect presets, official duration rounding, H3 LoRA, `minimax_thinking`, `additional_system_prompt`, model retention and enhancer caching. `prompt_override` bypasses llama.cpp entirely and is used verbatim. Otherwise a vision GGUF sees sampled source frames and the target character image. The returned MiniMax prompt is accepted without content validation or correction retries, including plain-text responses.

No audio is decoded or returned. `strict video-only (experimental)` is the node default; `native AV (discard audio)` internally denoises audio latents as in native H3. The example selects native AV for comparison with the native sampler. Select the mode deliberately; this new node has CPU routing tests but has not been validated end-to-end on a GPU.

## Decoded IMAGE output

Select **MiniMax H3 R2V Generate (Sample + VAE Decode)** or **MiniMax H3 V2V Generate (Sample + VAE Decode, Silent)** from **MiniMax H3 / All in One**.

The primary `images` output is an **IMAGE batch of generated frames after the internal sampler and VAE Decode**. Connect it directly to **Preview Image** or **VHS Video Combine**. For Video Combine, set the frame rate to **24 FPS**; R2V can also supply its `audio` output. V2V has no audio output.

**V2V outputs:**

| Output | Type | Content |
|---|---|---|
| `images` | IMAGE | H3-generated video frames after VAE Decode; connect to VHS Video Combine at 24 FPS or Preview Image |
| `qwen_image` | IMAGE | Decoded Qwen edit at its generated resolution, before resizing for H3; connect to Preview Image or Save Image |
| `minimax_prompt` | STRING | Enhanced MiniMax prompt actually passed to H3, including the silent-video policy |
| `qwen_prompt` | STRING | Enhanced edit prompt actually passed to Qwen Image 2.1 |

Connect either STRING output to a compatible text display/save node to inspect the prompt. All outputs become available after the all-in-one generation finishes; connecting only the Qwen preview still runs the full V2V pipeline. R2V also provides VIDEO and AUDIO outputs. No external sampler or VAE Decode node is needed.

Nodes labeled **Conditioning (Requires Sampler)** are separate advanced nodes and return intermediate model/conditioning/latent data.

**Updating existing workflows:** recreate the V2V node or load an updated example to remove obsolete sockets. V2V has two IMAGE inputs and four outputs: `images`, `qwen_image`, `minimax_prompt`, and `qwen_prompt`. The original `images` output remains slot 0. R2V keeps `images` first and `video` second.

Try [V2V decoded IMAGE frames](example_workflows/v2v_decoded_images.json): source video + reference image → generation → Preview Image.

## R2V Generate

`Load Image + Load Audio -> MiniMax H3 R2V Generate -> Save Video`

Required reference assets: `ref_image` and `ref_audio`. Optionally connect `ref_image_2`, `ref_image_3`, and `ref_audio_2` for up to three images and two audio clips. Existing workflows can leave all new inputs disconnected. No reference-video socket.

Connected images are labeled consecutively in input order: `<Picture 1>`, `<Picture 2>`, `<Picture 3>`. If only `ref_image` and `ref_image_3` are connected, they become `<Picture 1>` and `<Picture 2>`. Only the first image in each input batch is used. Audio labels are `<Audio 1>` (`ref_audio`) and `<Audio 2>` (`ref_audio_2`). The enhancer and H3 conditioning receive the same ordered references. Numbering does not pair an audio clip with an image: describe each subject's audio reference in `instruction`.

For example, with all inputs connected: `The person in <Picture 1> speaks using the voice from <Audio 2>; the person in <Picture 2> answers using the voice from <Audio 1>. Use <Picture 3> for the setting.` This guides generated audio; it does not splice the two clips into the output. When using `prompt_override`, use these same labels in your complete H3 prompt.

The node loads H3 ref2va, the H3 text encoder, video VAE and audio VAE. It writes a simple or official six-section prompt through llama.cpp, encodes references, samples with BasicGuider (guidance 1), decodes, and returns a VIDEO object with audio.

- `ref_image_1_as_first_frame`: Yes declares only `<Picture 1>` (`ref_image`) as the first frame in the prompt and adds a native H3 guide at frame 0. Additional images remain references. No uses the first image as an appearance reference. The first image also determines `same as reference` canvas proportions and is resized/cropped to the selected canvas, so a changed aspect ratio cannot preserve its original pixels exactly.
- `audio_mode`: generate from reference lets H3 use all connected audio clips as conditioning, with subject mapping specified in `instruction`. Reuse reference exactly copies only the original `ref_audio` (`<Audio 1>`) waveform into the output, trimmed to the generated duration; it never mixes or concatenates `ref_audio_2`. A shorter first audio reference ends before the video. Both connected audio clips still reach H3 conditioning, and H3's joint sampling runs in both R2V audio modes.
- The sampler, scheduler, steps and optional LoRA are widgets in the node. Choosing a turbo LoRA does not change steps automatically; choose the appropriate step count yourself.

### Duration-aware R2V enhancement

Both R2V nodes explicitly tell the enhancer the selected `duration_seconds`. For 5 seconds, the instruction is: **"This video is for 5 seconds. Write a prompt appropriate for a 5-second video."** It asks the LLM to choose the shots, actions, pacing and narrative to suit the duration and user request, keep any described times within that duration, and preserve reference labels and subject/audio mapping. The same instruction uses 10 or 15 when those durations are selected.

There are no fixed 5/10/15-second action templates, forced timelines, duration-specific prompt lengths or new timestamp rejection/rewriting rules. The LLM decides what fits. Both official and simple styles receive the duration; simple R2V permits the requested actions and pacing in its short paragraph. Existing format/first-frame policies remain, and V2V's simple-style behavior is unchanged.

H3's existing frame rounding is unchanged: 5/10/15 seconds produce 124/243/362 frames, or approximately 5.167/10.125/15.083 seconds. The instruction uses the **selected** duration; the standard official target metadata still reports the physical frame duration. This change does not retime or trim the video.

`prompt_override` still bypasses enhancement. Tests verify that 5/10/15 seconds reach the actual LLM messages without fixed action templates. LLM wording and adherence remain model-dependent; different durations do not guarantee distinct outputs.

## R2V FastH3 (experimental)

Select **MiniMax H3 R2V FastH3 Generate (Experimental)**, or load [the example with three images and two audio clips](example_workflows/r2v_fasth3_experimental.json). Choose your local model files, reference media and llama.cpp model/server before running.

This is a separate node (`MiniMaxH3R2VFastH3Generate`) that inherits R2V generation. It retains `MiniMaxH3ReferenceToVideo` conditioning, consecutive reference labels, the first-image-only frame guide, instruction-based audio-to-subject mapping, and exact reuse of only the first audio. Existing R2V workflows and defaults stay unchanged.

Only the model patch chain is adapted from the supplied `video_fastvideo_fasth3_i2v.json` example:

`Load Model (+ optional LoRA) -> MiniMaxH3SigmaShift -> ModelAttentionBackend -> BlockSparseAttention -> BasicGuider AND BasicScheduler`

| Native node | FastH3 variant defaults |
| --- | --- |
| `MiniMaxH3SigmaShift` (ModelSamplingMiniMaxH3) | Video shift `10`, audio shift `3` |
| `ModelAttentionBackend` | `comfy kitchen attention` |
| `BlockSparseAttention` (Model Sparse Attention) | `vsa`; keep `10%`; start `0.2`, end `1.0`; dense blocks empty; minimum tokens `12288`; extra tokens `256`; sink conditioning `exact_kv_and_rows`; verbose off |

The settings are exposed as `shift_video`, `shift_audio`, `attention_backend` and `vsa_*` widgets. The VSA method and exact conditioning-sink mode match the template. The sampler defaults to `res_multistep`, `simple`, 8 steps, with denoise 1 and BasicGuider guidance 1. The model selector prefers FastH3 filenames; the example selects `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`. Model, LoRA, sampler and step settings remain selectable. Patches are rebuilt from the loaded model each run, rather than accumulated in its loader cache.

**Checkpoint limitation:** the [upstream FastH3 8-Step V2 model card](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) says **FL2VA and Ref2VA were not distilled**. The supplied I2V template describes a broader FL2VA scope, but agrees that Ref2VA was not distilled. This variant applies the requested model patches while retaining R2V; it does not establish trained multi-reference support, reference fidelity, audio quality or a speedup. For supported reference-based use, keep the base H3 Ref2VA checkpoint and the original R2V node. [ComfyUI checkpoint files](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy).

**Dependencies:** recent ComfyUI providing all three native nodes, including `BlockSparseAttention`'s VSA selection, plus compatible Comfy Kitchen GPU kernels. The reference template records comfy-core `0.39.1`; feature availability is the requirement, not a proven minimum version. Missing patch imports produce an update message when this variant runs; they do not prevent loading the original nodes. Native `ModelAttentionBackend` warns and uses PyTorch if Comfy Kitchen dense attention is unavailable; sparse attention can also run dense outside its active range or below its token threshold. This is upstream behavior, so the configured chain is not a guarantee of GPU acceleration. No dependency installation or model download is performed automatically.

CPU tests cover patch order, shared patched-model routing into guider/scheduler, settings, repeated runs, missing dependencies and all inherited R2V reference/audio behavior. Full ComfyUI execution, kernel compatibility, VRAM, speed and generated quality still require a GPU validation run.

## V2V Edit

`Load Video (IMAGE output) + optional Load Image -> MiniMax H3 V2V Edit -> IMAGE output`

Source video is required; the additional reference image is optional:

| Input | Internal use |
|---|---|
| `source_video` (IMAGE batch) | Source frames for Qwen and internal pose extraction (or legacy RGB control) |
| `ref_image` (IMAGE) | Optional target clothing/person reference supplied as Qwen Image Edit's second image |

The `source_video` socket accepts an **IMAGE batch**, matching native H3's `ref_video` input. Connect the **IMAGE output of VHS Load Video**. Set `force_rate = 24`, `select_every_nth = 1`, and leave the loader's VAE input disconnected. IMAGE batches contain no FPS metadata, so the node interprets the frames at **24 FPS**. A batch loaded at a different FPS would change timing; resample in the video loader first.

Qwen's first image is extracted automatically from source frame 0. Qwen editing, DWPose extraction in pose-only mode, Fun ControlNet, H3 sampling, and VAE Decode run inside the node. There are no external pose, edit mask, or pre-edited-frame input sockets. The source audio is ignored; the node has no audio input, audio VAE loader, audio decode, or audio output.

After updating, recreate the V2V node or load the updated example. The `source_video` socket now uses IMAGE rather than VIDEO; reconnect the video loader\'s IMAGE output. Only `source_video` must be connected. `ref_image` may remain disconnected. Use `v2v_text_only_edit.json` for a source-video-and-text example.

Internally:

1. Read the supplied 24 FPS IMAGE batch and select a segment. Snap DOWN to a valid `17k+5` frame count, capped at 362 frames. The end can be shortened by up to 16 frames (0.67 seconds); no repeated last frames are added. This preserves the sampled source timing instead of stretching it.
2. Resize/crop the source to the selected canvas. `same as reference` follows the **source video's** aspect ratio in V2V, and the image's aspect ratio in R2V.
3. Enhance the single user instruction into separate internal Qwen and MiniMax prompts.
4. Using the Qwen prompt prepared by the shared enhancer, Qwen Image 2.1 edits source frame 0 as Image 1. A connected reference becomes Image 2; otherwise the text instruction supplies the target appearance.
5. Use the MiniMax prompt prepared by the shared enhancer. Supply only the Qwen-edited image as H3's appearance reference, without a forced first-frame guide.
6. In pose-only mode, extract DWPose skeletons internally and send them to Fun ControlNet; omit source RGB reference-video conditioning in H3. Legacy RGB mode instead sends source RGB frames to both paths. Neither mode uses masks.
7. Select strict video-only sampling (experimental, default), or native AV sampling with discarded audio latents for reference-workflow comparison. See the mode table below.
8. Sample and VAE-decode the video frames. Return the frames, the Qwen-generated image, and the two enhanced prompts. Neither mode decodes or returns audio; native AV still computes audio latents internally.

`change clothes` preserves source identity; `change person` changes the requested identity while preserving source performance. The target appearance comes from the optional reference or the text instruction. A custom instruction can describe either edit in Indonesian or English. Control compatibility and motion adherence depend on the selected mode, detector, and ControlNet weights. Exact pixel-level/person-motion equality is not guaranteed by a generative model. Full-frame edits can alter backgrounds.

### Comparing against the supplied V2V workflow

The supplied working graph uses source RGB frames directly as Fun ControlNet control, the Qwen result as H3's only reference image, no forced first-frame guide, and `res_multistep` with `simple` scheduling. The all-in-one V2V `rgb source (legacy)` mode follows those connections. The raw appearance reference is used only by Qwen. In the MiniMax prompt, `<Picture 1>` means the Qwen-edited image and `<Video 1>` means the source.

`h3_sampling_mode` makes a consequential difference:

| Mode | Internal computation | Output |
|---|---|---|
| `strict video-only (experimental)` (default) | Removes audio tokens with the experimental wrapper | Four existing outputs; no audio |
| `native AV (discard audio)` | Uses the ordinary H3 audio-video sampler and noise, without the wrapper | Four existing outputs; audio latents are discarded without decoding |

Native AV is provided for comparison with the reference graph; it **does compute audio latents internally**. Keep strict mode if the requirement is no audio generation at all. Strict mode is not equivalent to the native graph and can change video quality. Neither mode loads an audio VAE or returns audio.

[V2V H3 reference comparison](example_workflows/v2v_reference_native.json) uses the H3 settings from the supplied working graph: `UnZipMeMultiModal.safetensors`, Fun ControlNet Union 2.0 BF16, FP16 video VAE, 768p, 2 steps, `res_multistep`, `simple`, and `ref_image_size=max`. It explicitly selects native AV. The Qwen settings are retained from the user's confirmed-good Qwen stage (base Qwen 2.1, 6 steps, and the selected Qwen LoRA); this example is for comparing the H3 stage, not a general recommended Qwen preset. Select the installed filenames before running.

Existing nodes retain saved settings after updating: set the scheduler to `simple` explicitly or load the new example. Do not assume that different H3 checkpoints produce comparable results at the same two-step schedule. The reference's native sampling and the previous strict wrapper cannot be compared as if they were the same configuration.

These changes and routing tests do not establish a single proven cause of the reported smearing. No end-to-end GPU reproduction has been performed here.

### CLIP and VAE selection

The all-in-one node loads CLIP/text encoders and VAEs internally. Choose filenames in these dropdowns:

| Purpose | Node field | Model folder |
|---|---|---|
| MiniMax CLIP/text encoder | `h3_text_encoder` | `ComfyUI/models/text_encoders/` |
| MiniMax video VAE | `h3_video_vae` | `ComfyUI/models/vae/` |
| Qwen CLIP/text encoder | `qwen_text_encoder` | `ComfyUI/models/text_encoders/` |
| Qwen image VAE | `qwen_vae` | `ComfyUI/models/vae/` |

External CLIP/VAE loader connections are not part of this node's interface. The llama.cpp `llm_model` and `mmproj` fields belong to the prompt enhancer and do not replace these model selections.

### Separate MiniMax and Qwen LoRAs

| Model | LoRA selector | Strength |
|---|---|---|
| MiniMax H3 | `lora_name` | `lora_strength` |
| Qwen Image 2.1 (V2V only) | `qwen_lora_name` | `qwen_lora_strength` |

Place LoRA files in `ComfyUI/models/loras/` (subfolders are supported). The Qwen selector lists installed LoRAs with Qwen-named files first; filenames are not a compatibility check. Choose a LoRA compatible with the Qwen Image 2.1 diffusion model you selected.

The Qwen LoRA is applied only to the Qwen diffusion model before first-frame editing. It does not patch H3, the text encoder, or the llama.cpp prompt enhancer. Select `(none)` or set `qwen_lora_strength` to 0 to disable it. One Qwen LoRA is supported per run. LoRA selection does not change `qwen_steps` automatically.

Existing workflows default to no Qwen LoRA. Recreate the V2V node or load an updated example to display the two new controls.

### Required model files (examples)

| Folder under `ComfyUI/models` | Models |
|---|---|
| `diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`; V2V also `qwen_image_2.1_int8_convrot.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`; V2V also `qwen3vl_8b_int8_convrot.safetensors` |
| `vae` | `minimax_h3_video_vae_int8_convrot.safetensors`; R2V also `minimax_h3_audio_vae_fp32.safetensors`; V2V also `qwen_image_2.1_vae_bf16.safetensors` |
| `model_patches` | V2V: H3-compatible Fun ControlNet Union checkpoint, including Kijai's compatible converted weights |
| `LLM` | Vision GGUF for llama.cpp plus its matching mmproj |

Use the Qwen **2.1** diffusion model, encoder and VAE together. Older Qwen Image/Edit models are different architectures. The H3 text encoder and the prompter GGUF are separate models. This internal H3 loader supports native safetensors; diffusion-model GGUF loaders are not included.

## One instruction for Qwen and MiniMax

The V2V `ref_image` input is optional. Leave it disconnected for a text-directed edit such as **"Change her dress to red"**: Qwen receives only the source first frame as `<image1>`. With a reference connected, that image becomes `<image2>`. Both paths send the Qwen result to H3 as `<Picture 1>`, while motion comes from extracted skeletons in pose-only mode, or RGB frames in legacy mode. Without a reference, write a specific edit instruction; an empty instruction cannot supply a target outfit/person.

V2V exposes one `instruction` field. For example: **"Change her clothes to this"**, with the target outfit connected to `ref_image`.

Before diffusion sampling, the selected vision GGUF reads the source first frame, the reference image and sampled source-motion frames through llama.cpp. The enhancer supplies two coordinated prompts; official MiniMax sections are normalized into text:

- `qwen_prompt`: an editing directive using `<image1>` for the source first frame and `<image2>` only when an optional reference is connected.
- `minimax_prompt`: the six-section H3 video-editing prompt, preserving source performance and using the intended Qwen edit as an appearance reference, without a frame guide.

These are generated from the single user instruction and exposed as STRING outputs for inspection. The enhancer runs before Qwen generates the edited frame. The MiniMax prompt describes the planned edit; it does not claim to inspect a result that has not been generated yet.

Qwen editing rules are adapted from the [official edit enhancer system prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt). Qwen also publishes [PE-I2I weights](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I); this node uses your selected llama.cpp vision model with adapted instructions, rather than requiring those specific weights.

V2V still asks the enhancer for dedicated edit-preservation prompts, but all validation of the returned prompt content is bypassed. Missing or unexpected asset tags, first/last-frame or keyframe wording, incomplete sections, and audio wording are passed through without a correction retry. This changes prompt acceptance only; it does not add frame guides or audio generation.

The prompt enhancer requires working vision support and a matching mmproj. V2V extracts qwen_prompt and minimax_prompt when the response is valid JSON. String values are preserved verbatim; structured values are serialized as text without requiring particular fields. If a field or JSON envelope cannot be read, that output uses the raw response instead. A malformed shared response is therefore passed to both prompt inputs. Network, model-loading, and sampling errors are not bypassed.

### Motion control without masking

`motion_control = pose only (DWPose)` is the default for new V2V nodes:

1. The source first frame goes to Qwen as image 1; an optional appearance reference becomes image 2.
2. DWPose extracts body, hand and face skeletons from every source frame.
3. Fun ControlNet receives the skeleton images.
4. H3 receives only the Qwen-edited `<Picture 1>`. No source RGB video is passed to H3's reference-video conditioning.
5. H3 samples and decodes IMAGE frames as before.

There is **no masking, segmentation, inpainting, compositing, or external pose input** in this mode. The enhancer can inspect source frames as context, but its H3 prompt uses `[reference generation]` and no `<Video N>` labels. Lighting/background appearance is guided by the Qwen image; this does not guarantee exact background, lighting, or motion preservation.

Install [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) and its requirements in the same environment as ComfyUI, then restart. The registered `DWPreprocessor` is invoked internally. It uses `yolox_l.torchscript.pt` and `dw-ll_ucoco_384_bs5.torchscript.pt`; the auxiliary package downloads missing weights on its first run. No SDPose checkpoint is required. A missing DWPose node or a clip with no detected people produces an explicit error, never an automatic RGB fallback.

`pose_resolution` defaults to **512** and affects detection, not output resolution. `control_strength` still sets Fun ControlNet strength. Pose extraction adds preprocessing time and currently runs for each executed V2V generation; the enhancer/Qwen caches do not cache skeletons. Detection can miss occluded or small subjects.

`motion_control = rgb source (legacy)` retains the earlier source-RGB control and H3 reference-video path. Existing comparison examples explicitly select this mode. For the new path, open [V2V pose only](example_workflows/v2v_pose_only.json) or select `pose only (DWPose)` on your current node. This example uses native AV sampling with discarded audio; that mode still computes audio latents internally.

## Resolution and aspect ratio

Presets: 360p, 480p, native 768p. All canvases use multiples of 32. Each preset fixes a pixel **budget** that stays constant for every aspect ratio (including `same as reference`), matching the official H3 buckets. Extreme ratios trade side lengths inside the same budget instead of shrinking or inflating the canvas.

| Preset | 16:9 | 9:16 | 1:1 | Budget |
|---|---|---|---|---|
| 360p | 640 × 352 | 352 × 640 | 480 × 480 | ~0.23 MP |
| 480p | 832 × 480 | 480 × 832 | 640 × 640 | ~0.40 MP |
| 768p native | 1344 × 768 | 768 × 1344 | 992 × 992 | ~1.03 MP |

Also available: 1:1, 4:3, 3:4, 3:2, 2:3, 21:9, same as reference, custom. Set custom as `width:height`, e.g. `5:4`. Changing aspect ratio crops the source/reference and affects framing.

### Duration and frame alignment

R2V and V2V use the official duration expression at 24 FPS: `max(5, round(seconds * 24)) + (5 - (max(5, round(seconds * 24)) % 17)) % 17`, capped at this node's 362-frame limit. V2V no longer rounds down. Examples: 5s gives 124 frames (5.167s), 10s gives 243 frames (10.125s), and 15s gives 362 frames (15.083s). `max_seconds` is therefore a duration target, not a strict output cutoff.

V2V starts at `start_seconds` and takes the aligned number of source frames without stretching time. Extra source frames are used when available. If alignment extends beyond the supplied IMAGE batch, only the missing tail is filled by repeating the final source frame; the log reports `repeated_tail_frames`. If the source segment is shorter than requested, its available duration is aligned instead of freezing the rest of the requested clip. Input remains a 24 FPS IMAGE batch. Limit frames in the upstream video loader separately if you want to avoid loading the full video.

### Keep models loaded between runs

Set `keep_models_loaded=true` on the all-in-one node to skip its explicit `unload_all_models()` before an enhancer cache miss. V2V also retains its Qwen model, encoder, and VAE loader objects across edits, reusing unchanged weights even when the seed or prompt changes. Changed model paths or file metadata replace the corresponding cached object. H3/FunControlNet loader objects already persist within the node. Qwen LoRA patches are still applied for each new edit.

The default is `false` to preserve previous memory behavior. This is a retention preference, not a VRAM lock: ComfyUI can still offload models under memory pressure, and other nodes or cleanup extensions can unload them. Keeping the models and an external llama.cpp process on the same GPU requires enough free memory for both weights and runtime allocations. Turning the option off releases this node's retained Qwen loader; it does not force an immediate global memory purge on a cached run.

`unload_llm_after_prompt` is independent: it controls stopping the managed llama.cpp server, not H3 or FunControlNet. Keeping it `true` releases LLM memory before diffusion. `reuse_preprocessing=true` can skip identical enhancer/Qwen work entirely. The new option reduces avoidable unloading and loader reconstruction; it does not promise faster denoising. Compare the stage timings on consecutive runs.

### Separate enhancer thinking controls

V2V exposes `qwen_thinking` for the Qwen image-edit prompt and `minimax_thinking` for the H3 video prompt. Each offers `off`, `low`, `medium`, and `xhigh`, defaulting to `off`. R2V exposes only `minimax_thinking`. These control the llama.cpp prompt enhancer, not Qwen/H3 diffusion sampling. Both use the selected `llm_model` and `mmproj`.

When the modes match, a shared request generates both prompts. When they differ, Qwen prompting runs first with its own mode; MiniMax prompting then uses its mode and the accepted Qwen prompt as context. V2V does not validate or retry generated prompt content. Both settings are included in the enhancer cache key and printed in the V2V log.

Thinking support and effort levels depend on the selected model and llama.cpp chat template. Higher settings can increase time and consume the output budget; the existing backend may retry without thinking if it produces no final answer. These controls do not guarantee better video quality or pose preservation.

### MiniMax enhancement style

Both all-in-one nodes provide `minimax_prompt_style`:

- `simple`: concise free-form MiniMax instructions without the six section headings.
- `official` (default): the six official H3 sections in canonical order.

V2V still enhances one user instruction into a Qwen edit prompt and a MiniMax prompt. The style selector affects only MiniMax; both modes retain source/reference labels and the silent-video policy.

In official V2V mode, the enhancer is asked for a structured object with six fields; returned objects are serialized into H3 text without checking field count, content, or order. Returned strings are used as-is in both official and simple modes. The style setting guides generation, not validation.

`max_tokens` is the output budget per request, not a guarantee of correct formatting. Matching thinking modes share that budget across both prompts; different modes give each prompt its own request budget. An 8192-token budget can still produce invalid headings; increase it only when the response is actually truncated. V2V passes incomplete output through; inspect its STRING outputs if the result is unexpected.

### Automatic llama-server discovery

For a locally selected GGUF, the node finds and starts an installed `llama-server` automatically. It checks:

1. An explicit `llama_server_path` override, if provided.
2. `LLAMA_SERVER_PATH` / `LLAMA_CPP_DIR` environment variables and PATH.
3. The custom-node package, ComfyUI and its portable parent, registered LLM folders, the Python executable folder, home/Downloads, and common Windows/Linux locations such as `C:/llama.cpp` and `/workspace/llama.cpp`.

Known layouts include the folder itself, `bin`, `build/bin`, and `build/bin/Release`, including extracted `llama-*` release folders. The scan is bounded; it does not search every disk recursively. Keep the executable beside its required libraries. Discovery does not download llama.cpp.

If an old config contains a stale example path, set `llama_server_path` to an empty string or `auto`. A valid manual override retains priority; an invalid explicit override reports an error. Unusual locations can be listed in `llama_server_search_dirs` in config.json. The chosen executable is logged in the ComfyUI console.

Discovery also applies when configured external-server `autostart` is enabled. Selecting `(llama-server yang sudah jalan)` still uses the configured server URL; the node does not scan network ports or change that server's model.

## Prompt controls and memory

- `additional_system_prompt` appends user system instructions to the official-format system prompt. Node-level policies still enforce the first-frame option and V2V's audio-free prompt fields.
- R2V only: `prompt_override` skips the LLM. Its format must match `minimax_prompt_style`: free-form text for simple, six sections for official.
- The managed llama-server is stopped after prompting by default to release VRAM. An external server is never stopped automatically. Before prompting, GPU-resident ComfyUI models are unloaded so the external LLM can load; ComfyUI reloads models when needed.
- V2V exposes decoded video frames, the Qwen edit, and both enhanced prompts. R2V additionally exposes its prompt, canvas size and frame count.
- These additions do not replace or change the existing prompter, I2V, V2V or Qwen keyframe nodes.

### Diagnosing an unchanged V2V result

After updating, restart ComfyUI. Existing workflow widgets retain their selected values. The console logs a `V2V H3 config:` line with the actual model, encoder, VAE, H3 LoRA, sampling mode, sampler, scheduler, steps, seed, canvas, frame count and control settings. Share this line and the current workflow when comparing a failed render with a working one. A prompt alone cannot establish which sampling path or checkpoint ran, and fixing prompt wording is not proof that visual artifacts are resolved.

## V2V performance

`reuse_preprocessing` defaults to **true**. Each V2V node retains its last successful managed-llama.cpp prompt pair and one decoded Qwen image in CPU memory. It does not retain additional Qwen model weights or video latents. Identical prompt requests skip llama.cpp inference and the preceding ComfyUI model unload; identical Qwen inputs skip its loaders, encoding, sampling and decode.

The enhancer key includes the exact image/text request, generation parameters, configuration, and GGUF/mmproj file metadata (all shards for split GGUFs). External-server prompt results are not cached because the server can change models without changing its URL. The Qwen key includes every pixel of its one or two input images, prompt, seed, model/encoder/VAE and LoRA file metadata, steps, and resolution. Set `reuse_preprocessing=false` to clear/bypass both caches. Caches disappear when the node is recreated or ComfyUI restarts.

This mainly helps when rerunning with different **H3** sampler, steps, control strength, or model while preprocessing inputs stay identical. Changing the shared seed reruns both stages. First renders and changed inputs still perform their full preprocessing; no GPU speedup factor has been measured. Contiguous source clips also use a tensor view to avoid an unnecessary full-video copy.

The console prints `V2V timing:` per stage and a `V2V timing total:` summary: frame preparation, enhancer, Qwen edit, pose extraction, H3 loading, conditioning, control setup, H3 sampling, and VAE decode. These are wall-clock measurements without forced GPU synchronization, so asynchronous work may cross stage boundaries. Sampling includes lazy Fun ControlNet encoding/loading. The total covers this node, not video loading or final video encoding.

Keep the working model, scheduler, steps and resolution for the first timing comparison. If a managed llama.cpp server shares the generation GPU, `unload_llm_after_prompt=true` stops that server after enhancement and releases its allocation before diffusion; the next cache hit avoids restarting it. This setting does not stop an external server. Share the timing summary, GPU/VRAM, and offloading log before deciding on attention, quantization, or model changes.

## Validation status

API signatures and model filenames were checked against current official ComfyUI and Qwen sources. Workflow graph consistency is checked by `tools/check_workflows.cjs`. Unit tests cover canvas sizing, temporal sampling, first-frame prompt policy, audio-tag rejection and the zero-audio-token wrapper contract: `python -m unittest discover -s tests -v`.

Python syntax checks, all 63 unit tests (including discovery, prompt-format regressions, and native/strict V2V routing), and the workflow graph checks pass. End-to-end generation has **not** been run because no ComfyUI GPU runtime is available here. In particular, video-only denoising must be tested on your installed H3 model and ComfyUI version before treating it as stable. It changes H3's usual joint audio/video inference and may affect visual quality or encounter backend/quantization incompatibilities. It has no silent fallback to normal audio generation.

The example workflows were built using public official templates as integration references. They have not been validated through end-to-end generation.

## Sources

- Official full-reference prompt format: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- Native H3 nodes: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py
- Official R2V template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
- Fun ControlNet guide: https://github.com/Comfy-Org/docs/blob/main/tutorials/video/minimax/minimax-h3-fun-controlnet.mdx
- Qwen 2.1 template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json
