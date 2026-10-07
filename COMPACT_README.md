# MiniMax H3 All in One

Integrated R2V and V2V generation nodes with local llama.cpp prompting and native ComfyUI H3 sampling.

## Install

1. Clone `https://github.com/DrakenDrop/ComfyUI-MiniMaxH3-AllInOne.git` into `ComfyUI/custom_nodes/`. Keep only one installation of this node package in `custom_nodes` to avoid duplicate node IDs.
2. Use a current ComfyUI with `TextEncodeQwenImage21`, `MiniMaxH3FunControlNetApply`, and `MiniMaxH3AddGuide` (Qwen Image 2.1 support requires 0.37.0 or newer).
3. Install requirements using the same Python environment as ComfyUI: `python -m pip install -r requirements.txt`.
4. Install/extract llama.cpp `llama-server`. The node detects its executable automatically; `llama_server_path` can stay empty. Put a vision GGUF and its matching mmproj in `ComfyUI/models/LLM/`. Subfolders and split GGUF models are scanned. The model choice `(llama-server yang sudah jalan)` uses an existing server; it does not switch that server's model.
5. The V2V examples require ComfyUI-VideoHelperSuite for the video loader. Restart ComfyUI and open one of the workflows in `example_workflows/`. Select installed model filenames in the main node; placeholder filenames in the workflows are examples.

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

Required reference assets: one image and one audio clip. No reference-video socket.

The node loads H3 ref2va, the H3 text encoder, video VAE and audio VAE. It writes a simple or official six-section prompt through llama.cpp, encodes references, samples with BasicGuider (guidance 1), decodes, and returns a VIDEO object with audio.

- `ref_image_1_as_first_frame`: Yes declares `<Picture 1>` as the first frame in the prompt and adds a native H3 guide at frame 0. No uses the image as an appearance reference. The image is resized/cropped to the selected canvas, so a changed aspect ratio cannot preserve its original pixels exactly.
- `audio_mode`: generate from reference lets H3 use the audio as conditioning; reuse reference exactly copies the original waveform into the output, trimmed to the generated duration. A shorter reference ends before the video. H3's joint sampling still runs in both R2V audio modes.
- The sampler, scheduler, steps and optional LoRA are widgets in the node. Choosing a turbo LoRA does not change steps automatically; choose the appropriate step count yourself.

## V2V Edit

`Load Image + Load Video (IMAGE output) -> MiniMax H3 V2V Edit -> IMAGE output`

Exactly two media inputs are required:

| Input | Internal use |
|---|---|
| `source_video` (IMAGE batch) | Source frames connected directly to Fun ControlNet control_video |
| `ref_image` (IMAGE) | Target clothing/person reference supplied as Qwen Image Edit's second image |

The `source_video` socket accepts an **IMAGE batch**, matching native H3's `ref_video` input. Connect the **IMAGE output of VHS Load Video**. Set `force_rate = 24`, `select_every_nth = 1`, and leave the loader's VAE input disconnected. IMAGE batches contain no FPS metadata, so the node interprets the frames at **24 FPS**. A batch loaded at a different FPS would change timing; resample in the video loader first.

Qwen's first image is extracted automatically from source frame 0. Qwen editing, Fun ControlNet, H3 sampling, and VAE Decode run inside the node. There are no external pose, edit mask, or pre-edited-frame input sockets. The source audio is ignored; the node has no audio input, audio VAE loader, audio decode, or audio output.

After updating, recreate the V2V node or load the updated example. The `source_video` socket now uses IMAGE rather than VIDEO; reconnect the video loader\'s IMAGE output. Both media inputs must be connected. The old `v2v_video_only_input.json` filename is retained for existing download links but now also requires a reference image.

Internally:

1. Read the supplied 24 FPS IMAGE batch and select a segment. Snap DOWN to a valid `17k+5` frame count, capped at 362 frames. The end can be shortened by up to 16 frames (0.67 seconds); no repeated last frames are added. This preserves the sampled source timing instead of stretching it.
2. Resize/crop the source to the selected canvas. `same as reference` follows the **source video's** aspect ratio in V2V, and the image's aspect ratio in R2V.
3. Enhance the single user instruction into separate internal Qwen and MiniMax prompts.
4. Using the Qwen prompt prepared by the shared enhancer, Qwen Image 2.1 edits source frame 0 as Image 1 with the connected reference image as Image 2.
5. Use the MiniMax prompt prepared by the shared enhancer. Supply only the Qwen-edited image as H3's appearance reference, without a forced first-frame guide.
6. Load the H3 Fun ControlNet Union patch and pass the source IMAGE batch directly to its `control_video` input. No pose preprocessor or mask is used.
7. Select strict video-only sampling (experimental, default), or native AV sampling with discarded audio latents for reference-workflow comparison. See the mode table below.
8. Sample and VAE-decode the video frames. Return the frames, the Qwen-generated image, and the two enhanced prompts. Neither mode decodes or returns audio; native AV still computes audio latents internally.

`change clothes` preserves source identity and takes the reference outfit. `change person` takes target identity and preserves source performance. A custom instruction can describe either edit in Indonesian or English. Direct RGB control follows the requested workflow; compatibility and motion adherence depend on the selected ControlNet weights. Exact pixel-level/person-motion equality is not guaranteed by a generative model. Full-frame edits can alter backgrounds.

### Comparing against the supplied V2V workflow

The supplied working graph uses source RGB frames directly as Fun ControlNet control, the Qwen result as H3's only reference image, no forced first-frame guide, and `res_multistep` with `simple` scheduling. The all-in-one V2V routing now follows those connections. The raw appearance reference is used only by Qwen. In the MiniMax prompt, `<Picture 1>` means the Qwen-edited image and `<Video 1>` means the source.

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

V2V exposes one `instruction` field. For example: **"Change her clothes to this"**, with the target outfit connected to `ref_image`.

Before diffusion sampling, the selected vision GGUF reads the source first frame, the reference image and sampled source-motion frames through llama.cpp. A single enhancement response supplies two prompts; official MiniMax sections are normalized into text:

- `qwen_prompt`: an editing directive using `<image1>` for the source first frame and `<image2>` for the reference.
- `minimax_prompt`: the six-section H3 video-editing prompt, preserving source performance and using the intended edited frame as its frame-0 guide.

These are generated from the single user instruction and exposed as STRING outputs for inspection. The enhancer runs before Qwen generates the edited frame. The MiniMax prompt describes the planned edit; it does not claim to inspect a result that has not been generated yet.

Qwen editing rules are adapted from the [official edit enhancer system prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt). Qwen also publishes [PE-I2I weights](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I); this node uses your selected llama.cpp vision model with adapted instructions, rather than requiring those specific weights.

The prompt enhancer requires working vision support and a matching mmproj. Malformed or incomplete responses stop with an error instead of silently using an unrelated prompt.

### Direct video control

The source frames are passed directly to Fun ControlNet's `control_video` input. No SDPose checkpoint or pose extraction is required. The implementation uses this control input because native Fun ControlNet reads its separate `source_video` input only with a mask. Direct RGB control has not been GPU-validated here; use compatible weights from your working workflow.

## Resolution and aspect ratio

Presets: 360p, 480p, native 768p. All canvases use multiples of 32. `360p` means a 352px short edge. Native 768p follows core H3's `768*1344` area cap; ultrawide presets can have a shorter edge.

| Preset | 16:9 | 9:16 |
|---|---|---|
| 360p | 640 × 352 | 352 × 640 |
| 480p | 864 × 480 | 480 × 864 |
| 768p native | 1344 × 768 | 768 × 1344 |

Also available: 1:1, 4:3, 3:4, 3:2, 2:3, 21:9, same as reference, custom. Set custom as `width:height`, e.g. `5:4`. Changing aspect ratio crops the source/reference and affects framing.

### MiniMax enhancement style

Both all-in-one nodes provide `minimax_prompt_style`:

- `simple`: concise free-form MiniMax instructions without the six section headings.
- `official` (default): the six official H3 sections in canonical order.

V2V still enhances one user instruction into a Qwen edit prompt and a MiniMax prompt. The style selector affects only MiniMax; both modes retain source/reference labels and the silent-video policy.

In official V2V mode, the enhancer is asked for a structured object with six fields; the node formats it into the H3 text prompt. Complete text responses, Markdown headings, JSON section objects, and reordered sections are normalized. Missing, duplicate or empty sections produce an explicit error. Simple mode does not run the six-section validator.

`max_tokens` is an output budget shared by both prompts, not a guarantee of correct formatting. An 8192-token budget can still produce invalid headings; increase it only when the response is actually truncated. Errors now identify missing sections instead of assuming a token shortage.

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

## Validation status

API signatures and model filenames were checked against current official ComfyUI and Qwen sources. Workflow graph consistency is checked by `tools/check_workflows.cjs`. Unit tests cover canvas sizing, temporal sampling, first-frame prompt policy, audio-tag rejection and the zero-audio-token wrapper contract: `python -m unittest discover -s tests -v`.

Python syntax checks, all 18 unit tests (including discovery, prompt-format regressions, and native/strict V2V routing), and the workflow graph checks pass. End-to-end generation has **not** been run because no ComfyUI GPU runtime is available here. In particular, video-only denoising must be tested on your installed H3 model and ComfyUI version before treating it as stable. It changes H3's usual joint audio/video inference and may affect visual quality or encounter backend/quantization incompatibilities. It has no silent fallback to normal audio generation.

The example workflows were built using public official templates as integration references. They have not been validated through end-to-end generation.

## Sources

- Official full-reference prompt format: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- Native H3 nodes: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py
- Official R2V template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
- Fun ControlNet guide: https://github.com/Comfy-Org/docs/blob/main/tutorials/video/minimax/minimax-h3-fun-controlnet.mdx
- Qwen 2.1 template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json
