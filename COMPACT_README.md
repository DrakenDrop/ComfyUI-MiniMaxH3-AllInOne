# MiniMax H3 All in One

Integrated R2V and V2V generation nodes with local llama.cpp prompting and native ComfyUI H3 sampling.

## Install

1. Clone `https://github.com/DrakenDrop/ComfyUI-MiniMaxH3-AllInOne.git` into `ComfyUI/custom_nodes/`. Keep only one installation of this node package in `custom_nodes` to avoid duplicate node IDs.
2. Use a current ComfyUI with `TextEncodeQwenImage21`, `SDPoseKeypointExtractor`, `MiniMaxH3FunControlNetApply`, and `MiniMaxH3AddGuide` (Qwen Image 2.1 support requires 0.37.0 or newer).
3. Install requirements using the same Python environment as ComfyUI: `python -m pip install -r requirements.txt`.
4. Install llama.cpp `llama-server` and set `llama_server_path` in `config.json` (copy `config.example.json`). Put a vision GGUF and its matching mmproj in `ComfyUI/models/LLM/`. Subfolders and split GGUF models are scanned. The model choice `(llama-server yang sudah jalan)` uses an existing server; it does not switch that server's model.
5. Restart ComfyUI and open one of the workflows in `example_workflows/`. Select installed model filenames in the main node; placeholder filenames in the workflows are examples.

## Decoded IMAGE output

Select **MiniMax H3 R2V Generate (Sample + VAE Decode)** or **MiniMax H3 V2V Generate (Sample + VAE Decode, Silent)** from **MiniMax H3 / All in One**.

The primary `images` output is an **IMAGE batch of generated frames after the internal sampler and VAE Decode**. Connect it directly to **Preview Image** or **VHS Video Combine**. For Video Combine, set the frame rate to **24 FPS**; R2V can also supply its `audio` output. V2V has no audio output.

The second output, `video`, remains available for **Save Video**. No external sampler or VAE Decode node is needed.

Nodes labeled **Conditioning (Requires Sampler)** are separate advanced nodes and return intermediate model/conditioning/latent data.

**Updating existing workflows:** the first two outputs have changed order to `images`, then `video`. Reconnect these sockets on existing generation nodes, or load the updated examples. The node class IDs and input settings remain the same.

Try [V2V decoded IMAGE frames](example_workflows/v2v_decoded_images.json): source video + reference image → generation → Preview Image.

## R2V Generate

`Load Image + Load Audio -> MiniMax H3 R2V Generate -> Save Video`

Required reference assets: one image and one audio clip. No reference-video socket.

The node loads H3 ref2va, the H3 text encoder, video VAE and audio VAE. It writes the official six-section prompt through llama.cpp, encodes references, samples with BasicGuider (guidance 1), decodes, and returns a VIDEO object with audio.

- `ref_image_1_as_first_frame`: Yes declares `<Picture 1>` as the first frame in the prompt and adds a native H3 guide at frame 0. No uses the image as an appearance reference. The image is resized/cropped to the selected canvas, so a changed aspect ratio cannot preserve its original pixels exactly.
- `audio_mode`: generate from reference lets H3 use the audio as conditioning; reuse reference exactly copies the original waveform into the output, trimmed to the generated duration. A shorter reference ends before the video. H3's joint sampling still runs in both R2V audio modes.
- The sampler, scheduler, steps and optional LoRA are widgets in the node. Choosing a turbo LoRA does not change steps automatically; choose the appropriate step count yourself.

## V2V Edit

`Load Image + Load Video -> MiniMax H3 V2V Edit -> Save Video`

Exactly two media inputs are required:

| Input | Internal use |
|---|---|
| `source_video` (VIDEO) | Source frames, automatic pose extraction and Fun ControlNet motion guidance |
| `ref_image` (IMAGE) | Target clothing/person reference supplied as Qwen Image Edit's second image |

Qwen's first image is extracted automatically from source frame 0. Pose extraction, Qwen editing, Fun ControlNet, H3 sampling, and VAE Decode run inside the node. There are no external pose, edit mask, or pre-edited-frame input sockets. The source audio is ignored; the node has no audio input, audio VAE loader, audio decode, or audio output.

After updating, recreate the V2V node or load the updated example. The source input is now named `source_video`, and both media inputs must be connected. The old `v2v_video_only_input.json` filename is retained for existing download links but now also requires a reference image.

Internally:

1. Select a segment and resample to 24 FPS. Snap DOWN to a valid `17k+5` frame count, capped at 362 frames. The end can be shortened by up to 16 frames (0.67 seconds); no repeated last frames are added. This preserves the sampled source timing instead of stretching it.
2. Resize/crop the source to the selected canvas. `same as reference` follows the **source video's** aspect ratio in V2V, and the image's aspect ratio in R2V.
3. Extract body, hands, face and feet pose automatically from the source video using the selected native SDPose checkpoint. This uses full-frame single-person detection; multi-person videos may need a different workflow.
4. Qwen Image 2.1 edits source frame 0 as Image 1, using the connected reference image as Image 2.
5. llama.cpp writes an H3 `[video editing]` prompt, using source motion/camera and target appearance. The edited frame is anchored at frame 0.
6. Load the H3 Fun ControlNet Union patch and apply the pose extracted from the source video as motion control.
7. Apply the **experimental video-only wrapper**: target audio is a zero-length token stream inside the H3 transformer. The sampler carries a zero audio placeholder only for native AV-container compatibility; its noise generator produces randomness only for video. Nothing is decoded or exported as audio.
8. Sample and VAE-decode the generated frames. Return the decoded IMAGE batch and a VIDEO object with `audio=None`.

`change clothes` preserves source identity and takes the reference outfit. `change person` takes target identity and preserves source performance. A custom instruction can describe either edit in Indonesian or English. Pose control improves motion adherence; exact pixel-level/person-motion equality is not guaranteed by a generative model. Full-frame edits can alter backgrounds.

### Required model files (examples)

| Folder under `ComfyUI/models` | Models |
|---|---|
| `diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`; V2V also `qwen_image_2.1_int8_convrot.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`; V2V also `qwen3vl_8b_int8_convrot.safetensors` |
| `vae` | `minimax_h3_video_vae_int8_convrot.safetensors`; R2V also `minimax_h3_audio_vae_fp32.safetensors`; V2V also `qwen_image_2.1_vae_bf16.safetensors` |
| `model_patches` | V2V: H3-compatible Fun ControlNet Union checkpoint, including Kijai's compatible converted weights |
| `checkpoints` | V2V: `sdpose_wholebody_fp16.safetensors` |
| `LLM` | Vision GGUF for llama.cpp plus its matching mmproj |

Use the Qwen **2.1** diffusion model, encoder and VAE together. Older Qwen Image/Edit models are different architectures. The H3 text encoder and the prompter GGUF are separate models. This internal H3 loader supports native safetensors; diffusion-model GGUF loaders are not included.

## Resolution and aspect ratio

Presets: 360p, 480p, native 768p. All canvases use multiples of 32. `360p` means a 352px short edge. Native 768p follows core H3's `768*1344` area cap; ultrawide presets can have a shorter edge.

| Preset | 16:9 | 9:16 |
|---|---|---|
| 360p | 640 × 352 | 352 × 640 |
| 480p | 864 × 480 | 480 × 864 |
| 768p native | 1344 × 768 | 768 × 1344 |

Also available: 1:1, 4:3, 3:4, 3:2, 2:3, 21:9, same as reference, custom. Set custom as `width:height`, e.g. `5:4`. Changing aspect ratio crops the source/reference and affects framing.

## Prompt controls and memory

- `additional_system_prompt` appends user system instructions to the official-format system prompt. Node-level policies still enforce the first-frame option and V2V's audio-free prompt fields.
- `prompt_override` skips the LLM and must contain the six official sections in order. Truncated or incorrectly formatted responses fail visibly rather than being passed silently to H3.
- The managed llama-server is stopped after prompting by default to release VRAM. An external server is never stopped automatically. Before prompting, GPU-resident ComfyUI models are unloaded so the external LLM can load; ComfyUI reloads models when needed.
- Node outputs include the written prompt, actual canvas size/frame count, and V2V's edited frame/pose batch for inspection.
- These additions do not replace or change the existing prompter, I2V, V2V or Qwen keyframe nodes.

## Validation status

API signatures and model filenames were checked against current official ComfyUI and Qwen sources. Workflow graph consistency is checked by `tools/check_workflows.cjs`. Unit tests cover canvas sizing, temporal sampling, first-frame prompt policy, audio-tag rejection and the zero-audio-token wrapper contract: `python -m unittest discover -s tests -v`.

The creation environment has no usable Python/ComfyUI GPU runtime, so Python tests and end-to-end generation have **not** been run. In particular, video-only denoising must be tested on your installed H3 model and ComfyUI version before treating it as stable. It changes H3's usual joint audio/video inference and may affect visual quality or encounter backend/quantization incompatibilities. It has no silent fallback to normal audio generation.

The example workflows were built using public official templates as integration references. They have not been validated through end-to-end generation.

## Sources

- Official full-reference prompt format: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- Native H3 nodes: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py
- Official R2V template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json
- Fun ControlNet guide: https://github.com/Comfy-Org/docs/blob/main/tutorials/video/minimax/minimax-h3-fun-controlnet.mdx
- Qwen 2.1 template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json
