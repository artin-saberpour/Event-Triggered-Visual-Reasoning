# Event-Triggered Visual Reasoning

**Update visual descriptions when the scene changes, and reuse earlier descriptions when the context returns.**

Event-Triggered Visual Reasoning is an experimental pipeline for maintaining an open-ended description of a webcam scene without running a vision-language model (VLM) on every sampled frame. It combines region-aware visual change detection with a small, session-local memory of previous VLM descriptions.

The current interactive demo uses **MediaPipe**, **CLIP**, and **LLaVA 1.5**. It focuses on visible human activities and interactions with nearby objects, rather than classification into a predefined set of actions.

## Demo

[![Watch the webcam demonstration](https://i.ytimg.com/vi/rildjGtj7tQ/hqdefault.jpg)](https://youtu.be/rildjGtj7tQ)

[Watch the 1:20 webcam demo](https://youtu.be/rildjGtj7tQ)

The interface displays the camera feed, person/background segmentation, CLIP change scores, the latest description, and whether it came from LLaVA or session memory.

## How it works

```text
Webcam frame
     |
MediaPipe person/background segmentation
     |
CLIP embeddings + region-specific change gate
     |
     +-- No change --> Keep the current description
     |
     +-- Change ----> CLIP image-to-text memory lookup
                            |
                            +-- Clear match --> Reuse a previous description
                            |
                            +-- No match ----> Ask LLaVA about the full frame
                                                   |
                                            Store eligible description
```

1. **Detect visual change.** MediaPipe produces person and background views. CLIP embeds both, and a cosine-distance gate compares them with their stored references. The first sampled frame establishes a baseline. Later samples trigger when a region's distance exceeds the threshold.
2. **Look for previously described context.** On a trigger, the hybrid mode compares the full-frame CLIP image embedding with text embeddings of earlier LLaVA descriptions from the same session.
3. **Reuse or describe.** A sufficiently strong and unambiguous memory match reuses the prior text. Otherwise, LLaVA describes the **current full frame**, and its eligible response is added to memory.

The gate distinguishes person, background, and combined changes, but these are *visual-change signals*, not verified activity boundaries. The VLM receives a single image, not a video sequence or a before/after pair.

The default change threshold is `0.10`. Memory reuse requires at least two distinct stored descriptions, a best image/text cosine similarity of at least `0.30`, and a margin of at least `0.04` over the next candidate. These values are exploratory defaults, not calibrated confidence levels.

## Run the webcam demo

**The live webcam and semantic-memory implementation currently lives on the [demo branch](https://github.com/artin-saberpour/Event-Triggered-Visual-Reasoning/tree/demo/webcam-task-observations-20261106), not on `main`.** The `main` branch contains the earlier region-routing command-line implementation.

Python 3.10 or 3.11 is recommended. The full demo downloads pretrained CLIP and LLaVA weights on first use; LLaVA 1.5 7B needs substantial GPU memory and disk space. For CUDA, install a compatible PyTorch and torchvision build using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) before installing the optional model dependencies.

```bash
git clone https://github.com/artin-saberpour/Event-Triggered-Visual-Reasoning.git
cd Event-Triggered-Visual-Reasoning
git switch demo/webcam-task-observations-20261106

python -m venv .venv
# Activate .venv (Windows PowerShell: .\.venv\Scripts\Activate.ps1;
# macOS/Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
python -m pip install -e ".[models]"

python -m vision_events.task_demo --camera 0 --device auto
```

Use `--camera 1` for another camera. Press **Q** or **Esc** to quit. Press **D** to toggle expanded memory details in the dashboard. The `--device cpu` option is available but VLM inference may be very slow.

### Useful options

```bash
# Sample frames more frequently (subject to inference throughput)
python -m vision_events.task_demo --camera 0 --sample-seconds 0.5

# Compare with a mode that calls LLaVA at every visual trigger
python -m vision_events.task_demo --camera 0 --mode vlm

# Save structured event and inference records (choose a new file for each run)
python -m vision_events.task_demo --camera 0 --log outputs/session-01.jsonl

# Request a larger camera feed and show the dashboard full screen
python -m vision_events.task_demo --camera 0 --fullscreen --camera-resolution 1280x720
```

The default `hybrid` mode checks memory before asking LLaVA on a trigger. The `vlm` mode bypasses reuse, but **still uses the same visual-change gate**; it is not frame-by-frame VLM inference. The initial baseline is described by LLaVA in both modes.

The optional JSONL log records decisions, scores, timings, and model text; it does not save camera frames. The live preview may temporarily lag while inference is running, and queued frames may be replaced to keep the interface responsive.

## Earlier command-line pipeline

The `main` branch provides a simpler version of the region-aware change gate that can process a webcam or recorded video. Unlike the interactive demo, it optionally passes the selected **person**, **background**, or **full-frame** image directly to LLaVA and has no session-memory lookup.

```bash
# On main, install the lightweight package and run a model-free gate simulation
python -m pip install -e .
vision-events --simulate

# Install model dependencies to process a camera or local video
python -m pip install -e ".[models]"
vision-events --source 0 --device auto --every 30 --max-samples 100

# Add --vlm to describe triggered regions
vision-events --source 0 --device auto --every 30 --max-samples 100 --vlm
```

For a recording, replace `0` with its local file path. The `--every` argument samples every *N*th frame, not every *N* seconds. The project also contains a capture-report utility and automated gate tests; those are supporting tools rather than the focus of the interactive demonstration.

## Current scope

This is a **research prototype**, not a validated action-recognition or tracking system.

- The change gate uses CLIP embedding distances; lighting, viewpoint changes, occlusion, and segmentation errors can cause false triggers or missed changes.
- Cached descriptions are previous VLM outputs, not verified labels. A semantically similar new scene can receive the wrong earlier description.
- LLaVA describes a single frame. It cannot establish an action's intent, infer speech, or verify that an activity was completed.
- The webcam video is a qualitative demonstration. Controlled measurements of missed events, response latency, recognition accuracy, and VLM-call savings have **not** yet been reported.

A useful next evaluation is to replay the same recordings through the hybrid and VLM-only modes, then compare output quality, missed context changes, response delay, and VLM calls.

## Components and credits

This implementation uses pretrained third-party models and libraries:

- [MediaPipe Selfie Segmentation](https://github.com/google-ai-edge/mediapipe) for person/background masks
- [CLIP](https://github.com/openai/CLIP) using [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) for image/text embeddings
- [LLaVA 1.5 7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) for open-ended descriptions
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for model interfaces

Check upstream model and software licenses before redistribution or reuse.

[Project overview](https://artin-saberpour.github.io/projects/event-triggered-vision/)
