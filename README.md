# Event-triggered vision with region-aware change detection

An exploratory prototype that uses person/background segmentation and CLIP image embeddings to decide when an image region has changed enough to warrant further inspection. An optional LLaVA adapter can describe the selected region when a change is detected.

This repository is an **AI-assisted reconstruction and refactoring of exploratory research scripts** from the author's archive. It is separate from the final method of the associated research paper. MediaPipe, CLIP and LLaVA are pretrained third-party components, not models trained for this project.

## How it works

1. MediaPipe separates a sampled frame into person-only and background-only images; masked pixels are black.
2. CLIP encodes the full frame and both regions. The first sample establishes reference embeddings.
3. The gate measures cosine distance from the previous reference for each region. A person-only change selects the person image, a background-only change selects the background, and simultaneous changes select the full frame.
4. If `--vlm` is enabled, the selected image is sent to local LLaVA for a response. Otherwise, the program only reports the gate decisions.

The full-frame distance is reported for context; the two region distances determine routing. A region reference is updated after its event. For a full-frame event, both region references are refreshed, correcting a repeated-trigger issue in the archived script. The default distance threshold is a starting point, not a validated operating point.

**Validation status:** the model-independent gate and its synthetic demo pass tests. The cleaned camera/video, model-download and VLM paths have not yet been run end to end in the current environment. No latency, call-reduction, accuracy or deployment claims are made. MediaPipe segmentation, blacked-out regions, scene lighting and threshold choice can affect events; LLaVA output may be incorrect.

## Run the synthetic demo

Python 3.10 or 3.11 is recommended. In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
vision-events --simulate
python -m unittest discover -s tests -v
```

On macOS or Linux, activate with `source .venv/bin/activate`. The synthetic output is a deterministic illustration of routing using artificial vectors; it is not a model prediction or benchmark.

## Run on a camera or video

Install the optional model dependencies. For CUDA, first install the matching PyTorch and torchvision builds using the [official PyTorch selector](https://pytorch.org/get-started/locally/). The first run downloads model weights from Hugging Face and needs network access and disk space.

```powershell
python -m pip install -e ".[models]"
vision-events --source 0 --device cuda --every 30 --max-samples 100
```

`--source 0` selects the first camera; a local video path can be used instead. `--every 30` analyzes every 30th frame, not one frame per fixed wall-clock interval. The program prints one JSON object per analyzed frame, with a baseline flag, region distances and selected region. `null` means no event. Stop an unlimited run with Ctrl+C. Use `--device cpu` for CPU execution; inference can be slow.

To additionally load LLaVA and generate a response for each event:

```powershell
vision-events --source 0 --device cuda --every 30 --max-samples 100 --vlm --question "What changed in the visible scene?"
```

LLaVA receives only the selected current image, not the previous frame. Its answer describes or reasons about what is visible; it cannot directly compare the two frames. The 7B model needs substantial GPU memory and disk space; CPU execution needs much more system memory and may be impractical. The exact memory requirement depends on the environment. Omit `--vlm` for the lighter gate-only experiment.

Tune `--threshold` from 0 to 2 for cosine distance (default 0.1). Frame sampling and threshold should be assessed for the intended scene and task; this prototype has no calibrated defaults. The program does not save camera frames by default.

## Credits

- [MediaPipe Selfie Segmentation](https://github.com/google-ai-edge/mediapipe) for the person mask.
- [CLIP](https://github.com/openai/CLIP), via [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32), for image embeddings.
- [LLaVA](https://github.com/haotian-liu/LLaVA), via [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf), for optional image descriptions.
- The model interfaces use [Hugging Face Transformers](https://huggingface.co/docs/transformers/).

Review upstream code, model and dataset licenses before reuse.
