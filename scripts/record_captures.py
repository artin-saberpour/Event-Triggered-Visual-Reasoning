"""Save a reviewable, local report from real image files and model outputs.

This does not upload images or results. The model adapters are the same ones
used by vision-events; this script records their inputs, regions and scores.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from vision_events.gate import RegionGate
from vision_events.models import CLIP_MODEL_ID, CLIP_SAFE_REVISION


def record_frames(paths, model, gate: RegionGate, output: Path) -> list[dict]:
    """Process ordered photos, save lossless region images and gate decisions."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No image files were provided")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    image_dir = output / "images"
    image_dir.mkdir()
    records = []
    with (output / "events.jsonl").open("w", encoding="utf-8") as log:
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            # encode_frame accepts OpenCV's BGR ordering.
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            images, embeddings = model.encode_frame(bgr)
            if set(images) != {"full", "person", "background"}:
                raise ValueError("Model must return full, person and background images")
            decision = gate.observe(embeddings)
            image_paths = {}
            for name in ("full", "person", "background"):
                relative = Path("images") / f"{index:03d}_{name}.png"
                images[name].save(output / relative)
                image_paths[name] = relative.as_posix()
            record = {"sample": index, "source_image": path.name,
                      "baseline": decision.baseline, "selected_region": decision.region,
                      "cosine_distances": decision.distances, "images": image_paths}
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            records.append(record)
    (output / "run.json").write_text(json.dumps({
        "threshold": gate.threshold,
        "clip_model": CLIP_MODEL_ID,
        "clip_revision": CLIP_SAFE_REVISION,
        "segmentation": "MediaPipe Selfie Segmentation",
        "vlm_enabled": False,
        "input_count": len(records),
    }, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(render_report(records, gate.threshold), encoding="utf-8")
    return records


def render_report(records: list[dict], threshold: float) -> str:
    """Display actual saved views next to their corresponding gate values."""
    sections = []
    for item in records:
        dist = item["cosine_distances"]
        values = ("Baseline established" if item["baseline"] else
                  "Person distance: {:.4f} | Background distance: {:.4f} | Full-frame distance: {:.4f}".format(
                      dist["person"], dist["background"], dist["full"]))
        route = "Baseline" if item["baseline"] else item["selected_region"] or "No event"
        figures = "".join(
            f'<figure><img src="{escape(item["images"][name], quote=True)}" alt="{name} view">'
            f'<figcaption>{name.title()} view</figcaption></figure>'
            for name in ("full", "person", "background")
        )
        sections.append(f'<section><h2>Sample {item["sample"] + 1} · {escape(item["source_image"])}</h2>'
                        f'<p>{escape(values)}<br><strong>Selected region: {escape(route)}</strong></p>'
                        f'<div class="views">{figures}</div></section>')
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Vision event capture report</title><style>"
            "body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:3rem auto;padding:0 1rem;color:#222;background:white}"
            "section{border-top:1px solid #ddd;padding:1.2rem 0}h1{margin-bottom:.3rem}h2{font-size:1.1rem}"
            ".views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}figure{margin:0}"
            "img{width:100%;height:auto;display:block}figcaption{color:#555;margin-top:.3rem}"
            "@media(max-width:650px){.views{grid-template-columns:1fr}}"
            "</style></head><body><h1>Event-triggered vision · capture report</h1>"
            f"<p>CLIP cosine-distance threshold: {threshold:g}. "
            "Distances refer to the current stored region reference. LLaVA was not used.</p>"
            + "\n".join(sections) + "</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local report from ordered real-world image captures")
    parser.add_argument("--frames", type=Path, required=True, help="Folder of JPG or PNG frames, sorted by filename")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are not overwritten")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit the number of inputs; 0 means all")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    if not args.frames.is_dir():
        parser.error("--frames must be an existing directory")
    paths = sorted(p for p in args.frames.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        parser.error("No JPG or PNG files found in --frames")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    gate = RegionGate(args.threshold)
    from vision_events.models import VisionModels
    model = VisionModels(device=args.device, use_vlm=False)
    try:
        records = record_frames(paths, model, gate, args.output)
    finally:
        model.close()
    print(f"Saved {len(records)} samples to {args.output}. Open report.html to review.")


if __name__ == "__main__":
    main()
