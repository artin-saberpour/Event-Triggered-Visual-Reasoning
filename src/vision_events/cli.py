"""Run a deterministic gate demo or process a local camera/video stream."""

from __future__ import annotations

import argparse
import json

from .gate import RegionGate


def simulate(threshold: float) -> None:
    gate = RegionGate(threshold)
    a, b = [1.0, 0.0], [0.0, 1.0]
    frames = [
        {"full": a, "person": a, "background": a},
        {"full": a, "person": a, "background": a},
        {"full": b, "person": b, "background": a},
        {"full": b, "person": b, "background": a},
        {"full": a, "person": a, "background": b},
        {"full": a, "person": a, "background": b},
    ]
    for index, frame in enumerate(frames):
        decision = gate.observe(frame)
        print(json.dumps({"sample": index, "synthetic": True, "baseline": decision.baseline,
                          "region": decision.region, "distances": decision.distances}))


def run_video(args: argparse.Namespace) -> None:
    from .models import VisionModels

    if args.every < 1 or args.max_samples < 0:
        raise ValueError("--every must be positive and --max-samples must be nonnegative")
    model = VisionModels(device=args.device, use_vlm=args.vlm)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = model.cv2.VideoCapture(source)
    if not capture.isOpened():
        model.close()
        raise RuntimeError(f"Could not open video source: {args.source}")
    gate, frame_number, samples = RegionGate(args.threshold), 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if (frame_number - 1) % args.every:
                continue
            images, embeddings = model.encode_frame(frame)
            decision = gate.observe(embeddings)
            record = {"frame": frame_number, "baseline": decision.baseline,
                      "region": decision.region, "distances": decision.distances}
            if decision.region and args.vlm:
                record["response"] = model.describe(images[decision.region], args.question)
            print(json.dumps(record), flush=True)
            samples += 1
            if args.max_samples and samples >= args.max_samples:
                break
    finally:
        capture.release()
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental region-aware vision event detection")
    parser.add_argument("--simulate", action="store_true", help="Run a synthetic gate-only demo")
    parser.add_argument("--source", help="Camera index (such as 0) or path to a local video")
    parser.add_argument("--threshold", type=float, default=0.1, help="Cosine distance threshold, 0 to 2")
    parser.add_argument("--every", type=int, default=30, help="Analyze every Nth frame")
    parser.add_argument("--max-samples", type=int, default=0, help="Stop after N analyzed frames; 0 means unlimited")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vlm", action="store_true", help="Load LLaVA and ask about triggered regions")
    parser.add_argument("--question", default="Describe what is visible in this image.")
    args = parser.parse_args()
    if args.simulate and args.source is not None:
        parser.error("Choose either --simulate or --source")
    if args.simulate:
        simulate(args.threshold)
    elif args.source is not None:
        run_video(args)
    else:
        parser.error("Supply --simulate or --source")


if __name__ == "__main__":
    main()
