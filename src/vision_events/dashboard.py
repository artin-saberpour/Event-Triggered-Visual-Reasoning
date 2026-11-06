"""Presentation and diagnostic views for the live webcam demo."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# OpenCV uses BGR; these muted ink colors are shared by the chart and its key.
PAPER = (237, 240, 242)
PANEL = (247, 249, 249)
INK = (43, 43, 36)
MUTED = (103, 104, 94)
RULE = (192, 196, 190)
PERSON = (121, 85, 42)
BACKGROUND = (58, 81, 157)
RESET = (128, 131, 125)


def wrap_display_text(cv2: Any, value: str, max_width: int, scale: float, max_lines: int,
                      *, measure: Callable[[str], float] | None = None) -> list[str]:
    """Wrap text to the panel width; the complete response stays in the log."""
    value = " ".join(value.split()).encode("ascii", "replace").decode("ascii")
    font = cv2.FONT_HERSHEY_SIMPLEX

    def fits(text: str) -> bool:
        width = measure(text) if measure else cv2.getTextSize(text, font, scale, 1)[0][0]
        return width <= max_width

    lines: list[str] = []
    line = ""
    for word in value.split():
        proposed = f"{line} {word}".strip()
        if fits(proposed):
            line = proposed
            continue
        if line:
            lines.append(line)
            line = ""
        while word and not fits(word):
            end = len(word)
            while end > 1 and not fits(word[:end]):
                end -= 1
            lines.append(word[:end])
            word = word[end:]
        line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and not fits(last + "..."):
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    return lines


@lru_cache(maxsize=32)
def _label_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Use the system UI font where possible, with portable fallbacks."""
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        windows_fonts / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=1024)
def _label_mask(value: str, size: int, bold: bool) -> tuple[Image.Image, int, int]:
    """Cache antialiased label masks so static text is not rasterized every frame."""
    font = _label_font(size, bold)
    left, top, right, bottom = font.getbbox(value, anchor="ls")
    mask = Image.new("L", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(mask).text((-left, -top), value, fill=255, font=font, anchor="ls")
    return mask, left, top


def _paint_labels(canvas: np.ndarray, labels: list[tuple[str, int, int, int, tuple[int, int, int], bool]]) -> np.ndarray:
    """Paint all labels together to avoid a full image conversion per label."""
    image = Image.fromarray(canvas)
    for value, x, y, size, color, bold in labels:
        # The image contains BGR channels, so use the same order for ink colors.
        mask, offset_x, offset_y = _label_mask(value, size, bold)
        image.paste(color, (x + offset_x, y + offset_y), mask)
    return np.asarray(image).copy()


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _nice_step(minimum: float) -> float:
    """Choose a readable axis interval without quantizing plotted values."""
    magnitude = 10 ** math.floor(math.log10(minimum))
    fraction = minimum / magnitude
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= factor + 1e-10:
            return factor * magnitude
    return 10.0 * magnitude


def clip_chart_window(history: list[dict[str, Any]], elapsed: float, threshold: float,
                      window_seconds: float = 30.0) -> tuple[list[dict[str, Any]], float, float, float]:
    """Return visible processed samples and a cosine-distance plot range."""
    end = max(float(elapsed), 0.001)
    start = max(0.0, end - window_seconds)
    samples = [item for item in history if start <= item["time_seconds"] <= end]
    max_distance = max((max(item["person"], item["background"]) for item in samples), default=0.0)
    target = min(2.0, max(0.20, threshold * 1.25, max_distance * 1.1))
    step = _nice_step(target / 4.0)
    upper = min(2.0, math.ceil(target / step - 1e-10) * step)
    return samples, start, end, upper


def draw_clip_chart(cv2: Any, canvas: np.ndarray, history: list[dict[str, Any]], elapsed: float,
                    threshold: float, x: int, y: int, width: int, height: int) -> None:
    """Plot actual gate distances, with a clear scale and event markers."""
    muted = MUTED
    person_color, background_color = PERSON, BACKGROUND
    grid, threshold_color = (218, 222, 217), (140, 144, 138)
    samples, start, end, upper = clip_chart_window(history, elapsed, threshold)
    left, right = x + 53, x + width - 10
    top, bottom = y + 8, y + height - 24
    plot_w, plot_h = right - left, bottom - top

    def text(value: str, tx: int, ty: int, color: tuple[int, int, int] = muted) -> None:
        cv2.putText(canvas, value, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    def sx(seconds: float) -> int:
        return int(round(left + (seconds - start) / (end - start) * plot_w))

    def sy(distance: float) -> int:
        return int(round(bottom - min(max(distance, 0.0), upper) / upper * plot_h))

    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL, -1)
    step = _nice_step(upper / 4.0)
    for index in range(int(round(upper / step)) + 1):
        value = min(upper, index * step)
        tick_y = sy(value)
        cv2.line(canvas, (left, tick_y), (right, tick_y), grid, 1, cv2.LINE_AA)
        label = f"{value:.2f}"
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
        text(label, left - label_width - 9, tick_y + 4)

    # The x-axis marks capture time, with gaps left empty when frames were not processed.
    first_tick = int(math.ceil(start / 5.0) * 5)
    for tick in range(first_tick, int(end) + 1, 5):
        tick_x = sx(float(tick))
        if tick_x - left < 60 or right - tick_x < 60:
            continue
        cv2.line(canvas, (tick_x, top), (tick_x, bottom), grid, 1, cv2.LINE_AA)
        text(_clock(tick), tick_x - 16, y + height - 4)

    threshold_y = sy(threshold)
    for cursor in range(left, right, 18):
        cv2.line(canvas, (cursor, threshold_y), (min(cursor + 10, right), threshold_y),
                 threshold_color, 1, cv2.LINE_AA)
    text(_clock(start), left, y + height - 4)
    text(_clock(end), right - 38, y + height - 4)
    if not samples:
        text("Waiting for processed CLIP samples", left + 18, top + plot_h // 2, muted)
        return
    # A triggering frame becomes its region's new reference. Its distance to
    # that reference is exactly zero; dashed drops mark this derived reset,
    # rather than presenting it as an extra sampled observation.
    for item in samples:
        affected = item["trigger_region"]
        if affected is not None:
            values = ([item["person"], item["background"]] if affected == "full"
                      else [item[affected]])
            marker_x = sx(item["time_seconds"])
            marker_y = min(sy(value) for value in values)
            for cursor in range(min(marker_y + 6, bottom), bottom, 9):
                cv2.line(canvas, (marker_x, cursor), (marker_x, min(cursor + 4, bottom)),
                         RESET, 1, cv2.LINE_AA)
    for region, color in (("person", person_color), ("background", background_color)):
        previous = None
        for item in samples:
            point = (sx(item["time_seconds"]), sy(item[region]))
            # Connect from the exact post-reset zero, but never across long gaps.
            if previous is not None and item["time_seconds"] - previous[0] <= 4.0:
                segment_start = ((previous[1][0], bottom) if previous[2] in (region, "full")
                                 else previous[1])
                cv2.line(canvas, segment_start, point, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
            if item["trigger_region"] in (region, "full"):
                cv2.circle(canvas, point, 5, INK, 1, cv2.LINE_AA)
            previous = (item["time_seconds"], point, item["trigger_region"])




def render_dashboard(cv2: Any, frame: np.ndarray, state: dict[str, Any], elapsed: float,
                     show_details: bool = False) -> np.ndarray:
    """Render the camera, context, model stages and change trace."""
    height, width = 1080, 1920
    canvas = np.full((height, width, 3), PAPER, dtype=np.uint8)
    labels: list[tuple[str, int, int, int, tuple[int, int, int], bool]] = []

    def text(value: str, x: int, y: int, size: int = 16,
             color: tuple[int, int, int] = INK, bold: bool = False) -> None:
        labels.append((str(value), x, y, size, color, bold))

    def wrapped(value: str, max_width: int, size: int, max_lines: int,
                bold: bool = False) -> list[str]:
        font = _label_font(size, bold)
        return wrap_display_text(cv2, value, max_width, 0.5, max_lines, measure=font.getlength)

    def line(x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int] = RULE,
             thickness: int = 1) -> None:
        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def place_view(image: np.ndarray | None, x: int, y: int, w: int, h: int) -> None:
        canvas[y:y + h, x:x + w] = (33, 34, 32)
        if image is None:
            text("Awaiting frame", x + 16, y + h // 2, 15, (220, 222, 219))
            return
        source_height, source_width = image.shape[:2]
        scale = min(w / source_width, h / source_height)
        image_w = max(1, int(source_width * scale))
        image_h = max(1, int(source_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        view = cv2.resize(image, (image_w, image_h), interpolation=interpolation)
        x0, y0 = x + (w - image_w) // 2, y + (h - image_h) // 2
        canvas[y0:y0 + image_h, x0:x0 + image_w] = view

    # A quiet header and rules give the screen the character of a lab instrument.
    text("EVENT-TRIGGERED VISUAL REASONING", 25, 38, 27, INK, True)
    text("CAMERA  /  VISUAL CHANGE  /  CONTEXT", 26, 62, 13, MUTED, True)
    cv2.circle(canvas, (1732, 32), 4, BACKGROUND, -1, cv2.LINE_AA)
    text("LIVE", 1744, 37, 14, BACKGROUND, True)
    text(_clock(elapsed), 1838, 38, 18, INK, True)
    line(24, 75, 1896, 75, INK)

    text("CAMERA", 25, 101, 14, INK, True)
    text("LIVE INPUT", 818, 101, 13, MUTED)
    place_view(frame, 24, 111, 880, 472)
    line(914, 90, 914, 583)

    px, pw = 935, 961
    text("CURRENT CONTEXT", px, 101, 14, INK, True)
    description = state["task"] or "Waiting for the first observation"
    for index, item in enumerate(wrapped(description, pw - 26, 26, 3)):
        text(item, px, 145 + index * 33, 26, INK)
    line(px, 239, px + pw, 239)
    source = state["last_source"]
    if source == "vlm":
        source_text, source_color = "GENERATED BY LLaVA", INK
    elif source == "cache":
        source_text, source_color = "REUSED FROM CLIP MEMORY", PERSON
    else:
        source_text, source_color = "WAITING FOR A DESCRIPTION", MUTED
    text(source_text, px, 266, 14, source_color, True)
    if state["answer_time"] is not None:
        age = max(0, elapsed - state["answer_time"])
        text(f"Frame captured {age:.1f}s ago", px, 290, 15, MUTED)
    else:
        text("The first observation is being prepared", px, 290, 15, MUTED)
    status = "PROCESSING" if state["busy"] else "MONITORING"
    text(status, px + pw - 110, 290, 13, PERSON if state["busy"] else MUTED, True)

    line(px, 317, px + pw, 317, INK)
    if show_details:
        text("EXPANDED DETAILS", px, 346, 15, INK, True)
        text("D  BACK TO HISTORY", px + pw - 153, 346, 13, MUTED)
        line(px, 358, px + pw, 358)
        match = state["last_match"] or {}
        candidate = match.get("best_description") or "No candidate yet"
        text("BEST CACHED CANDIDATE", px, 384, 13, MUTED, True)
        for index, item in enumerate(wrapped(candidate, pw - 20, 17, 2)):
            text(item, px, 411 + index * 23, 17, INK)
        best = match.get("best_similarity")
        gap = match.get("margin")
        best_str = f"{best:.3f}" if best is not None else "--"
        gap_str = f"{gap:.3f}" if gap is not None else "--"
        text(f"Similarity: {best_str}   Gap: {gap_str}   Decision: {match.get('reason', '--')}",
             px, 478, 16, INK)
        text(f"Samples: {state['samples']}   Triggers: {state['triggers']}   Queued replaced: {state['dropped']}",
             px, 512, 16, INK)
        text("Timings are measured after the change gate.", px, 548, 14, MUTED)
    else:
        text("RECENT OBSERVATIONS", px, 346, 15, INK, True)
        text("LATEST 3", px + pw - 73, 346, 13, MUTED)
        line(px, 358, px + pw, 358)
        recent = state["timeline"][-3:]
        if not recent:
            text("Observations will appear here", px, 392, 16, MUTED)
        for index, event in enumerate(recent):
            y = 387 + index * 62
            text(_clock(event["time_seconds"]), px, y, 15, INK, True)
            event_source = "LLaVA" if event["source"] == "vlm" else "Memory"
            event_color = MUTED if event["source"] == "vlm" else PERSON
            text(event_source, px, y + 20, 13, event_color)
            for line_index, item in enumerate(wrapped(event["task"], pw - 114, 17, 2)):
                text(item, px + 94, y + line_index * 22, 17, INK)
            if index < len(recent) - 1:
                line(px, y + 33, px + pw, y + 33)

    # Stage panels use dividers, not floating cards, so they read as one pipeline.
    line(24, 599, 1896, 599, INK)
    line(633, 611, 633, 843)
    line(1253, 611, 1253, 843)
    left, mid, right = 24, 645, 1265
    text("01", left, 629, 15, PERSON, True)
    text("MediaPipe", left + 36, 629, 23, INK, True)
    text("Separate person and background", left, 654, 15, MUTED)
    text("PERSON VIEW", left, 680, 13, MUTED, True)
    text("BACKGROUND VIEW", left + 290, 680, 13, MUTED, True)
    previews = state["region_previews"]
    place_view(previews.get("person"), left, 688, 275, 127)
    place_view(previews.get("background"), left + 290, 688, 275, 127)
    preview_time = state["preview_time"]
    preview_label = _clock(preview_time) if preview_time is not None else "--"
    text(f"Sample captured at: {preview_label}", left, 837, 14, MUTED)

    text("02", mid, 629, 15, PERSON, True)
    text("CLIP", mid + 36, 629, 23, INK, True)
    text("Detect visual change and match memory", mid, 654, 15, MUTED)
    distances = state["last_distances"]
    person_value = f"{distances['person']:.3f}" if distances else "--"
    background_value = f"{distances['background']:.3f}" if distances else "--"
    match = state["last_match"] or {}
    best, gap = match.get("best_similarity"), match.get("margin")
    best_str = f"{best:.3f}" if best is not None else "--"
    gap_str = f"{gap:.3f}" if gap is not None else "--"
    reasons = {"matched": "Matched", "empty": "No descriptions", "need_more_examples": "Need 2 descriptions",
               "low_similarity": "Low similarity", "ambiguous": "Ambiguous match"}
    reason = reasons.get(match.get("reason"), "--")
    clip_rows = [
        ("Samples / triggers", f"{state['samples']} / {state['triggers']}"),
        ("Person / background distance", f"{person_value} / {background_value}"),
        ("Last trigger / gate threshold", f"{state['last_trigger'] or '--'} / {state['gate_threshold']:.2f}"),
        ("Memory similarity / gap", f"{best_str} / {gap_str}"),
        ("Memory decision", reason),
        ("Match / gap thresholds", f"{state['min_similarity']:.2f} / {state['min_margin']:.2f}"),
    ]
    for index, (label, value) in enumerate(clip_rows):
        y = 683 + index * 30
        text(label, mid, y, 16, MUTED)
        text(value, mid + 342, y, 16, INK)

    text("03", right, 629, 15, PERSON, True)
    text("LLaVA", right + 36, 629, 23, INK, True)
    text("Describe when no memory matches", right, 654, 15, MUTED)
    source_label = {"vlm": "LLaVA", "cache": "CLIP memory"}.get(source, "--")
    resolve = state["last_resolution_ms"]
    resolve_text = f"{resolve:.0f} ms" if resolve is not None else "--"
    llava_rows = [
        ("Context source", source_label),
        ("LLaVA calls / memory reuses", f"{state['vlm_calls']} / {state['cache_hits']}"),
        ("Descriptions stored", str(state["cache_size"])),
        ("Resolve time after gate", resolve_text),
        ("Queued frames replaced", str(state["dropped"])),
        ("Mode", state["mode"].upper()),
    ]
    for index, (label, value) in enumerate(llava_rows):
        y = 683 + index * 30
        text(label, right, y, 16, MUTED)
        text(value, right + 342, y, 16, INK)

    # The full-width trace retains the measured values and explicit reset marks.
    cv2.rectangle(canvas, (24, 855), (1896, 1038), PANEL, -1)
    line(24, 855, 1896, 855, INK)
    text("CLIP CHANGE OVER TIME", 45, 881, 17, INK, True)
    text("Cosine distance to each region reference", 335, 881, 13, MUTED)
    line(810, 875, 834, 875, PERSON, 2)
    text("Person", 845, 881, 14, PERSON, True)
    line(932, 875, 956, 875, BACKGROUND, 2)
    text("Background", 967, 881, 14, BACKGROUND, True)
    text(f"Threshold {state['gate_threshold']:.2f}", 1126, 881, 14, MUTED)
    cv2.circle(canvas, (1327, 875), 5, INK, 1, cv2.LINE_AA)
    text("Trigger  |  dashed drop = reset", 1339, 881, 14, MUTED)
    draw_clip_chart(cv2, canvas, state["distance_history"], elapsed, state["gate_threshold"],
                    42, 889, 1836, 144)

    text("D  Expanded details", 25, 1065, 13, MUTED)
    text("Q  Quit", 190, 1065, 13, MUTED)
    text("Dashed drops: reference reset   |   Gaps: samples unavailable", 410, 1065, 13, MUTED)
    text("Contexts are inferred from individual frames.", 1460, 1065, 13, MUTED)
    return _paint_labels(canvas, labels)
