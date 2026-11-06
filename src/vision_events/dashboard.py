"""Presentation and diagnostic views for the live webcam demo."""

from __future__ import annotations

from typing import Any

import numpy as np


def wrap_display_text(cv2: Any, value: str, max_width: int, scale: float, max_lines: int) -> list[str]:
    """Wrap text to the panel width; the complete response stays in the log."""
    value = " ".join(value.split()).encode("ascii", "replace").decode("ascii")
    font = cv2.FONT_HERSHEY_SIMPLEX

    def fits(text: str) -> bool:
        return cv2.getTextSize(text, font, scale, 1)[0][0] <= max_width

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


def _rounded_box(cv2: Any, canvas: np.ndarray, x: int, y: int, width: int, height: int,
                 color: tuple[int, int, int], radius: int = 12) -> None:
    """Fill a rounded rectangle using only OpenCV drawing primitives."""
    right, bottom = x + width - 1, y + height - 1
    cv2.rectangle(canvas, (x + radius, y), (right - radius, bottom), color, -1)
    cv2.rectangle(canvas, (x, y + radius), (right, bottom - radius), color, -1)
    for cx, cy in ((x + radius, y + radius), (right - radius, y + radius),
                   (x + radius, bottom - radius), (right - radius, bottom - radius)):
        cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def render_dashboard(cv2: Any, frame: np.ndarray, state: dict[str, Any], elapsed: float,
                     show_details: bool = False) -> np.ndarray:
    """Render the real camera, latest context and optional diagnostic details."""
    height, width = 720, 1280
    background, card = (19, 17, 16), (34, 30, 27)
    white, muted = (244, 243, 240), (177, 165, 154)
    teal = (207, 226, 91)
    amber, divider = (145, 198, 245), (62, 56, 50)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)

    def text(value: str, x: int, y: int, scale: float = 0.5,
             color: tuple[int, int, int] = white, thickness: int = 1) -> None:
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def line(x1: int, y1: int, x2: int, y2: int) -> None:
        cv2.line(canvas, (x1, y1), (x2, y2), divider, 1, cv2.LINE_AA)

    # Header: a title, actual live state and elapsed time, without metrics.
    text("EVENT-TRIGGERED VISION", 25, 37, 0.72, white, 2)
    text("LIVE CONTEXT DEMO", 26, 60, 0.4, muted)
    cv2.circle(canvas, (1110, 36), 4, teal, -1, cv2.LINE_AA)
    text("LIVE", 1121, 41, 0.46, teal)
    text(_clock(elapsed), 1190, 41, 0.5, muted)

    # Camera panel: letterbox to preserve the actual frame's aspect ratio.
    vx, vy, vw, vh = 24, 88, 786, 442
    canvas[vy:vy + vh, vx:vx + vw] = (10, 12, 16)
    source_height, source_width = frame.shape[:2]
    scale = min(vw / source_width, vh / source_height)
    image_w, image_h = max(1, int(source_width * scale)), max(1, int(source_height * scale))
    view = cv2.resize(frame, (image_w, image_h))
    x0, y0 = vx + (vw - image_w) // 2, vy + (vh - image_h) // 2
    canvas[y0:y0 + image_h, x0:x0 + image_w] = view
    _rounded_box(cv2, canvas, vx + 15, vy + 15, 145, 30, background, 8)
    cv2.circle(canvas, (vx + 30, vy + 30), 4, teal, -1, cv2.LINE_AA)
    text("LIVE CAMERA", vx + 42, vy + 35, 0.43, white)

    # A short explanation makes the trigger-and-understand concept legible.
    _rounded_box(cv2, canvas, 24, 546, 786, 126, card)
    text("FROM VIDEO TO CONTEXT", 45, 575, 0.43, muted)
    last_change = state["last_trigger_time"]
    change_label = _clock(last_change) if last_change is not None else "--"
    text(f"LAST VISUAL CHANGE  {change_label}", 586, 575, 0.38, muted)
    text("01", 45, 620, 0.68, teal, 2)
    text("Live scene", 80, 620, 0.55, white)
    text(">", 244, 620, 0.57, muted)
    text("02", 279, 620, 0.68, teal, 2)
    text("Visual change", 316, 620, 0.55, white)
    text(">", 500, 620, 0.57, muted)
    text("03", 536, 620, 0.68, teal, 2)
    text("Context update", 574, 620, 0.55, white)
    text("The context updates when the system notices a visual change.", 45, 650, 0.42, muted)

    # Current context: clearly identify if the text was generated or reused.
    px, pw = 830, 426
    _rounded_box(cv2, canvas, px, 88, pw, 256, card)
    text("CURRENT CONTEXT", px + 22, 120, 0.46, muted)
    description = state["task"] or "Waiting for the first observation"
    for index, item in enumerate(wrap_display_text(cv2, description, 378, 0.67, 3)):
        text(item, px + 22, 156 + index * 26, 0.67, white, 1)
    line(px + 22, 239, px + pw - 22, 239)

    source = state["last_source"]
    if source == "vlm":
        source_text, source_color = "GENERATED BY AI", teal
    elif source == "cache":
        source_text, source_color = "REUSED FROM MEMORY", amber
    else:
        source_text, source_color = "WAITING FOR A DESCRIPTION", muted
    cv2.circle(canvas, (px + 27, 263), 4, source_color, -1, cv2.LINE_AA)
    text(source_text, px + 40, 268, 0.42, source_color)
    if state["answer_time"] is not None:
        age = max(0, elapsed - state["answer_time"])
        text(f"Frame captured {age:.1f}s ago", px + 22, 296, 0.46, muted)
    else:
        text("The first observation is being prepared", px + 22, 296, 0.42, muted)
    if state["busy"]:
        text("Checking the scene...", px + 22, 322, 0.42, teal)
    else:
        text("Watching for visual changes", px + 22, 322, 0.42, muted)

    # The detail view replaces the history card, leaving camera and context in place.
    _rounded_box(cv2, canvas, px, 360, pw, 312, card)
    if show_details:
        text("DETAILS", px + 22, 390, 0.5, white, 1)
        text("D  BACK TO PRESENTATION", px + 213, 390, 0.35, muted)
        line(px + 22, 402, px + pw - 22, 402)
        distances = state["last_distances"]
        distance_text = (f"{distances['person']:.3f} / {distances['background']:.3f}"
                         if distances else "--")
        match = state["last_match"] or {}
        best = match.get("best_similarity")
        gap = match.get("margin")
        score_text = f"{best:.3f}" if best is not None else "--"
        gap_text = f"{gap:.3f}" if gap is not None else "--"
        resolve = state["last_resolution_ms"]
        resolve_text = f"{resolve:.0f} ms" if resolve is not None else "--"
        details = [
            ("Samples / triggers", f"{state['samples']} / {state['triggers']}"),
            ("Person / background distance", distance_text),
            ("Last trigger region", state["last_trigger"] or "--"),
            ("VLM calls / memory reuses", f"{state['vlm_calls']} / {state['cache_hits']}"),
            ("Descriptions stored", str(state["cache_size"])),
            ("Best similarity / gap", f"{score_text} / {gap_text}"),
            ("Match decision", match.get("reason", "--")),
            ("Resolve time after gate", resolve_text),
            ("Queued frames replaced", str(state["dropped"])),
            ("Gate / match / gap thresholds", f"{state['gate_threshold']:.2f} / {state['min_similarity']:.2f} / {state['min_margin']:.2f}"),
        ]
        for index, (label, value) in enumerate(details):
            y = 424 + index * 23
            text(label, px + 22, y, 0.42, muted)
            text(value, px + 257, y, 0.42, white)
    else:
        text("RECENT OBSERVATIONS", px + 22, 390, 0.5, white)
        text("LATEST 3", px + 326, 390, 0.36, muted)
        line(px + 22, 402, px + pw - 22, 402)
        recent = state["timeline"][-3:]
        if not recent:
            text("Observations will appear here", px + 22, 439, 0.46, muted)
        for index, event in enumerate(recent):
            y = 430 + index * 57
            text(_clock(event["time_seconds"]), px + 22, y, 0.43, white)
            event_source = "AI" if event["source"] == "vlm" else "Memory"
            event_color = teal if event["source"] == "vlm" else amber
            text(event_source, px + 22, y + 18, 0.36, event_color)
            for line_index, item in enumerate(wrap_display_text(cv2, event["task"], 302, 0.48, 2)):
                text(item, px + 100, y + line_index * 18, 0.48, white)
        line(px + 22, 607, px + pw - 22, 607)
        text("AI descriptions", px + 22, 628, 0.4, muted)
        text(str(state["vlm_calls"]), px + 22, 655, 0.64, white, 2)
        text("Memory reuses", px + 227, 628, 0.4, muted)
        text(str(state["cache_hits"]), px + 227, 655, 0.64, white, 2)

    text("D  Details", 25, 701, 0.4, muted)
    text("Q  Quit", 123, 701, 0.4, muted)
    text("Descriptions are model observations from individual frames.", 830, 701, 0.38, muted)
    return canvas
