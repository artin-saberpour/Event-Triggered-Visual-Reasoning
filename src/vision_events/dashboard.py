"""Presentation and diagnostic views for the live webcam demo."""

from __future__ import annotations

import math
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
    white, muted = (244, 243, 240), (177, 165, 154)
    person_color, background_color = (207, 226, 91), (145, 198, 245)
    grid, threshold_color = (65, 60, 56), (170, 166, 160)
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

    cv2.rectangle(canvas, (left, top), (right, bottom), (25, 23, 22), -1)
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
    # Stems mark actual trigger times; the curves remain disconnected after a reset.
    for item in samples:
        affected = item["trigger_region"]
        if affected is not None:
            values = ([item["person"], item["background"]] if affected == "full"
                      else [item[affected]])
            marker_x = sx(item["time_seconds"])
            marker_y = min(sy(value) for value in values)
            cv2.line(canvas, (marker_x, min(marker_y + 6, bottom)), (marker_x, bottom),
                     (86, 80, 74), 1, cv2.LINE_AA)
    for region, color in (("person", person_color), ("background", background_color)):
        previous = None
        for item in samples:
            point = (sx(item["time_seconds"]), sy(item[region]))
            # Do not connect across long gaps or a refreshed region reference.
            if (previous is not None and item["time_seconds"] - previous[0] <= 4.0
                    and previous[2] not in (region, "full")):
                cv2.line(canvas, previous[1], point, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
            if item["trigger_region"] in (region, "full"):
                cv2.circle(canvas, point, 5, white, 1, cv2.LINE_AA)
            previous = (item["time_seconds"], point, item["trigger_region"])



def render_dashboard(cv2: Any, frame: np.ndarray, state: dict[str, Any], elapsed: float,
                     show_details: bool = False) -> np.ndarray:
    """Render the camera, context and all pipeline stages on the main view."""
    height, width = 1080, 1920
    background, card = (19, 17, 16), (34, 30, 27)
    white, muted = (244, 243, 240), (177, 165, 154)
    teal, amber, divider = (207, 226, 91), (145, 198, 245), (62, 56, 50)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)

    def text(value: str, x: int, y: int, scale: float = 0.5,
             color: tuple[int, int, int] = white, thickness: int = 1) -> None:
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def line(x1: int, y1: int, x2: int, y2: int) -> None:
        cv2.line(canvas, (x1, y1), (x2, y2), divider, 1, cv2.LINE_AA)

    def place_view(image: np.ndarray | None, x: int, y: int, w: int, h: int) -> None:
        canvas[y:y + h, x:x + w] = (10, 12, 16)
        if image is None:
            text("Awaiting frame", x + 16, y + h // 2, 0.43, muted)
            return
        source_height, source_width = image.shape[:2]
        scale = min(w / source_width, h / source_height)
        image_w = max(1, int(source_width * scale))
        image_h = max(1, int(source_height * scale))
        view = cv2.resize(image, (image_w, image_h))
        x0, y0 = x + (w - image_w) // 2, y + (h - image_h) // 2
        canvas[y0:y0 + image_h, x0:x0 + image_w] = view

    text("EVENT-TRIGGERED VISUAL REASONING", 25, 38, 0.76, white, 2)
    text("LIVE PERCEPTION DEMO", 26, 63, 0.43, muted)
    cv2.circle(canvas, (1760, 37), 5, teal, -1, cv2.LINE_AA)
    text("LIVE", 1774, 42, 0.48, teal)
    text(_clock(elapsed), 1848, 42, 0.5, muted)

    # The live camera is independent of the most recently processed frame.
    vx, vy, vw, vh = 24, 88, 880, 495
    place_view(frame, vx, vy, vw, vh)
    _rounded_box(cv2, canvas, vx + 16, vy + 16, 150, 32, background, 8)
    cv2.circle(canvas, (vx + 31, vy + 32), 4, teal, -1, cv2.LINE_AA)
    text("LIVE CAMERA", vx + 43, vy + 37, 0.45, white)

    px, pw = 924, 972
    _rounded_box(cv2, canvas, px, 88, pw, 230, card)
    text("CURRENT CONTEXT", px + 22, 120, 0.48, muted)
    description = state["task"] or "Waiting for the first observation"
    for index, item in enumerate(wrap_display_text(cv2, description, pw - 49, 0.70, 3)):
        text(item, px + 22, 156 + index * 26, 0.70, white)
    line(px + 22, 239, px + pw - 22, 239)
    source = state["last_source"]
    if source == "vlm":
        source_text, source_color = "GENERATED BY LLaVA", teal
    elif source == "cache":
        source_text, source_color = "REUSED FROM CLIP MEMORY", amber
    else:
        source_text, source_color = "WAITING FOR A DESCRIPTION", muted
    cv2.circle(canvas, (px + 27, 263), 4, source_color, -1, cv2.LINE_AA)
    text(source_text, px + 40, 268, 0.45, source_color)
    if state["answer_time"] is not None:
        age = max(0, elapsed - state["answer_time"])
        text(f"Frame captured {age:.1f}s ago", px + 22, 296, 0.46, muted)
    else:
        text("The first observation is being prepared", px + 22, 296, 0.43, muted)
    status = "PROCESSING" if state["busy"] else "MONITORING"
    text(status, px + pw - 142, 296, 0.40, teal if state["busy"] else muted)

    _rounded_box(cv2, canvas, px, 334, pw, 249, card)
    if show_details:
        text("EXPANDED DETAILS", px + 22, 365, 0.53, white)
        text("D  BACK TO HISTORY", px + pw - 164, 365, 0.39, muted)
        line(px + 22, 376, px + pw - 22, 376)
        match = state["last_match"] or {}
        candidate = match.get("best_description") or "No candidate yet"
        text("BEST CACHED CANDIDATE", px + 22, 401, 0.42, muted)
        for index, item in enumerate(wrap_display_text(cv2, candidate, pw - 52, 0.48, 2)):
            text(item, px + 22, 426 + index * 20, 0.48, white)
        best = match.get("best_similarity")
        gap = match.get("margin")
        best_str = f"{best:.3f}" if best is not None else "--"
        gap_str = f"{gap:.3f}" if gap is not None else "--"
        text(f"Similarity: {best_str}   Gap: {gap_str}   Decision: {match.get('reason', '--')}",
             px + 22, 491, 0.46, muted)
        text(f"Samples: {state['samples']}   Triggers: {state['triggers']}   Queued replaced: {state['dropped']}",
             px + 22, 522, 0.46, muted)
        text("Timings are measured after the change gate.", px + 22, 555, 0.42, muted)
    else:
        text("RECENT OBSERVATIONS", px + 22, 365, 0.53, white)
        text("LATEST 3", px + pw - 110, 365, 0.4, muted)
        line(px + 22, 376, px + pw - 22, 376)
        recent = state["timeline"][-3:]
        if not recent:
            text("Observations will appear here", px + 22, 411, 0.48, muted)
        for index, event in enumerate(recent):
            y = 403 + index * 53
            text(_clock(event["time_seconds"]), px + 22, y, 0.45, white)
            event_source = "LLaVA" if event["source"] == "vlm" else "Memory"
            event_color = teal if event["source"] == "vlm" else amber
            text(event_source, px + 22, y + 19, 0.39, event_color)
            for line_index, item in enumerate(wrap_display_text(cv2, event["task"], pw - 142, 0.54, 2)):
                text(item, px + 106, y + line_index * 19, 0.54, white)

    # All components and the former diagnostic measurements remain visible.
    top, card_h = 599, 249
    left, mid, right = 24, 644, 1264
    left_w, mid_w, right_w = 600, 600, 632
    for x, panel_width in ((left, left_w), (mid, mid_w), (right, right_w)):
        _rounded_box(cv2, canvas, x, top, panel_width, card_h, card)

    text("01", left + 20, 629, 0.65, teal, 2)
    text("MediaPipe", left + 63, 629, 0.68, white, 2)
    text("Separate person and background", left + 20, 653, 0.52, muted)
    text("PERSON VIEW", left + 20, 678, 0.43, muted)
    text("BACKGROUND VIEW", left + 306, 678, 0.43, muted)
    previews = state["region_previews"]
    place_view(previews.get("person"), left + 20, 687, 274, 127)
    place_view(previews.get("background"), left + 306, 687, 274, 127)
    preview_time = state["preview_time"]
    preview_label = _clock(preview_time) if preview_time is not None else "--"
    text(f"Sample captured at: {preview_label}", left + 20, 837, 0.43, muted)

    text("02", mid + 20, 629, 0.65, teal, 2)
    text("CLIP", mid + 63, 629, 0.68, white, 2)
    text("Detect visual change and match memory", mid + 20, 653, 0.52, muted)
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
        y = 682 + index * 30
        text(label, mid + 20, y, 0.50, muted)
        text(value, mid + 345, y, 0.50, white)

    text("03", right + 20, 629, 0.65, teal, 2)
    text("LLaVA", right + 63, 629, 0.68, white, 2)
    text("Describe when no memory matches", right + 20, 653, 0.52, muted)
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
        y = 682 + index * 30
        text(label, right + 20, y, 0.50, muted)
        text(value, right + 345, y, 0.50, white)

    # A full-width chart keeps the trace readable without hiding model details.
    _rounded_box(cv2, canvas, 24, 858, 1872, 181, card)
    text("CLIP CHANGE OVER TIME", 45, 882, 0.58, white, 1)
    text("Cosine distance to each region reference", 333, 882, 0.43, muted)
    cv2.line(canvas, (810, 876), (834, 876), teal, 2, cv2.LINE_AA)
    text("Person", 845, 882, 0.45, teal)
    cv2.line(canvas, (932, 876), (956, 876), amber, 2, cv2.LINE_AA)
    text("Background", 967, 882, 0.45, amber)
    text(f"Threshold {state['gate_threshold']:.2f}", 1126, 882, 0.45, muted)
    cv2.circle(canvas, (1327, 876), 5, white, 1, cv2.LINE_AA)
    text("Trigger / reference refresh", 1339, 882, 0.45, muted)
    draw_clip_chart(cv2, canvas, state["distance_history"], elapsed, state["gate_threshold"],
                    42, 889, 1836, 144)

    text("D  Expanded details", 25, 1065, 0.42, muted)
    text("Q  Quit", 190, 1065, 0.42, muted)
    text("Line breaks: reference refreshed or samples unavailable", 410, 1065, 0.42, muted)
    text("Contexts are inferred from individual frames.", 1450, 1065, 0.40, muted)
    return canvas
