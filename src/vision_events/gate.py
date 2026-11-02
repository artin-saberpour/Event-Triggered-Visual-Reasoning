"""A testable, model-independent gate for region-aware change detection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

REGIONS = ("full", "person", "background")


@dataclass(frozen=True)
class Decision:
    region: str | None
    distances: dict[str, float]
    baseline: bool = False


def _normalize(value: object) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("Embeddings must be nonempty, finite, one-dimensional vectors")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding norm must be finite and nonzero")
    return vector / norm


class RegionGate:
    """Compare embeddings to the last reference used for the affected region.

    The first observation sets a baseline without emitting an event. If both
    regions change, a full-frame event refreshes both region references too.
    This fixes repeated triggering present in the archived prototype.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        if not math.isfinite(threshold) or not 0 <= threshold <= 2:
            raise ValueError("Cosine distance threshold must be between 0 and 2")
        self.threshold = threshold
        self._references: dict[str, np.ndarray] | None = None

    def observe(self, embeddings: Mapping[str, object]) -> Decision:
        if set(embeddings) != set(REGIONS):
            raise ValueError("Exactly full, person, and background embeddings are required")
        vectors = {region: _normalize(embeddings[region]) for region in REGIONS}
        if len({value.shape for value in vectors.values()}) != 1:
            raise ValueError("All region embeddings must have the same dimensions")
        if self._references is None:
            self._references = vectors
            return Decision(None, {}, baseline=True)
        if any(vectors[key].shape != self._references[key].shape for key in REGIONS):
            raise ValueError("Embedding dimensions changed since the baseline")
        distances = {
            key: float(np.clip(1.0 - np.dot(vectors[key], self._references[key]), 0.0, 2.0))
            for key in REGIONS
        }
        changed_person = distances["person"] > self.threshold
        changed_background = distances["background"] > self.threshold
        if changed_person and changed_background:
            region = "full"
            self._references = vectors
        elif changed_person:
            region = "person"
            self._references["person"] = vectors["person"]
            self._references["full"] = vectors["full"]
        elif changed_background:
            region = "background"
            self._references["background"] = vectors["background"]
            self._references["full"] = vectors["full"]
        else:
            region = None
        return Decision(region, distances)
