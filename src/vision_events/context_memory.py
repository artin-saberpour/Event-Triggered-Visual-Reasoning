"""Session-local CLIP text embeddings for cautiously reusing descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _unit_vector(value: object) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("CLIP embeddings must be finite, nonempty vectors")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("CLIP embedding norm must be finite and nonzero")
    return vector / norm


@dataclass(frozen=True)
class Match:
    description: str | None
    best_similarity: float | None
    margin: float | None
    reason: str
    candidate: str | None = None


class ContextMemory:
    """Keep one text embedding per distinct VLM description for this run."""

    def __init__(self, min_similarity: float = 0.30, min_margin: float = 0.04) -> None:
        if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("--min-similarity must be between -1 and 1")
        if not math.isfinite(min_margin) or not 0 <= min_margin <= 2:
            raise ValueError("--min-margin must be between 0 and 2")
        self.min_similarity, self.min_margin = min_similarity, min_margin
        self._entries: list[tuple[str, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, description: str, embedding: object) -> bool:
        description = " ".join(description.split())
        if not description:
            raise ValueError("Cannot cache an empty description")
        if any(existing.casefold() == description.casefold() for existing, _ in self._entries):
            return False
        vector = _unit_vector(embedding)
        if self._entries and vector.shape != self._entries[0][1].shape:
            raise ValueError("CLIP text embedding dimensions changed")
        self._entries.append((description, vector))
        return True

    def match(self, image_embedding: object) -> Match:
        if not self._entries:
            return Match(None, None, None, "empty")
        vector = _unit_vector(image_embedding)
        if vector.shape != self._entries[0][1].shape:
            raise ValueError("CLIP image and text embedding dimensions do not match")
        scores = np.array([np.clip(np.dot(vector, entry), -1.0, 1.0) for _, entry in self._entries])
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best = float(scores[best_index])
        candidate = self._entries[best_index][0]
        margin = best - float(scores[int(order[1])]) if len(order) > 1 else None
        # With only one candidate there is no runner-up margin. Seed at least
        # two descriptions before trusting a match to reduce premature reuse.
        if margin is None:
            return Match(None, best, None, "need_more_examples", candidate)
        if best < self.min_similarity:
            return Match(None, best, margin, "low_similarity", candidate)
        if margin < self.min_margin:
            return Match(None, best, margin, "ambiguous", candidate)
        return Match(candidate, best, margin, "matched", candidate)
