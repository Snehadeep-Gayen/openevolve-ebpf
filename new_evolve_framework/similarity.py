"""
Similarity helpers for idea embeddings.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from openevolve.embedding import EmbeddingClient

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity; return 0.0 if shapes mismatch."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    arr1 = np.array(vec1, dtype=np.float32)
    arr2 = np.array(vec2, dtype=np.float32)
    denom = (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    if denom == 0:
        return 0.0
    return float(np.dot(arr1, arr2) / denom)


class IdeaSimilarity:
    """Embedding-backed similarity checks."""

    def __init__(self, model_name: str = "text-embedding-3-small", threshold: float = 0.92):
        self.client = EmbeddingClient(model_name)
        self.threshold = threshold

    def embed(self, text: str) -> List[float]:
        try:
            emb = self.client.get_embedding(text)
            return emb or []
        except Exception as exc:  # pragma: no cover
            logger.warning("Embedding failed: %s", exc)
            return []

    def is_too_similar(self, candidate_vec: List[float], existing_vec: Optional[List[float]]) -> bool:
        if not existing_vec:
            return False
        score = cosine_similarity(candidate_vec, existing_vec)
        return score >= self.threshold
