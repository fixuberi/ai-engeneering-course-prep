"""Cosine similarity: dot(a, b) / (norm(a) * norm(b))."""

import numpy as np


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two equal-length, non-zero vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Cosine ignores vector length, so rescale to max |x| = 1 first:
    # this keeps the squares inside norm() from overflowing or underflowing.
    a = a / np.max(np.abs(a))
    b = b / np.max(np.abs(b))
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(result, -1.0, 1.0))
