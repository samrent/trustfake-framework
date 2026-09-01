"""TB-E3 validity gates G1-G3 (spec: an arm that fails a gate is fixed, not trained).

G1  metadata baseline: a headers-only classifier (aspect, resolution, JPEG
    quality -- no pixels) must score detection AUROC < 0.55 per environment
    on the arm's training set. CASIA's ~0.92 metadata AUC is the cautionary
    tale; SID-Set's squares-are-fake artifact is the local one.

G2  leakage: no near-duplicates between a training arm and any frozen
    evaluation leg, measured as max cosine in the embedding space of the
    frozen instrument. The threshold is a recorded protocol constant, and
    exact duplicates sit at ~1.0 by construction (same bytes, same
    features). Identity overlap in the face sense is handled upstream by
    the shortlist (no FF++-derived content, FakeClue excluded).

G3  cell minimums: every (class x environment) cell meets a recorded
    minimum; undersized cells are flagged so per-class metrics on them are
    reported with n rather than headlined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trustfake.logging import get_logger

logger = get_logger("curation.gates")

__all__ = [
    "G1_MAX_AUROC",
    "G2_COSINE_THRESHOLD",
    "G3_MIN_CELL",
    "g1_metadata_auroc",
    "g2_max_cosine",
    "g3_cell_counts",
]

#: G1 pass line, from the spec.
G1_MAX_AUROC = 0.55

#: G2 near-duplicate line. 0.95 max-cosine on frozen B/16 embeddings is well
#: above same-scene/different-photo pairs (burst shots in device data sit
#: lower) and well below only for genuinely shared content; exact dups are
#: ~1.0. Recorded here as the protocol constant the spec requires.
G2_COSINE_THRESHOLD = 0.95

#: G3 minimum rows per (class x environment) cell in a training arm.
G3_MIN_CELL = 200


def g1_metadata_auroc(index: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """Headers-only detection AUROC per environment of a training index.

    Features: aspect ratio, log-area, width, height, squareness, JPEG
    quality (NaN where not JPEG -- HistGradientBoosting consumes NaN
    natively) and a JPEG indicator. 5-fold out-of-fold AUROC of
    ``label_bin``; an environment with a single class reports NaN.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    results: dict[str, float] = {}
    for env, group in index.groupby("dataset"):
        y = group["label_bin"].to_numpy()
        if len(np.unique(y)) < 2 or len(group) < 50:
            results[str(env)] = float("nan")
            continue
        w = group["width"].to_numpy(dtype=np.float64)
        h = group["height"].to_numpy(dtype=np.float64)
        q = group["jpeg_q"].to_numpy(dtype=np.float64)
        x = np.column_stack(
            [
                w,
                h,
                w / h,
                np.log(w * h),
                (w == h).astype(np.float64),
                q,
                np.isnan(q).astype(np.float64),
            ]
        )
        model = HistGradientBoostingClassifier(random_state=seed)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probabilities = cross_val_predict(model, x, y, cv=folds, method="predict_proba")
        results[str(env)] = float(roc_auc_score(y, probabilities[:, 1]))
    return results


def g2_max_cosine(
    train_features: np.ndarray,
    leg_features: np.ndarray,
    device: str = "cuda",
    chunk: int = 8192,
) -> np.ndarray:
    """Max cosine of every training row against a leg's features.

    Features are L2-normalized in the cache, so cosine is a matmul; chunked
    on the GPU because the pools are hundreds of thousands of rows.
    """
    torch_device = torch.device(device)
    legs = torch.from_numpy(leg_features).to(torch_device, torch.float16)
    best = np.empty(train_features.shape[0], dtype=np.float32)
    for start in range(0, train_features.shape[0], chunk):
        block = torch.from_numpy(train_features[start : start + chunk]).to(
            torch_device, torch.float16
        )
        best[start : start + chunk] = (
            (block @ legs.T).max(dim=1).values.float().cpu().numpy()
        )
    return best


def g3_cell_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Rows per (class x environment), with the undersized flag."""
    counts = (
        index.groupby(["dataset", "label3"]).size().rename("n").reset_index()
    )
    counts["undersized"] = counts["n"] < G3_MIN_CELL
    return counts
