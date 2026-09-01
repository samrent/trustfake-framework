"""Probe fits on cached features -- the TB-E2 recipe, transposed.

The head and recipe are held fixed across every arm (spec instrument):
``nn.Linear(512, 3)`` (1,539 parameters) on L2-normalized frozen-B/16
features, CE loss, Adam lr=5e-4, batch 32, 8 epochs, a seeded 5% row-level
validation carve for model selection, best epoch kept by validation MACRO
F1 -- exactly the knobs `jobs/track_b_backbone_grid.sh` passed to
`src/train.py` (selection_metric=val_f1_score, dataloaders.batch_size=32,
trainer.max_epochs=8, val_fraction=0.05), so a TB-E3 fit differs from a
TB-E2 fit only in where the features come from.

Metric semantics mirror `trustfake.metrics.evaluation` exactly:
``accuracy`` is MACRO (mean per-class recall), ``accuracy_top1`` plain
top-1, ``detection_auroc`` ranks 1 - P(real) with every non-real class
folded to fake (per-modality breakouts restrict rows to {real, class}),
``fd_auroc`` ranks errors by 1 - MSP, and undefined AUROCs are NaN, never
0.0. AURC comes from `aurc_from_scores` itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from trustfake.logging import get_logger
from trustfake.metrics.evaluation.selective_classification import aurc_from_scores

logger = get_logger("curation.fit")

__all__ = ["fit_probe", "evaluate_probe", "CLASS_NAMES"]

CLASS_NAMES = ("real", "synthetic", "tampered")

LR = 5e-4
BATCH = 32
EPOCHS = 8
VAL_FRACTION = 0.05


def _macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> float:
    scores = []
    for c in range(num_classes):
        tp = float(((preds == c) & (targets == c)).sum())
        fp = float(((preds == c) & (targets != c)).sum())
        fn = float(((preds != c) & (targets == c)).sum())
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Binary AUROC with the repo's NaN-when-undefined convention."""
    positives = positives.astype(bool)
    if (
        scores.size == 0
        or positives.all()
        or not positives.any()
        or np.min(scores) == np.max(scores)
    ):
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(positives, scores.astype(np.float64)))


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    num_classes: int = 3,
    device: str = "cuda",
    binary_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train the linear head on cached features; return weights + history.

    ``binary_mask`` marks rows trained with the BINARY objective only (arm
    C3b's multi-task mapping): those rows carry ``labels`` of -1 and
    contribute a real-vs-fake CE on p_real = softmax(logits)[real] versus
    1 - p_real, leaving the synthetic/tampered separation untouched. None
    (every other arm) trains plain 3-class CE.
    """
    generator = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    order = torch.randperm(n, generator=generator).numpy()
    n_val = max(1, int(round(n * VAL_FRACTION)))
    # Validation rows must have a 3-class label to select on macro F1;
    # binary-only rows are excluded from the carve, never from training.
    if binary_mask is not None:
        eligible = order[~binary_mask[order]]
        val_rows, fit_rows = eligible[:n_val], np.setdiff1d(order, eligible[:n_val])
    else:
        val_rows, fit_rows = order[:n_val], order[n_val:]

    torch_device = torch.device(device)
    x_all = torch.from_numpy(features).to(torch_device, torch.float32)
    y_all = torch.from_numpy(labels).to(torch_device, torch.long)
    fit_idx = torch.from_numpy(fit_rows).to(torch_device)
    x_val = x_all[torch.from_numpy(val_rows).to(torch_device)]
    y_val = y_all[torch.from_numpy(val_rows).to(torch_device)]
    binary = (
        torch.from_numpy(binary_mask).to(torch_device)
        if binary_mask is not None
        else None
    )

    head = torch.nn.Linear(features.shape[1], num_classes).to(torch_device)
    torch.manual_seed(seed)
    torch.nn.init.normal_(head.weight, std=0.01)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)

    epoch_generator = torch.Generator().manual_seed(seed + 1)
    best = {"f1": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(EPOCHS):
        head.train()
        perm = fit_idx[torch.randperm(len(fit_idx), generator=epoch_generator)]
        for start in range(0, len(perm), BATCH):
            rows = perm[start : start + BATCH]
            logits = head(x_all[rows])
            y = y_all[rows]
            if binary is not None:
                b = binary[rows]
                loss = torch.tensor(0.0, device=torch_device)
                if (~b).any():
                    loss = loss + F.cross_entropy(logits[~b], y[~b])
                if b.any():
                    p_real = torch.softmax(logits[b], dim=-1)[:, 0]
                    p_real = p_real.clamp(1e-7, 1 - 1e-7)
                    # binary-only rows are all fakes of unknown modality
                    loss = loss + F.binary_cross_entropy(
                        1 - p_real, torch.ones_like(p_real)
                    )
            else:
                loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_preds = head(x_val).argmax(dim=-1).cpu().numpy()
        f1 = _macro_f1(val_preds, y_val.cpu().numpy(), num_classes)
        history.append(round(f1, 6))
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "epoch": epoch,
                "state": {
                    k: v.detach().cpu().clone()
                    for k, v in head.state_dict().items()
                },
            }

    logger.info(
        f"fit seed={seed} n={n} best epoch {best['epoch']} val f1 {best['f1']:.4f} "
        f"history {history}"
    )
    return {
        "state": best["state"],
        "best_epoch": best["epoch"],
        "val_f1": best["f1"],
        "history": history,
        "n_fit": int(len(fit_rows)),
        "n_val": int(len(val_rows)),
        "seed": seed,
    }


def evaluate_probe(
    state: dict[str, torch.Tensor],
    features: np.ndarray,
    labels: np.ndarray,
    device: str = "cuda",
    num_classes: int = 3,
) -> dict[str, float]:
    """Leg metrics for one fitted head, repo semantics throughout."""
    torch_device = torch.device(device)
    head = torch.nn.Linear(features.shape[1], num_classes).to(torch_device)
    head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(features).to(torch_device, torch.float32))
        probs = torch.softmax(logits, dim=-1).double().cpu().numpy()

    y = labels.astype(np.int64)
    preds = probs.argmax(axis=1)
    errors = (preds != y).astype(np.float64)
    uncertainty = 1.0 - probs.max(axis=1)
    p_fake = 1.0 - probs[:, 0]
    fake = y != 0

    recalls = {
        f"recall_{CLASS_NAMES[c]}": (
            float((preds[y == c] == c).mean()) if (y == c).any() else float("nan")
        )
        for c in range(num_classes)
    }
    per_modality = {}
    for c in range(1, num_classes):
        keep = (y == 0) | (y == c)
        per_modality[f"detection_auroc_{CLASS_NAMES[c]}"] = _auroc(
            p_fake[keep], fake[keep]
        )

    return {
        "n": int(y.size),
        "accuracy": float(
            np.nanmean(
                [recalls[f"recall_{CLASS_NAMES[c]}"] for c in range(num_classes)]
            )
        ),
        "accuracy_top1": float((preds == y).mean()),
        "f1_score": _macro_f1(preds, y, num_classes),
        "detection_auroc": _auroc(p_fake, fake),
        **per_modality,
        "fd_auroc": _auroc(uncertainty, errors.astype(bool)),
        "aurc": float(aurc_from_scores(uncertainty, errors, weights="block")),
        **recalls,
    }
