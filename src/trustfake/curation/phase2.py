"""TB-E3 phase 2 (H4): training objectives on the winning curation.

{pooled ERM, GroupDRO, V-REx} on the winning arm's features, same head and
recipe as every ladder fit; groups are ENVIRONMENTS (the dataset column).
Pooled ERM is the ladder's own C3a fits — not re-run. The H4 rule, verbatim
from the spec: an invariance objective is adopted only if it beats pooled
ERM on L3+L4 without losing > 0.01 on L1. Curation table and objective
table are separate tables.

Objective constants (recorded before the fits ran): GroupDRO step size
eta = 0.01 (Sagawa et al. 2020's default regime); V-REx penalty beta = 10.
Both consume the same batches as ERM; only the loss changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from trustfake.curation.fit import BATCH, EPOCHS, LR, VAL_FRACTION, _macro_f1
from trustfake.logging import get_logger

logger = get_logger("curation.phase2")

__all__ = ["fit_probe_grouped"]

GROUPDRO_ETA = 0.01
VREX_BETA = 10.0


def fit_probe_grouped(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    objective: str,
    num_classes: int = 3,
    device: str = "cuda",
) -> dict[str, Any]:
    """GroupDRO / V-REx fit; mirrors fit_probe's carve, init and selection."""
    if objective not in ("groupdro", "vrex"):
        raise ValueError(f"objective must be groupdro|vrex, got {objective}")
    generator = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    order = torch.randperm(n, generator=generator).numpy()
    n_val = max(1, int(round(n * VAL_FRACTION)))
    val_rows, fit_rows = order[:n_val], order[n_val:]

    torch_device = torch.device(device)
    x_all = torch.from_numpy(features).to(torch_device, torch.float32)
    y_all = torch.from_numpy(labels).to(torch_device, torch.long)
    unique_groups, group_ids = np.unique(groups, return_inverse=True)
    g_all = torch.from_numpy(group_ids).to(torch_device)
    fit_idx = torch.from_numpy(fit_rows).to(torch_device)
    x_val = x_all[torch.from_numpy(val_rows).to(torch_device)]
    y_val = y_all[torch.from_numpy(val_rows).to(torch_device)]

    head = torch.nn.Linear(features.shape[1], num_classes).to(torch_device)
    torch.manual_seed(seed)
    torch.nn.init.normal_(head.weight, std=0.01)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)

    n_groups = len(unique_groups)
    dro_weights = torch.ones(n_groups, device=torch_device) / n_groups

    epoch_generator = torch.Generator().manual_seed(seed + 1)
    best = {"f1": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(EPOCHS):
        head.train()
        perm = fit_idx[torch.randperm(len(fit_idx), generator=epoch_generator)]
        for start in range(0, len(perm), BATCH):
            rows = perm[start : start + BATCH]
            logits = head(x_all[rows])
            losses = F.cross_entropy(logits, y_all[rows], reduction="none")
            batch_groups = g_all[rows]
            present = torch.unique(batch_groups)
            group_losses = torch.stack(
                [losses[batch_groups == g].mean() for g in present]
            )
            if objective == "groupdro":
                with torch.no_grad():
                    dro_weights[present] = dro_weights[present] * torch.exp(
                        GROUPDRO_ETA * group_losses.detach()
                    )
                    dro_weights /= dro_weights.sum()
                loss = (dro_weights[present] * group_losses).sum() / (
                    dro_weights[present].sum() + 1e-12
                )
            else:  # vrex
                loss = group_losses.mean()
                if len(present) > 1:
                    loss = loss + VREX_BETA * group_losses.var(unbiased=False)
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
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                },
            }

    logger.info(
        f"{objective} seed={seed} n={n} groups={n_groups} best epoch "
        f"{best['epoch']} val f1 {best['f1']:.4f}"
    )
    return {
        "state": best["state"],
        "best_epoch": best["epoch"],
        "val_f1": best["f1"],
        "history": history,
        "objective": objective,
        "groups": [str(g) for g in unique_groups],
        "seed": seed,
    }
