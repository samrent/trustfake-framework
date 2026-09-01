"""8/255 attack battery on the TB-E3 winning heads (project threat model).

The winning C3a heads load into `CLIPProbeClassifier` (standard frozen
B/16), wrapped in `BaseWrapper`, and face the existing battery on the
SID-Set test prefix (limit_test protocol, 1000 rows, resize mode):

  * PGD-40 at eps=8/255 (white-box evasion): accuracy clean vs attacked.
  * query_underconf at eps=8/255, 400 queries (the confidence attack that
    took the undefended B/32 probe's Phi to chance in TB-E2): fd_auroc
    clean vs attacked, argmax-preservation check.

T=1.0 (temperature is monotone; both the ranking metrics and the attack
objectives are unaffected). Results: $OUTPUT_PATH/tb_e3/hardening/attacks_*.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from trustfake.attacks.pgd import PGD
from trustfake.attacks.query_confidence import QueryConfidence
from trustfake.curation.fit import _auroc
from trustfake.curation.ladder import _out
from trustfake.data.sid_set import SIDSetDataModule
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.torch.clip import clip_probe
from trustfake.models.wrapper.base import BaseWrapper

EPS = 8 / 255
LIMIT_TEST = 1000
BATCH = 32


def build_wrapper(head_state: dict, device: torch.device) -> BaseWrapper:
    probe = clip_probe(
        num_classes=3, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    probe.head.load_state_dict(
        {k.removeprefix("head."): v for k, v in head_state.items()}
    )
    wrapper = BaseWrapper(
        normalization_layer=torch.nn.Identity(),  # CLIP normalizes internally
        model=probe,
        loss_fn=torch.nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
        temperature=1.0,
    )
    return wrapper.to(device).eval()


def main() -> None:
    device = torch.device("cuda")
    out = _out(os.environ["OUTPUT_PATH"]) / "hardening"
    out.mkdir(parents=True, exist_ok=True)

    datamodule = SIDSetDataModule(
        data_dir=os.path.join(os.environ["DATA_PATH"], "sid_set"),
        profile="train",
        limit_test=LIMIT_TEST,
        batch_size=BATCH,
        num_workers=6,
        normalization_layer=None,
    )
    datamodule.setup()
    loader = datamodule.test_dataloader()

    head_file = _out(os.environ["OUTPUT_PATH"]) / "fits" / "C3a_natural_s1.pt"
    wrapper = build_wrapper(torch.load(head_file, weights_only=True), device)

    pgd = PGD(eps=EPS, steps=40)
    quc = QueryConfidence(eps=EPS, n_queries=400, direction="under")

    records = {"clean": [], "pgd": [], "quc": []}
    for images, labels in loader:
        images = images.to(device)
        with torch.no_grad():
            _, _, preds, uncertainty = wrapper(images)
        records["clean"].append(
            (preds.cpu().numpy(), uncertainty.cpu().numpy(), labels.numpy())
        )
        adv = pgd(wrapper, images)
        with torch.no_grad():
            _, _, preds_a, uncertainty_a = wrapper(adv)
        records["pgd"].append(
            (preds_a.cpu().numpy(), uncertainty_a.cpu().numpy(), labels.numpy())
        )
        adv_q = quc(wrapper, images)
        with torch.no_grad():
            _, _, preds_q, uncertainty_q = wrapper(adv_q)
        records["quc"].append(
            (preds_q.cpu().numpy(), uncertainty_q.cpu().numpy(), labels.numpy())
        )

    summary = {}
    clean_preds = np.concatenate([r[0] for r in records["clean"]])
    for key, rows in records.items():
        preds = np.concatenate([r[0] for r in rows])
        uncertainty = np.concatenate([r[1] for r in rows])
        labels = np.concatenate([r[2] for r in rows])
        errors = preds != labels
        summary[key] = {
            "accuracy_top1": float((preds == labels).mean()),
            "fd_auroc": _auroc(uncertainty, errors),
            "argmax_preserved": float((preds == clean_preds).mean()),
            "n": int(labels.size),
        }
    summary["protocol"] = {
        "eps": "8/255 (project threat model)",
        "pgd_steps": 40,
        "query_budget": 400,
        "head": "C3a natural s1",
        "split": "SID-Set test prefix 1000 (train profile, resize mode)",
        "temperature": 1.0,
    }
    (out / "attacks_c3a_s1.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("ATTACKS_COMPLETE")


if __name__ == "__main__":
    main()
