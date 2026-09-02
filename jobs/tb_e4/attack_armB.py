"""8/255 battery on TB-E4 Arm B (full fine-tuned CLIPProbeClassifier)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from trustfake.attacks.pgd import PGD
from trustfake.attacks.query_confidence import QueryConfidence
from trustfake.curation.fit import _auroc
from trustfake.data.sid_set import SIDSetDataModule
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.torch.clip import clip_probe
from trustfake.models.wrapper.base import BaseWrapper

OUT = Path(os.environ["OUTPUT_PATH"]) / "tb_e4"
EPS = 8 / 255


def main():
    device = torch.device("cuda")
    model = clip_probe(num_classes=3, model_name="ViT-B-16",
                       pretrained="laion2b_s34b_b88k", freeze_backbone=False)
    model.load_state_dict(torch.load(OUT / "armB/best.pt", weights_only=True))
    wrapper = BaseWrapper(
        normalization_layer=torch.nn.Identity(), model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(), temperature=1.0,
    ).to(device).eval()

    datamodule = SIDSetDataModule(
        data_dir=os.path.join(os.environ["DATA_PATH"], "sid_set"),
        profile="train", limit_test=1000, batch_size=32, num_workers=6,
        normalization_layer=None,
    )
    datamodule.setup()
    pgd = PGD(eps=EPS, steps=40)
    quc = QueryConfidence(eps=EPS, n_queries=400, direction="under")

    records = {"clean": [], "pgd": [], "quc": []}
    for images, labels in datamodule.test_dataloader():
        images = images.to(device)
        with torch.no_grad():
            _, _, preds, unc = wrapper(images)
        records["clean"].append((preds.cpu().numpy(), unc.cpu().numpy(),
                                 labels.numpy()))
        adv = pgd(wrapper, images)
        with torch.no_grad():
            _, _, p, u = wrapper(adv)
        records["pgd"].append((p.cpu().numpy(), u.cpu().numpy(), labels.numpy()))
        adv = quc(wrapper, images)
        with torch.no_grad():
            _, _, p, u = wrapper(adv)
        records["quc"].append((p.cpu().numpy(), u.cpu().numpy(), labels.numpy()))

    clean_preds = np.concatenate([r[0] for r in records["clean"]])
    summary = {}
    for key, rows in records.items():
        preds = np.concatenate([r[0] for r in rows])
        unc = np.concatenate([r[1] for r in rows])
        labels = np.concatenate([r[2] for r in rows])
        summary[key] = {
            "accuracy_top1": float((preds == labels).mean()),
            "fd_auroc": _auroc(unc, preds != labels),
            "argmax_preserved": float((preds == clean_preds).mean()),
        }
    summary["protocol"] = {"eps": "8/255", "pgd_steps": 40, "queries": 400,
                           "model": "TB-E4 Arm B best (epoch 4)",
                           "split": "SID test prefix 1000"}
    (OUT / "armB/attacks.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("ATTACKS_ARMB_COMPLETE")


if __name__ == "__main__":
    main()
