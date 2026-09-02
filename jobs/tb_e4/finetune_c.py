"""TB-E4 Arm C: adversarial fine-tune (PGD-3, eps=8/255) from Arm B's best.

Registered 2026-09-02 (user-directed amendment): Madry-style training on
adversarial examples only, alpha = 2.5*eps/3, 2 epochs, backbone 5e-6 /
head 5e-4 cosine, flip-then-attack, clean-dev selection. Afterwards: clean
frozen-leg eval + the 8/255 battery, so the adoption rule can be applied
mechanically (PGD-40 acc >= 0.40 AND quc fd >= 0.65 AND shifted legs
within 0.05 of Arm B).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trustfake.curation.fit import _auroc
from trustfake.models.torch.clip import clip_probe

OUT = Path(os.environ["OUTPUT_PATH"])
PACK = OUT / "tb_e4" / "pixels"
RUN = OUT / "tb_e4" / "armC"
EPS = 8 / 255
STEPS = 3
ALPHA = 2.5 * EPS / STEPS
BATCH, EPOCHS, SEED = 48, 2, 1


def load_split(name):
    meta = json.loads((PACK / f"{name}.json").read_text())
    pixels = np.load(PACK / f"{name}.u8", mmap_mode="r")[: meta["n"]]
    labels = pd.read_parquet(PACK / f"{name}.parquet")["label3"].to_numpy()
    return pixels, labels


@torch.no_grad()
def evaluate(model, pixels, labels, device, batch=192):
    model.eval()
    probs = []
    for start in range(0, len(pixels), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(pixels[start:start + batch]))
        chunk = chunk.to(device).permute(0, 3, 1, 2).float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(chunk)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    y = labels.astype(np.int64)
    p_fake = 1.0 - probs[:, 0]
    preds = probs.argmax(1)
    out = {
        "detection_auroc": _auroc(p_fake, y != 0),
        "accuracy_top1": float((preds == y).mean()),
        "fd_auroc": _auroc(1 - probs.max(1), preds != y),
    }
    for c, cname in ((1, "synthetic"), (2, "tampered")):
        keep = (y == 0) | (y == c)
        out[f"detection_auroc_{cname}"] = _auroc(p_fake[keep], (y != 0)[keep])
    return out


def pgd_batch(model, chunk, targets):
    """PGD-3 crafted against the CURRENT model, fp32 grads on the input."""
    adv = chunk + torch.empty_like(chunk).uniform_(-EPS, EPS)
    adv = adv.clamp(0, 1)
    for _ in range(STEPS):
        adv = adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = torch.nn.functional.cross_entropy(model(adv), targets)
            grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + ALPHA * grad.sign()
        adv = chunk + (adv - chunk).clamp(-EPS, EPS)
        adv = adv.clamp(0, 1)
    return adv.detach()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(SEED)

    train_px, train_y = load_split("train")
    dev = {name: load_split(name) for name in ("dev_L3", "dev_L4")}

    model = clip_probe(num_classes=3, model_name="ViT-B-16",
                       pretrained="laion2b_s34b_b88k", freeze_backbone=False)
    model.load_state_dict(
        torch.load(OUT / "tb_e4/armB/best.pt", weights_only=True))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.visual.parameters(), "lr": 5e-6},
         {"params": model.head.parameters(), "lr": 5e-4}],
        weight_decay=0.05)
    steps_per_epoch = math.ceil(len(train_px) / BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * steps_per_epoch)

    start_epoch, best = 0, {"score": -1.0, "epoch": -1}
    last = RUN / "last.ckpt"
    if last.exists():
        state = torch.load(last, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        print(f"resumed at epoch {start_epoch}")

    y_all = torch.from_numpy(train_y.astype(np.int64))
    history = []
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        order = torch.randperm(
            len(train_px),
            generator=torch.Generator().manual_seed(SEED * 77 + epoch)).numpy()
        running = 0.0
        for step in range(steps_per_epoch):
            rows = np.sort(order[step * BATCH:(step + 1) * BATCH])
            chunk = torch.from_numpy(np.ascontiguousarray(train_px[rows]))
            chunk = chunk.to(device).permute(0, 3, 1, 2).float().div_(255)
            if torch.rand(1).item() < 0.5:
                chunk = torch.flip(chunk, dims=[3])
            targets = y_all[rows].to(device)
            adv = pgd_batch(model, chunk, targets)
            loss = torch.nn.functional.cross_entropy(model(adv), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running += loss.item()
            if step % 250 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} "
                      f"adv-loss {running / (step + 1):.4f}", flush=True)

        scores = {n: evaluate(model, px, y, device) for n, (px, y) in dev.items()}
        mean_dev = float(np.mean([s["detection_auroc"] for s in scores.values()]))
        history.append({"epoch": epoch, "dev": scores, "mean_dev": mean_dev})
        print(f"epoch {epoch}: mean dev {mean_dev:.4f}", flush=True)
        if mean_dev > best["score"]:
            best = {"score": mean_dev, "epoch": epoch}
            torch.save(model.state_dict(), RUN / "best.pt")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "epoch": epoch, "best": best}, last)
        (RUN / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(RUN / "best.pt", weights_only=True))
    final = {}
    for leg in ("L1", "L2", "L3", "L4"):
        px, y = load_split(leg)
        final[leg] = evaluate(model, px, y, device)
        print(leg, {k: round(v, 4) if isinstance(v, float) and v == v else v
                    for k, v in final[leg].items()}, flush=True)
    (RUN / "final_legs.json").write_text(
        json.dumps({"best": best, "legs": final}, indent=2))
    print("ARM_C_COMPLETE")


if __name__ == "__main__":
    main()
