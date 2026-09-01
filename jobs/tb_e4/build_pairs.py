"""TB-E4 stage 1: enumerate original/edit pairs, re-encode both to JPEG q85.

Writes $DATA_PATH/tb_e4_qf85/<env>/... plus pairs_manifest.parquet
(uid, path, env, label3, generator, pair_id, source_split). Every image is
decoded once and saved at quality 85 -- within a pair the photo, size and
compression profile are then identical; only the edit differs. Idempotent:
existing staged files are skipped.
"""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(os.environ["DATA_PATH"])
STAGE = DATA / "tb_e4_qf85"
QUALITY = 85


def _row(**kw):
    return kw


def enumerate_pairs() -> pd.DataFrame:
    rows = []

    # --- AUDITS train+val: per-method manipulated/ + original/ trees
    meta = pq.read_table(DATA / "audits/data/train-00000-of-00001.parquet").to_pandas()
    keep = meta["training"].isin(["train", "val"])
    keep &= meta["manipulation_type"] != "Authentic"
    wanted = meta[keep]
    for r in wanted.itertuples(index=False):
        sub = r.subset.lower()
        stem = f"{int(r.id):012d}" if r.subset == "COCO" else str(r.id)
        base = (
            DATA / "audits" / r.training / f"{r.training}_{sub}"
            / r.manipulation_type
        )
        pair = f"audits:{r.training}:{sub}:{r.manipulation_type}:{r.id}"
        for kind, label in (("manipulated", 2), ("original", 0)):
            src = base / kind / f"{stem}.jpg"
            uid = f"{pair}" if kind == "manipulated" else f"{pair}:orig"
            rows.append(_row(uid=uid, src=str(src), env="audits", label3=label,
                             generator=r.manipulation_type if label == 2 else None,
                             pair_id=pair, source_split=r.training))

    # --- SAGI-D train+val, powerpaint excluded; originals deduped by src_path
    sagi = pd.read_csv(DATA / "sagi_d/sagid.csv")
    sagi = sagi[sagi["split"].isin(["train", "val"])]
    sagi = sagi[~sagi["inpainting_model"].fillna("").str.contains("powerpaint")]
    def s_resolve(raw):
        rel = raw.replace("\\", "/").removeprefix("sagid/")
        p = DATA / "sagi_d" / rel
        return p if p.exists() else DATA / "sagi_d" / "sagid" / rel
    for r in sagi.itertuples(index=False):
        pair = "sagi:" + r.src_path.replace("\\", "/")
        rows.append(_row(uid="e4:" + r.img_path.replace("\\", "/"),
                         src=str(s_resolve(r.img_path)),
                         env="sagi_d", label3=2, generator=r.inpainting_model,
                         pair_id=pair, source_split=r.split))
    for r in sagi.drop_duplicates("src_path").itertuples(index=False):
        pair = "sagi:" + r.src_path.replace("\\", "/")
        rows.append(_row(uid="e4:" + r.src_path.replace("\\", "/"),
                         src=str(s_resolve(r.src_path)),
                         env="sagi_d", label3=0, generator=None,
                         pair_id=pair, source_split=r.split))

    # --- TGIF training split: tools except orig/ps-sp/flux1filldev*; pair by id prefix
    ex = DATA / "tgif/extracted"
    orig_index = {}
    for p in (ex / "orig" / "training").rglob("*.png"):
        orig_index[(p.parent.name, p.name.split("_")[0])] = p
    for tool_dir in sorted(ex.iterdir()):
        tool = tool_dir.name
        if tool in ("orig", "ps-sp") or tool.startswith("flux1filldev"):
            continue
        for p in (tool_dir / "training").rglob("*.png"):
            key = (p.parent.name, p.name.split("_")[0])
            orig = orig_index.get(key)
            if orig is None:
                continue
            pair = f"tgif:{key[0]}/{key[1]}"
            rows.append(_row(uid=f"e4:tgif:{tool}/{key[0]}/{p.name}", src=str(p),
                             env="tgif", label3=2, generator=tool,
                             pair_id=pair, source_split="training"))
    seen_origs = set()
    for r in [x for x in rows if x["env"] == "tgif"]:
        if r["pair_id"] in seen_origs:
            continue
        seen_origs.add(r["pair_id"])
        cat, ident = r["pair_id"].removeprefix("tgif:").split("/")
        orig = orig_index[(cat, ident)]
        rows.append(_row(uid=f"e4:tgif:orig/{cat}/{orig.name}", src=str(orig),
                         env="tgif", label3=0, generator=None,
                         pair_id=r["pair_id"], source_split="training"))

    # --- IMD2020 real_life: folders holding a *_orig.jpg
    for folder in sorted((DATA / "imd2020/real_life").iterdir()):
        if not folder.is_dir():
            continue
        origs = list(folder.glob("*_orig.jpg"))
        if not origs:
            continue
        pair = f"imd:{folder.name}"
        rows.append(_row(uid=f"e4:imd:{folder.name}/{origs[0].name}", src=str(origs[0]),
                         env="imd2020", label3=0, generator=None,
                         pair_id=pair, source_split="real_life"))
        for p in sorted(folder.iterdir()):
            bad = p.suffix.lower() not in (".jpg", ".jpeg", ".png")
            if bad or "mask" in p.name or p.stem.endswith("_orig"):
                continue
            rows.append(_row(uid=f"e4:imd:{folder.name}/{p.name}", src=str(p),
                             env="imd2020", label3=2, generator="human_manual",
                             pair_id=pair, source_split="real_life"))

    frame = pd.DataFrame(rows)
    missing = ~frame["src"].map(lambda s: Path(s).exists())
    if missing.any():
        print(f"dropping {int(missing.sum())} rows with missing sources")
        frame = frame[~missing]
    # drop pairs that lost a member
    counts = frame.groupby("pair_id")["label3"].nunique()
    whole = set(counts[counts == 2].index)
    frame = frame[frame["pair_id"].isin(whole)]
    def staged(env, uid):
        bucket = zlib.crc32(uid.encode()) % 100  # stable across runs
        name = uid.replace("/", "_").replace(":", "_")[-120:] + ".jpg"
        return str(STAGE / env / f"{bucket:02d}" / name)

    frame["path"] = [
        staged(env, uid)
        for env, uid in zip(frame["env"], frame["uid"], strict=True)
    ]
    return frame.reset_index(drop=True)


def reencode(task):
    src, dst = task
    dst = Path(dst)
    if dst.exists():
        return 0
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im.convert("RGB").save(dst, "JPEG", quality=QUALITY)
        return 0
    except Exception:
        return 1


def main():
    frame = enumerate_pairs()
    print(frame.groupby(["env", "label3"]).size())
    print("total:", len(frame), "| pairs:", frame["pair_id"].nunique())
    STAGE.mkdir(parents=True, exist_ok=True)
    tasks = list(zip(frame["src"], frame["path"], strict=True))
    failed = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, r in enumerate(pool.map(reencode, tasks, chunksize=256)):
            failed += r
            if i % 50000 == 0:
                print(f"re-encoded {i}/{len(tasks)}", flush=True)
    print("failed re-encodes:", failed)
    frame = frame[frame["path"].map(lambda p: Path(p).exists())]
    table = pa.Table.from_pandas(frame.drop(columns=["src"]), preserve_index=False)
    pq.write_table(table, STAGE / "pairs_manifest.parquet")
    print("manifest:", len(frame), "rows")
    print("BUILD_PAIRS_COMPLETE")


if __name__ == "__main__":
    main()
