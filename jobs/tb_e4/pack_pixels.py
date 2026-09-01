"""TB-E4 stage 5a: pack the exact 224px pixel stream for pixel-level runs.

One uint8 memmap per split -- train (the C2-prime arm), each frozen leg,
each dev leg -- produced by the same PIL-bilinear resize the feature cache
used, so Arm B sees bit-identical pixels to what Arm A's features encode.
Sources are resolved per uid: staged QF-85 files for pair rows, original
parquet cells for SID/CF/So-Fake-OOD, deterministic file paths for
synthbuster / vision / audits-test / tgif.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path(os.environ["DATA_PATH"])
OUT = Path(os.environ["OUTPUT_PATH"])
E3, E4 = OUT / "tb_e3", OUT / "tb_e4"
PACK = E4 / "pixels"
SIZE = 224


def build_resolvers():
    pairs = pq.read_table(DATA / "tb_e4_qf85/pairs_manifest.parquet").to_pandas()
    pair_path = dict(zip(pairs["uid"], pairs["path"], strict=True))

    synth_files = {}
    for p in DATA.glob("synthbuster/**/*"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
            synth_files[f"{p.parent.name}/{p.name}"] = str(p)

    def resolve(uid: str):
        """-> ('file', path) | ('parquet', dataset, shard, row)"""
        if uid in pair_path:
            return ("file", pair_path[uid])
        head = uid.split(":", 1)[0]
        if head == "synthbuster":
            return ("file", synth_files[uid.split(":", 2)[2]])
        if head == "vision":
            device, collection, name = uid.split(":", 2)[2].split("/")
            return ("file", str(DATA / "vision/dataset" / device / "images"
                                / collection / name))
        if head == "tgif":
            _, split, rest = uid.split(":", 2)
            tool, category, name = rest.split("/")
            return ("file", str(DATA / "tgif/extracted" / tool / split
                                / category / name))
        if head == "audits":
            _, split, sub, method, ident = uid.split(":")
            stem = f"{int(ident):012d}" if sub == "coco" else ident
            folder = "original" if method == "Authentic" else method
            return ("file", str(DATA / "audits" / split / f"{split}_{sub}"
                                / folder / f"{stem}.jpg"))
        return ("parquet", head, None, None)

    return resolve


def parquet_lookup(index_root: str, dataset: str, uids: pd.Series):
    """(shard, row, data_dir) triples for parquet-backed rows, via the cache index."""
    root = Path(index_root) / dataset
    frames = [pq.read_table(f, columns=["uid", "shard", "row"]).to_pandas()
              for f in sorted(root.glob("index-*.parquet"))]
    index = pd.concat(frames).set_index("uid")
    picked = index.loc[uids]
    return picked["shard"].tolist(), picked["row"].tolist()


def decode_file(task):
    position, path = task
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Resize((SIZE, SIZE))
    try:
        with Image.open(path) as im:
            arr = np.asarray(tf(im.convert("RGB")), dtype=np.uint8)
        return position, arr
    except Exception:
        return position, None


DATASET_DIRS = {
    "sid_set": "sid_set",
    "so_fake_ood": "so_fake_ood",
    "community_forensics_small": "community_forensics_small",
}
IMAGE_COLUMNS = {
    "sid_set": ("img_id", "image"),
    "so_fake_ood": ("id", "image"),
    "community_forensics_small": (None, "image_data"),
}


def pack(name: str, frame: pd.DataFrame, resolve):
    dest = PACK / f"{name}.u8"
    meta = PACK / f"{name}.json"
    if meta.exists():
        print(f"SKIP {name}")
        return
    n = len(frame)
    memmap = np.lib.format.open_memmap(
        dest, mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    ok = np.zeros(n, dtype=bool)

    kinds = [resolve(u) for u in frame["uid"]]
    file_tasks = [(i, k[1]) for i, k in enumerate(kinds) if k[0] == "file"]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for position, arr in pool.map(decode_file, file_tasks, chunksize=64):
            if arr is not None:
                memmap[position] = arr
                ok[position] = True

    from io import BytesIO

    from PIL import Image
    from torchvision import transforms
    tf = transforms.Resize((SIZE, SIZE))
    parquet_rows = [(i, frame["uid"].iloc[i]) for i, k in enumerate(kinds)
                    if k[0] == "parquet"]
    by_dataset: dict[str, list] = {}
    for i, uid in parquet_rows:
        head = uid.split(":", 1)[0]
        dataset = ("community_forensics_small" if head == "community_forensics"
                   else head)
        by_dataset.setdefault(dataset, []).append((i, uid))
    for dataset, items in by_dataset.items():
        uids = pd.Series([u for _, u in items])
        shards, rows = parquet_lookup(str(E3 / "features"), dataset, uids)
        per_shard: dict[str, list] = {}
        for (i, _), shard, row in zip(items, shards, rows, strict=True):
            per_shard.setdefault(shard, []).append((i, row))
        data_dir = DATA / DATASET_DIRS[dataset]
        column = IMAGE_COLUMNS[dataset][1]
        for shard, members in per_shard.items():
            shard_path = next(data_dir.rglob(shard))
            table = pq.read_table(shard_path, columns=[column])
            cells = table[column].to_pylist()
            for i, row in members:
                cell = cells[row]
                data = cell["bytes"] if isinstance(cell, dict) else cell
                try:
                    with Image.open(BytesIO(data)) as im:
                        memmap[i] = np.asarray(tf(im.convert("RGB")), dtype=np.uint8)
                    ok[i] = True
                except Exception:
                    pass

    kept = frame.loc[ok].reset_index(drop=True)
    if not ok.all():
        # compact the memmap so rows stay aligned with the kept frame
        memmap[: ok.sum()] = memmap[np.flatnonzero(ok)]
        print(f"{name}: dropped {int((~ok).sum())} undecodable rows")
    memmap.flush()
    kept[["uid", "label3"]].to_parquet(PACK / f"{name}.parquet")
    meta.write_text(json.dumps({"n": int(ok.sum()), "shape": [SIZE, SIZE, 3]}))
    print(f"packed {name}: {int(ok.sum())} rows")


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    resolve = build_resolvers()
    arm = pq.read_table(E4 / "arm/c2prime.parquet").to_pandas()
    pack("train", arm, resolve)
    for leg in ("L1", "L2", "L3", "L4", "calib"):
        frame = pq.read_table(E3 / "legs" / f"{leg}.parquet").to_pandas()
        pack(leg, frame, resolve)
    for leg in ("dev_L3", "dev_L4"):
        frame = pq.read_table(E4 / "legs_dev" / f"{leg}.parquet").to_pandas()
        pack(leg, frame, resolve)
    print("PACK_PIXELS_COMPLETE")


if __name__ == "__main__":
    main()
