"""TB-E4 stage 2: embed the QF-85 pair cache under the frozen standard B/16.

Reads pairs_manifest.parquet, embeds in 2000-row chunks into
$OUTPUT_PATH/tb_e4/features_qf85/pairs/ using the same machinery and policy
as the TB-E3 cache (bytes are the staged QF-85 files; jpeg_q measures 85 by
construction). The index carries `env` so gates can group per environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq

from trustfake.curation import embed as embed_module

DATA = Path(os.environ["DATA_PATH"])
OUT = Path(os.environ["OUTPUT_PATH"]) / "tb_e4" / "features_qf85"


def _iter_manifest(data_dir):
    frame = pq.read_table(DATA / "tb_e4_qf85/pairs_manifest.parquet").to_pandas()
    rows = [
        {
            "uid": r.uid,
            "path": r.path,
            "label3": int(r.label3),
            "label_bin": int(r.label3 > 0),
            "generator": r.generator if isinstance(r.generator, str) else None,
            "source_split": r.source_split,
            "env": r.env,
            "pair_id": r.pair_id,
        }
        for r in frame.itertuples(index=False)
    ]
    rows.sort(key=lambda r: r["uid"])
    for start in range(0, len(rows), 2000):
        yield f"pairs-{start // 2000:04d}", rows[start : start + 2000]


def main():
    from trustfake.curation import readers

    readers.READERS["tb_e4_pairs"] = lambda data_dir: _iter_manifest(data_dir)
    embed_module.embed_dataset("tb_e4_pairs", data_dir=DATA, out_dir=OUT, num_workers=6)
    print("EMBED_PAIRS_COMPLETE")


if __name__ == "__main__":
    main()
