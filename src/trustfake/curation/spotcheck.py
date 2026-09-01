"""G4: the 20-image manual spot-check of the unified label map.

For one ingested dataset, sample 20 rows stratified by mapped label,
re-decode the ORIGINAL bytes (not the cache -- the check is on the mapping,
so it must see what the reader saw), and render one contact sheet with the
mapped label over every tile. A human (or the executing agent, which can
read images) inspects the sheet and records the verdict in the docs before
the dataset enters any arm. FakeClue's inverted label convention is the
reason this gate exists.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from trustfake.curation.readers import iter_dataset
from trustfake.logging import get_logger

logger = get_logger("curation.spotcheck")

__all__ = ["make_contact_sheet"]

LABEL_NAMES = {0: "REAL", 1: "SYNTHETIC", 2: "TAMPERED", -1: "UNMAPPABLE"}
TILE = 224
COLUMNS = 5


def make_contact_sheet(
    dataset: str,
    data_dir: str | Path,
    out_file: str | Path,
    n: int = 20,
    seed: int = 0,
) -> Path:
    """Sample `n` rows (label-stratified, seeded) and write one PNG sheet.

    Per-class reservoir sampling over a streamed prefix of the dataset, so
    only the ~n kept rows' bytes are ever resident -- the raw pools carry
    hundreds of KB per row and this box does not have the RAM for a
    materialised sample.
    """
    rng = np.random.default_rng(seed)
    reservoirs: dict[int, list[dict]] = {}
    seen: dict[int, int] = {}
    per_class = max(1, n // 3)
    scanned = 0
    for _, shard_rows in iter_dataset(dataset, data_dir):
        for row in shard_rows:
            label = row["label3"]
            seen[label] = seen.get(label, 0) + 1
            reservoir = reservoirs.setdefault(label, [])
            if len(reservoir) < per_class:
                reservoir.append(row)
            else:
                slot = int(rng.integers(0, seen[label]))
                if slot < per_class:
                    reservoir[slot] = row
        scanned += len(shard_rows)
        if scanned >= 20000:  # variety, not the whole set
            break
    picked_rows = [row for reservoir in reservoirs.values() for row in reservoir][:n]

    n_rows = (len(picked_rows) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * TILE, n_rows * (TILE + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, row in enumerate(picked_rows):
        with Image.open(io.BytesIO(row["image"])) as pil:
            tile = pil.convert("RGB").resize((TILE, TILE))
        x = (position % COLUMNS) * TILE
        y = (position // COLUMNS) * (TILE + 18)
        sheet.paste(tile, (x, y))
        caption = f"{LABEL_NAMES[row['label3']]}  {str(row.get('generator') or '')[:24]}"
        draw.text((x + 2, y + TILE + 2), caption, fill="black")

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_file)
    logger.info(f"G4 sheet for {dataset}: {out_file} ({len(picked_rows)} tiles)")
    return out_file
