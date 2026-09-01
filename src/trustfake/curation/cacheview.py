"""Loading the embed-once cache back into (index, features) pairs.

A dataset's cache is the set of ``index-*.parquet`` / ``features-*.npy``
pairs written by `trustfake.curation.embed`. This module concatenates them
in a deterministic (sorted-shard) order so that a row number is stable
across sessions, and every downstream artifact (arm index lists, gate
reports) references rows by ``uid``, never by position.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trustfake.logging import get_logger

logger = get_logger("curation.cache")

__all__ = ["load_cache", "gather_features"]


def load_cache(
    features_root: str | Path, dataset: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one dataset's full cache.

    Returns:
        (index, features): a DataFrame with the index columns plus
        ``cache_row``, and an (N, 512) fp16 array aligned with it.
    """
    root = Path(features_root) / dataset
    index_files = sorted(root.glob("index-*.parquet"))
    if not index_files:
        msg = f"No cache for '{dataset}' under {root}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    frames, blocks = [], []
    for index_file in index_files:
        stem = index_file.name.removeprefix("index-").removesuffix(".parquet")
        feature_file = root / f"features-{stem}.npy"
        table = pq.read_table(index_file).to_pandas()
        block = np.load(feature_file)
        if len(table) != block.shape[0]:
            msg = (
                f"Cache misalignment in {dataset}/{stem}: {len(table)} index "
                f"rows vs {block.shape[0]} feature rows"
            )
            logger.error(msg)
            raise ValueError(msg)
        frames.append(table)
        blocks.append(block)
    index = pd.concat(frames, ignore_index=True)
    features = np.concatenate(blocks, axis=0)
    index["cache_row"] = np.arange(len(index))
    if index["uid"].duplicated().any():
        duplicated = index.loc[index["uid"].duplicated(), "uid"].head(3).tolist()
        msg = f"Duplicate uids in {dataset} cache, e.g. {duplicated} -- row key broken"
        logger.error(msg)
        raise ValueError(msg)
    logger.info(f"cache {dataset}: {len(index)} rows, {len(index_files)} shards")
    return index, features


def gather_features(
    index: pd.DataFrame, features: np.ndarray, uids: pd.Series | list[str]
) -> np.ndarray:
    """Features for `uids`, in that order. Raises on any unknown uid."""
    positions = index.set_index("uid")["cache_row"].reindex(uids)
    if positions.isna().any():
        missing = positions[positions.isna()].index[:3].tolist()
        raise KeyError(f"uids not in cache, e.g. {missing}")
    return features[positions.to_numpy(dtype=np.int64)]
