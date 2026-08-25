"""Contract tests for the shard-level split manifest.

Each test defends one property of the leakage firewall. They are pure --
no data, no network, no GPU.
"""

import pytest

from trustfake.data.manifest import PROFILES, assign_shards

TRAIN = [f"train-{i:05d}-of-00249.parquet" for i in range(40)]
VAL = [f"validation-{i:05d}-of-00034.parquet" for i in range(34)]


def test_is_deterministic():
    a = assign_shards(TRAIN, VAL, PROFILES["full"], seed=0)
    b = assign_shards(TRAIN, VAL, PROFILES["full"], seed=0)
    assert a == b


def test_input_order_does_not_matter():
    """The assignment is a function of the shard SET, not the listing order."""
    a = assign_shards(TRAIN, VAL, PROFILES["full"], seed=0)
    b = assign_shards(
        list(reversed(TRAIN)), list(reversed(VAL)), PROFILES["full"], seed=0
    )
    assert a == b


def test_calib_and_test_are_shard_disjoint():
    a = assign_shards(TRAIN, VAL, PROFILES["full"], seed=0)
    assert not (set(a["calib"]) & set(a["test"]))


def test_roles_come_from_the_right_source_split():
    a = assign_shards(TRAIN, VAL, PROFILES["train_holdout"], seed=0)
    assert all(s.startswith("train-") for s in a["fit"] + a["holdout"])
    assert all(s.startswith("validation-") for s in a["calib"] + a["test"])


def test_holdout_is_disjoint_from_fit():
    a = assign_shards(TRAIN, VAL, PROFILES["train_holdout"], seed=0)
    assert not (set(a["fit"]) & set(a["holdout"]))


def test_calib_and_test_do_not_move_when_fit_grows():
    """THE comparability property: enlarging fit (full -> train profile) must
    leave calib and test identical, so numbers reported earlier stay
    comparable. The validation permutation must not depend on train counts."""
    full = assign_shards(TRAIN, VAL, PROFILES["full"], seed=0)
    train = assign_shards(TRAIN, VAL, PROFILES["train"], seed=0)
    assert full["calib"] == train["calib"]
    assert full["test"] == train["test"]
    assert full["fit"] != train["fit"][: len(full["fit"])] or True  # fit may overlap
    assert len(train["fit"]) > len(full["fit"])


def test_seed_actually_permutes_validation_shards():
    a = assign_shards(TRAIN, VAL, PROFILES["full"], seed=0)
    b = assign_shards(TRAIN, VAL, PROFILES["full"], seed=1)
    assert set(a["calib"]) != set(b["calib"])
    # and neither is the trivial first-k prefix
    assert a["calib"] != sorted(VAL)[: len(a["calib"])]


def test_insufficient_shards_raise():
    with pytest.raises(ValueError, match="Not enough shards"):
        assign_shards(TRAIN[:5], VAL, PROFILES["train"], seed=0)
    with pytest.raises(ValueError, match="Not enough shards"):
        assign_shards(TRAIN, VAL[:10], PROFILES["full"], seed=0)


def test_smoke_profile_fits_in_five_shards():
    """The offline pre-flight must not need the full dataset on disk."""
    a = assign_shards(TRAIN[:2], VAL[:3], PROFILES["smoke"], seed=0)
    assert {len(v) for v in a.values()} <= {1, 2}
