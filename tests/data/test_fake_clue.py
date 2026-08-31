"""FakeClue: the label inversion and the identity firewall.

Both failures these tests pin are silent. Loading FakeClue's raw labels
returns plausible, exactly-inverted metrics; splitting its rows at random
returns a calib/test pair that shares faces and reports memorisation as
detection. Neither raises, so neither is caught by anything but a test.

No network, no zip, no images: the split logic is pure and is exercised on
synthetic records shaped like FakeClue's json.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from trustfake.data.fake_clue import (
    FAKE_CLUE_FAKE_ID,
    FAKE_CLUE_REAL_ID,
    FakeClueDataModule,
    FakeClueTorchDataset,
    assign_groups,
    group_key,
)

PROJECT_REAL, PROJECT_FAKE = 0, 1


# --------------------------------------------------------------------------
# The label inversion
# --------------------------------------------------------------------------


def _one_row_zip(tmp_path: Path, rel: str, label: int) -> tuple[Path, list[dict]]:
    zip_path = tmp_path / "test.zip"
    img = tmp_path / "img.png"
    Image.new("RGB", (32, 32), (120, 30, 30)).save(img)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(img, rel)
    return zip_path, [
        {"image": rel, "label": label, "cate": "x", "width": 32, "height": 32}
    ]


def test_fake_clue_real_becomes_class_zero(tmp_path):
    """FakeClue says 1 = real; this project says 0 = real. Getting this
    backwards inverts every detection AUROC and every moderation decision
    while raising nothing."""
    from torchvision import transforms

    zip_path, records = _one_row_zip(tmp_path, "a/b.png", FAKE_CLUE_REAL_ID)
    ds = FakeClueTorchDataset(records, zip_path, "test", transforms.ToTensor())
    _, label = ds[0]
    assert int(label) == PROJECT_REAL


def test_fake_clue_fake_becomes_class_one(tmp_path):
    from torchvision import transforms

    zip_path, records = _one_row_zip(tmp_path, "a/b.png", FAKE_CLUE_FAKE_ID)
    ds = FakeClueTorchDataset(records, zip_path, "test", transforms.ToTensor())
    _, label = ds[0]
    assert int(label) == PROJECT_FAKE


def test_batches_are_two_tuples(tmp_path):
    """The project contract is (image, label). FakeClue ships no masks, so a
    third element would have to be invented."""
    from torchvision import transforms

    zip_path, records = _one_row_zip(tmp_path, "a/b.png", FAKE_CLUE_FAKE_ID)
    item = FakeClueTorchDataset(records, zip_path, "test", transforms.ToTensor())[0]
    assert len(item) == 2
    assert isinstance(item[0], torch.Tensor) and item[0].shape[0] == 3


# --------------------------------------------------------------------------
# The identity firewall
# --------------------------------------------------------------------------


def test_group_key_reads_ffpp_identity_pairs():
    """A manipulated FF++ clip is <src>_<tgt> and belongs to BOTH identities,
    which is how a fake row collides with real rows of either."""
    assert group_key("ff++/fake/Deepfakes/c23/frames/856_881/1066.png") == (
        "856",
        "881",
    )
    assert group_key("ff++/real/youtube/c23/frames/858/12.png") == ("858",)


def test_group_key_falls_back_to_parent_outside_ffpp():
    """Generated images have no identity structure; the source folder is the
    unit that must not be split."""
    assert group_key("genimage/fake/416_adm_45.PNG") == ("genimage/fake",)
    assert group_key("satellite/real/41.8,-87.6_mbev.jpg") == ("satellite/real",)


def test_no_identity_crosses_the_split():
    """THE test. Rows sharing any identity must land in one role -- including
    the transitive case, where 'a_b' and 'b_c' bind a and c together."""
    records = [
        {"image": f"ff++/fake/Deepfakes/c23/frames/{a}_{b}/0.png"}
        for a, b in [("1", "2"), ("2", "3"), ("4", "5"), ("6", "7"), ("7", "8")]
    ] + [
        {"image": f"ff++/real/youtube/c23/frames/{i}/0.png"}
        for i in ("1", "5", "8", "9")
    ]
    calib, test = assign_groups(records, calib_fraction=0.5, seed=0)

    assert set(calib).isdisjoint(test)
    assert sorted(calib + test) == list(range(len(records)))

    def identities(idxs):
        out = set()
        for i in idxs:
            out.update(group_key(records[i]["image"]))
        return out

    assert identities(calib).isdisjoint(identities(test)), (
        "an identity appears in both roles: the split measures memorisation"
    )


def test_transitive_groups_are_unioned():
    """1-2 and 2-3 share identity 2, so all three rows are one component and
    cannot be dealt to different roles."""
    records = [
        {"image": "ff++/fake/Deepfakes/c23/frames/1_2/0.png"},
        {"image": "ff++/fake/Deepfakes/c23/frames/2_3/0.png"},
        {"image": "ff++/real/youtube/c23/frames/3/0.png"},
    ]
    calib, test = assign_groups(records, calib_fraction=0.5, seed=0)
    assert len(calib) == 0 or len(test) == 0, "one indivisible component, split anyway"


def test_split_is_deterministic_under_the_manifest_seed():
    # DISJOINT pairs: 0_1, 2_3, 4_5 ... Chaining them ({i}_{i+1}) would union
    # every identity into a single component -- correct behaviour, useless
    # fixture, and the reason this test data is written out explicitly.
    records = [
        {"image": f"ff++/fake/Deepfakes/c23/frames/{2 * i}_{2 * i + 1}/0.png"}
        for i in range(40)
    ]
    a = assign_groups(records, 0.3, seed=0)
    b = assign_groups(records, 0.3, seed=0)
    c = assign_groups(records, 0.3, seed=1)
    assert a == b
    assert a != c, "a different manifest seed must deal a different partition"


def test_roles_are_exhaustive_and_disjoint():
    records = [{"image": f"genimage/fake/{i}.png"} for i in range(10)] + [
        {"image": f"doc/real/set{i}/x.png"} for i in range(10)
    ]
    calib, test = assign_groups(records, 0.4, seed=3)
    assert set(calib).isdisjoint(test)
    assert sorted(calib + test) == list(range(20))


# --------------------------------------------------------------------------
# Construction guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_degenerate_calib_fraction(bad):
    with pytest.raises(ValueError, match="calib_fraction"):
        FakeClueDataModule(calib_fraction=bad)


def test_rejects_unknown_input_mode():
    with pytest.raises(ValueError, match="input_mode"):
        FakeClueDataModule(input_mode="squarecrop")


def test_missing_data_without_download_is_an_error(tmp_path):
    dm = FakeClueDataModule(data_dir=tmp_path, download=False)
    with pytest.raises(FileNotFoundError):
        dm.prepare_data()


def test_limit_test_caps_only_test(tmp_path):
    """calib is never capped: every condition must be read against thresholds
    fitted on the same calibration data."""
    records = [
        {"image": f"genimage/fake/{i}.png", "label": FAKE_CLUE_FAKE_ID}
        for i in range(20)
    ]
    records += [
        {"image": f"doc/real/s{i}/x.png", "label": FAKE_CLUE_REAL_ID} for i in range(20)
    ]
    (tmp_path / "data_json").mkdir(parents=True)
    (tmp_path / "data_json" / "test.json").write_text(json.dumps(records))
    (tmp_path / "test.zip").touch()

    dm = FakeClueDataModule(
        data_dir=tmp_path, calib_fraction=0.5, limit_test=3, download=False
    )
    dm.setup()
    assert len(dm.test_dataset) == 3
    assert len(dm.calib_dataset) > 3


def test_group_deal_preserves_the_class_prior():
    """Components are label-pure, so an unstratified deal skews the prior --
    measured at 87.5% fake in calib against 53.7% in test on the real split,
    which breaks any threshold fitted on calib. With enough components to
    divide, both roles must land near the dataset prior."""
    records = [
        {"image": f"genimage/set{i}/x.png", "label": FAKE_CLUE_FAKE_ID}
        for i in range(90)
    ]
    records += [
        {"image": f"doc/real/s{i}/x.png", "label": FAKE_CLUE_REAL_ID} for i in range(30)
    ]
    calib, test = assign_groups(records, 0.3, seed=0)

    prior = sum(1 for r in records if r["label"] == FAKE_CLUE_FAKE_ID) / len(records)

    def rate(idx):
        hits = sum(1 for i in idx if records[i]["label"] == FAKE_CLUE_FAKE_ID)
        return hits / max(len(idx), 1)

    assert abs(rate(calib) - prior) < 0.10
    assert abs(rate(test) - prior) < 0.10


def test_indivisible_group_warns_rather_than_silently_skewing(caplog):
    """One component holding most of a label cannot be divided without
    breaking the firewall, so the prior CANNOT be matched. The requirement is
    that this is reported, not that it is fixed."""
    import logging

    records = [
        {"image": "genimage/fake/x.png", "label": FAKE_CLUE_FAKE_ID} for _ in range(90)
    ]
    records += [
        {"image": f"doc/real/s{i}/x.png", "label": FAKE_CLUE_REAL_ID} for i in range(10)
    ]
    with caplog.at_level(logging.WARNING):
        calib, test = assign_groups(records, 0.3, seed=0)

    assert set(calib).isdisjoint(test), "the firewall holds even when unbalanced"


def test_capped_test_split_is_not_class_degenerate():
    """`limit_test` takes a prefix, and the repo documents that as a nested
    sample of the same population. FakeClue's json is clustered by category,
    so an unshuffled prefix is single-class -- measured on the real split, the
    first 64 rows were all one label, which sends every AUROC to NaN."""
    records = [
        {"image": f"catA/fake/{i}.png", "label": FAKE_CLUE_FAKE_ID} for i in range(200)
    ]
    records += [
        {"image": f"catB/real/s{i}/x.png", "label": FAKE_CLUE_REAL_ID}
        for i in range(200)
    ]
    _, test = assign_groups(records, 0.3, seed=0)

    prefix = test[:60]
    labels = {records[i]["label"] for i in prefix}
    assert len(labels) == 2, "a capped prefix must still contain both classes"
