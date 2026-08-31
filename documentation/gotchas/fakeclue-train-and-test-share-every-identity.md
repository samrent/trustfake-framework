---
type: gotcha
title: FakeClue's own train and test splits share 100% of their identities
status: current
as_of: 2026-09-01
source: "computed from data_json/train.json + test.json, 104,343 + 5,000 rows"
tags: [data, fakeclue, leakage]
links: [fakeclue-splits-leak-ffpp-identities, fakeclue-dataset-properties]
---

**Trap.** Training on FakeClue's train split and evaluating on its test split is a total leak:

```
ALL group keys:   train 734   test 680   SHARED 680   (every one)
FF++ identities:  train 720   test 666   SHARED 666   (100.0%)
test rows whose group appears in train: 5,000 / 5,000 (100.0%)
```

Every identity in the test split is also in train, and every test row belongs to a group seen
during training. The measurement is memorisation, not detection.

This is distinct from [[fakeclue-splits-leak-ffpp-identities]], which is about carving a calib/test
split *within* one FakeClue split. This one is about the dataset's own train/test boundary, and it
is worse: not a leak you can fix by splitting more carefully, because the contamination is already
in the published splits.

**Fix.** Evaluate on **So-Fake-OOD**, a genuinely separate dataset. FakeClue-test may still be
reported, but only as contaminated-by-construction, and never as evidence that multi-dataset
training worked.

**Who this affects.** The colleague repo's `combined.py` folds FakeClue-train into joint training
and `eval_fake_clue.py` then evaluates on FakeClue-test — contaminated as shipped. Their advice to
"watch for data leaks, and evaluate on so-fake-ood" turns out to be two halves of one instruction.
