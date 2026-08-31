---
type: reference
title: FakeClue — composition, floors, and how to read an accuracy on it
status: current
as_of: 2026-08-31
source: "computed from data_json/test.json, labels only, 5000 rows"
tags: [data, fakeclue]
links: [fakeclue-label-convention-is-inverted, fakeclue-splits-leak-ffpp-identities]
---

Test split: **5000 rows, 63.8% fake**. So the **majority floor is 0.638**, not 0.5.

The metadata rule "square image => fake" — reading **no pixels at all** — scores **0.6904**. Any
accuracy below ~0.69 on this dataset is worse than a rule that never looks at the image. 60.9% of
images are square; among squares 77% are fake.

Category composition is skewed by label, so a category-stratified breakout matters:

| category | n | fake share |
|---|---:|---:|
| deepfake | 1168 | 80% |
| satellite | 875 | 51% |
| object | 867 | 55% |
| animal | 749 | 49% |
| doc | 576 | 80% |
| human | 403 | 70% |
| scene | 362 | 62% |

**Ground truth is binary and does not distinguish a fully generated image from an edited one**, so
scoring a 3-class model here requires the fold, and the per-modality breakout is unavailable. That
is a property of the benchmark, not a modelling choice — and it is why So-Fake-OOD (whose labels
are REAL / FULL_SYNTHETIC / TAMPERED, matching SID-Set) is the better shift condition.

Sizes: `test.zip` 1.3 GB, `train.zip` **28 GB** (not needed — FakeClue is used here as an
evaluation benchmark only).
