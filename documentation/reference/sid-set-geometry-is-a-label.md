---
type: reference
title: SID-Set geometry is a label — every fake is 1024×1024, 96% of reals are not square
status: current
as_of: 2026-09-04
source: "width/height columns of all 34 validation-* parquet shards, counted 2026-09-04 on the box; class balance from the same columns"
tags: [sid-set, dataset, shortcut, geometry]
links: [fakeclue-dataset-properties, snapshots/2026-09-04-track-c-first-results, specs/tb-e3-curation-ladder]
---

# SID-Set geometry is a label

Counted over the 34 validation shards (10,000 rows per class):

| class | n | square | median W×H | most common sizes |
|---|---:|---:|---|---|
| real | 10,000 | 0.042 | 1024×768 | 1024×768 (2,447), 1024×683 (1,726), 768×1024 (498) |
| synthetic | 10,000 | 1.000 | 1024×1024 | 1024×1024 (10,000) |
| tampered | 10,000 | 1.000 | 1024×1024 | 1024×1024 (10,000) |

`width == height` separates real from fake with 0.958 recall on real and 1.0 on both fake
classes, reading no pixels. It carries **nothing** about synthetic vs tampered (both 100%
square). The fit split (30 train shards) is balanced: 8,409 / 8,415 / 8,496.

## What survives the datamodule's controls

- `input_mode=resize` (the default and every Track A/B/C number): `Resize((224, 224))` destroys
  the aspect ratio but not the evidence — a 1024×768 real is resampled anisotropically (4.6× on
  one axis, 3.4× on the other) and every fake isotropically. The resampling signature is a
  high-frequency cue; it is the kind of feature adversarial training removes.
- `squarecrop=true` centre-crops to the short side before the resize, so every image is
  resampled isotropically by the same factor. Fit and eval must both use it. It does not remove
  the "short side == 1024" residue (the datamodule's own comment), and a Track C rerun under it
  needs a new depth store (the store records `squarecrop`).
- `geometry_filter` restricts calib/test to rows where the rule carries no label information;
  on real SID-Set `nonsquare` leaves 569 rows that are all real — there is no fake to keep.

## How it shows up in Track C

After 8/255 adversarial training, real-vs-tampered collapses toward chance (real recall 0.38,
62% of reals predicted tampered; max-probability sits at 0.505 ± 0.01 on every real and
tampered image) while synthetic stays at 0.95 recall under a 10-step PGD. Synthetic vs
tampered cannot be geometry (both square), so the robust synthetic signal is something else
(generator fingerprint); the real-vs-tampered signal the standard arm used (0.82 / 0.70 recall)
is a non-robust feature, and the resampling asymmetry is the obvious candidate. The 0.75 clean
floor in the Track C spec is therefore not just "the 8/255 cost": it is the cost of losing the
geometry-and-resampling cue that in-domain real-vs-tampered accuracy leaned on. This is a
measurement of the dataset, not an accusation of the model: the separable feature is in the
files.
