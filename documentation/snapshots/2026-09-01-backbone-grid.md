---
type: snapshot
title: Backbone grid — TeCoA4 B/32, standard B/16, FARE4 B/16
status: current
as_of: 2026-09-01
source: "jobs/track_b_backbone_grid.sh on the 3090 box; limit_test=1000, 9/9 eval cells, 0 failures"
tags: [track-b, results, confidence-axis, clip, backbone]
links: [2026-09-01-track-b-full-matrix, clip-is-more-fragile-on-the-confidence-axis]
---

Three arms x (train + 3 evals) = 12 steps, all with `.done` markers, all `ok` in
`grid.log`. **No cell failed.** Metrics read from each cell's own
`test_lightning_logs/version_N/metrics.csv` as named in its log; attacked cells use the
`query_underconf_` prefix, clean cells `nat_`.

## Every completed cell

| arm | dataset | condition | accuracy | fd_auroc | aurc | det_auroc | det_syn | det_tam | rec_syn | rec_tam |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_vit_b32_tecoa_probe | SID-Set (in-domain) | clean | 0.7132 | 0.7430 | 0.1376 | 0.8509 | 0.9401 | 0.7487 | 0.8785 | 0.6741 |
| clip_vit_b32_tecoa_probe | SID-Set (in-domain) | query_underconf | 0.7132 | 0.7219 | 0.1469 | 0.8456 | 0.9362 | 0.7418 | 0.8785 | 0.6741 |
| clip_vit_b32_tecoa_probe | So-Fake-OOD | clean | 0.4241 | 0.5688 | 0.5288 | 0.6199 | 0.6658 | 0.5758 | 0.4171 | 0.2937 |
| clip_vit_b16_probe | SID-Set (in-domain) | clean | 0.9093 | 0.8219 | 0.0245 | 0.9619 | 0.9977 | 0.9209 | 0.9972 | 0.8766 |
| clip_vit_b16_probe | SID-Set (in-domain) | query_underconf | 0.9093 | 0.5804 | 0.0558 | 0.9071 | 0.9893 | 0.8129 | 0.9972 | 0.8766 |
| clip_vit_b16_probe | So-Fake-OOD | clean | 0.4870 | 0.6512 | 0.4064 | 0.7250 | 0.8634 | 0.5925 | 0.8094 | 0.0476 |
| clip_vit_b16_fare_probe | SID-Set (in-domain) | clean | 0.7895 | 0.7714 | 0.0849 | 0.9004 | 0.9748 | 0.8152 | 0.9448 | 0.7405 |
| clip_vit_b16_fare_probe | SID-Set (in-domain) | query_underconf | 0.7895 | 0.7536 | 0.0913 | 0.8970 | 0.9711 | 0.8120 | 0.9448 | 0.7405 |
| clip_vit_b16_fare_probe | So-Fake-OOD | clean | 0.4961 | 0.5972 | 0.4250 | 0.6713 | 0.7682 | 0.5785 | 0.6630 | 0.2407 |

Accuracy, `recall_synthetic` and `recall_tampered` are bit-identical clean vs attacked in
all three arms — argmax preservation held again, so every `query_underconf` effect lives on
the confidence axis alone.

## 1. Does TeCoA4 reproduce FARE4's confidence-axis stability?

Yes, and slightly more so.

| arm | Phi clean | Phi query_underconf | drop |
|---|---:|---:|---:|
| B/32 standard (prior) | 0.8284 | 0.4837 | 0.345 |
| B/32 FARE4 (prior) | 0.7793 | 0.7485 | 0.031 |
| **B/32 TeCoA4** | 0.7430 | 0.7219 | **0.021** |
| **B/16 FARE4** | 0.7714 | 0.7536 | **0.018** |

TeCoA4's drop of 0.021 is in the same regime as FARE4's 0.031 and 16x smaller than the
undefended B/32 probe's 0.345 — at the same architecture, same base checkpoint, same
frozen-encoder protocol. A *supervised* robustification objective and an *unsupervised* one
land in the same place, so the confidence-axis effect belongs to adversarial fine-tuning in
general rather than to FARE specifically. TeCoA pays more for it than FARE does: clean
accuracy 0.7132 against FARE's 0.7873 and the undefended 0.8811.

## 2. Does standard ViT-B/16 show any of that stability?

No. Its Phi falls 0.8219 -> 0.5804, a drop of **0.241** — an order of magnitude worse than
any adversarially-trained arm (0.018–0.031) and the same qualitative collapse as the
undefended B/32 probe. The control behaves as predicted, so the effect is not an artifact of
patch size.

The clean comparison is B/16-standard (0.241) against B/16-FARE (0.018): same patch grid,
same `laion2B-s34B-b88K` base, differing only in adversarial training, and a 13x gap in the
collapse. That isolates the mechanism.

One honest caveat: 0.241 is meaningfully smaller than B/32-standard's 0.345, and the finer
grid starts from a higher clean Phi (0.8219 vs 0.8284) and lands above chance rather than at
it (0.5804 vs 0.4837). So patch size buys a *little* headroom on the confidence axis. It does
not buy stability, and it does not confound the robustness conclusion.

## 3. Does B/16 recover recall_tampered?

In-domain marginally, out-of-domain **not at all**.

| arm | rec_tam in-domain | rec_tam OOD |
|---|---:|---:|
| B/32 standard (prior) | 0.8418 | 0.0503 |
| **B/16 standard** | **0.8766** | **0.0476** |
| B/32 FARE4 (prior) | — | 0.2963 |
| B/16 FARE4 | 0.7405 | 0.2407 |
| B/32 TeCoA4 | 0.6741 | 0.2937 |

Standard B/16 gains +0.035 in-domain and moves OOD tampered recall from 0.0503 to 0.0476 —
i.e. nowhere. The finer patch grid does not repair the opposite-class blindness. What does is
adversarial training: all three robustified arms land at 0.24–0.29 OOD regardless of patch
size, and B/16-FARE (0.2407) is in fact slightly *worse* there than B/32-FARE (0.2963). The
rank bound removed by the square projection is therefore not the mechanism behind the OOD
tampered recovery.

**But do not read the OOD recovery as better tampered detection.** OOD
`detection_auroc_tampered` is near chance for every arm here — 0.5925 (B/16 std), 0.5785
(B/16 FARE), 0.5758 (B/32 TeCoA) — so nothing is ranking tampered images better. The
robustified arms simply sit at an operating point that predicts tampered more often, and pay
for it in `recall_synthetic` (0.8094 -> 0.6630 for B/16-FARE, -> 0.4171 for B/32-TeCoA). This
is the same trade-off shape flagged for B/32-FARE on 2026-09-01, and it is a threshold effect,
not a discrimination gain.

## What B/16 does buy

Standard B/16 is the strongest in-domain arm measured so far: accuracy 0.9093 (vs 0.8811 for
B/32-standard), detection_auroc 0.9619, aurc 0.0245. That gain is real and it is purely a
patch-size effect. It survives none of the shift: OOD accuracy 0.4870, still at chance.
