---
type: snapshot
title: TB-E3 curation ladder — 48 fits, H1-H3 verdicts
status: current
as_of: 2026-09-01
source: "official run on the 3090 box, src/curate.py ladder; 48/48 fits, 0 failures; verdicts.json"
tags: [track-b, curation, results, tb-e3]
links: [specs/tb-e3-curation-ladder, 2026-09-01-backbone-grid, decisions/tb-e3-mapping-is-strict-drop]
---

# TB-E3 — the numbers

Pool: 998,479 rows over 8 training environments; G2 leg-decontamination removed 84,328
(shared-original leakage into the frozen legs), C1's within-class dedup another 96, 030 (nine
TGIF tools forge the same originals). Legs frozen before any arm: L1 22,941 / L2 3,998 /
L3 9,000 / L4 115,377. Instrument: frozen standard ViT-B/16 + 1,539-param head (TB-E2 recipe;
reproduces TB-E2's in-domain cell to <=0.0095). 8 arms x 2 sizes x 3 seeds; every head saved.

## detection_auroc, natural size (mean of 3 seeds; seed SD ~0.001-0.009)

| arm | L1 in-dom | L2 So-Fake-OOD | L3 held-out gens | L4 held-out tools |
|---|---:|---:|---:|---:|
| C0 naive pool (914k) | 0.7246 | 0.6644 | 0.9399 | 0.6791 |
| C1 +hygiene (817k) | 0.7436 | 0.6593 | 0.9424 | 0.6576 |
| C2 +matching (~35k) | 0.7523 | **0.7458** | **0.9732** | 0.5434 |
| C3a strict-drop | 0.7523 | 0.7458 | 0.9732 | 0.5434 |
| C3b multi-task | 0.7523 | 0.7458 | 0.9732 | 0.5434 |
| C4 equalized (~14k) | **0.7590** | 0.6764 | 0.9228 | 0.4760 |
| C5a random @B | 0.7186 | 0.7190 | 0.9433 | 0.4154 |
| C5b k-center @B | 0.7150 | 0.7440 | 0.9625 | **0.5286** |

Size-matched control (all arms at 4,651 rows): C2 vs C1 becomes +0.134 (L2) and +0.190 (L3)
— the L2/L3 effect is data QUALITY, not quantity. L4 at matched size is bad for every arm
(0.42-0.53): the tampered-tool leg needs mass nobody keeps at 4.6k rows.

## Verdicts (pre-registered rules, applied mechanically; verdicts.json)

- **H1 — mixed, leg-resolved.** C2 > C1 beyond 2x pooled seed SD on L2 (+0.0865, 2SD 0.0028)
  and L3 (+0.0307, 2SD 0.0011); C2 < C1 on L4 (-0.1142, 2SD 0.0044). The conjunctive rule
  therefore does not fire; the null branch (C0 ~ C5b) does not fire either. Honest summary:
  **matching buys generation-shift generalisation and pays for it in tampered-tool
  generalisation**, because the nuisance-matched pool retains almost no tampered mass from the
  header-separable environments (AUDITS ~0, SID-Set ~10 rows — their G1 is ~1.0).
- **H2 — strict-drop (C3a)**, by the tie rule: C3a == C3b to 4 dp on every leg (the ingested
  shortlist has zero unmappable binary fakes; the contrast is structural, as registered).
- **H3 — coreset earns its keep**: C5b > C5a beyond noise on 3/3 shifted legs
  (L2 +0.025, L3 +0.019, L4 +0.113). The one unqualified positive.
- **Watchdog**: 7 fd_auroc flags, all on transitions INTO small arms (C1->C2 on L3/L4,
  C3->C4, C4->C5a on L4) — confidence quality degrades when arms shrink; follow up before
  adopting any small arm as a deployment model.

## The two headline mechanisms

1. **The naive pool's shortcut is total.** Headers-only G1 on the pool: audits 0.9998,
   CF 0.995, imd2020 0.999, sagi_d 0.9998, sid_set 0.9994; only TGIF (all-PNG) is
   partially clean (0.678). Propensity-matched C2 drives per-env G1 to ~0.51 — the matched
   arm is the first arm in this project whose training signal cannot be file headers.
2. **Tampered needs pairs that matching cannot keep.** L4 detection tracks tampered training
   mass (C0 0.679 -> C2 0.543 -> C4 0.476), and the environments carrying original-vs-edit
   pairs are exactly the header-separable ones that matching guts. Curation for the
   synthetic axis and curation for the tampered axis are DIFFERENT problems: the first wants
   nuisance overlap, the second wants paired contrasts.

## Read beside this

TB-E2 single-dataset probes sat at chance on shifted accuracy; every arm here clears chance
on every leg. Multi-dataset training moves the needle even naively; curated multi-dataset
moves it far on the generation axis (L3 0.97, L2 0.75). The per-modality rows: tampered
in-domain is still weak everywhere (0.46-0.59) — the resize-224 low-pass hypothesis stands.

## Phase 2 (H4) and the hardening cells (same day)

**H4 — no invariance objective adopted** (rule: beat pooled ERM on L3+L4, lose <=0.01 L1).
GroupDRO L1 0.6754 / L2 0.7156 / L3 0.9683 / L4 0.6485; V-REx 0.7723 / 0.7155 / 0.9601 /
0.6232; pooled ERM (C3a) 0.7523 / 0.7458 / 0.9732 / 0.5434. Neither fires the rule — but both
repair +0.08-0.10 of the L4 damage that matching caused, at L2/L3 cost. Directed follow-up:
invariance helps exactly where matching starves the arm.

**FARE4-B/16 column on the winning arm** (adversarially fine-tuned encoder, eps=4/255): L1
0.669 / L2 0.644 / L3 0.913 / L4 0.534 (3 seeds, SD <= 0.003) vs the standard encoder's
0.752 / 0.746 / 0.973 / 0.543. The robustification tax survives curation on every clean leg —
good data does not close the gap.

**8/255 battery on the winning head** (C3a s1, SID prefix-1000, project threat model):
clean top-1 0.661, fd_auroc 0.7215. query_underconf (400 queries): argmax 100% preserved,
fd_auroc 0.7229 — the confidence axis HELD under the black-box budget that collapsed the
undefended B/32 probe in TB-E2. PGD-40 white-box: accuracy 0.104, fd_auroc 0.367 (below
chance) — undefended encoders die white-box at 8/255, the standing argument for the custom
8/255 fine-tune (no published robust CLIP exists above 4/255).
