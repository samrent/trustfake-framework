---
type: handoff
title: Track A / Track B state, and how to resume
status: current
as_of: 2026-09-01
source: "TB-E3 executed end-to-end on the 3090 box, 2026-09-01 (branch claude/tb-e3-curation-ladder-255c6d)"
tags: [experiments, track-a, track-b, tb-e3]
links: [specs/tb-e3-curation-ladder, 2026-09-01-tb-e3-curation-ladder, decisions/tb-e3-mapping-is-strict-drop, 2026-09-01-backbone-grid]
---

# Where we are

**Track A (ResNet-18 EV-AT ladder)** — colleagues' lane, unchanged. Its transfer failure is
now better understood: TB-E3 showed the data was the lever, so re-running EV-AT arms on the
winning composition is the natural (conversation-first) follow-up.

**Track B — TB-E3 (the curation ladder) EXECUTED 2026-09-01**, same-day registration to
verdicts. Everything below happened on branch `claude/tb-e3-curation-ladder-255c6d` (pushed;
merge to main pending review).

## What exists now

- **Nine datasets ingested, embedded once, G4-verified** (~1.5M cached rows under
  `_runs/out/tb_e3/features/`): SID-Set, So-Fake-OOD, CommunityForensics-Small, AUDITS,
  Synthbuster, VISION, IMD2020, TGIF/TGIF2, SAGI-D. GenImage deferred (spec change log);
  NIST MFC / Chameleon / RAISE still need the human emails/forms.
- **Frozen legs** (`tb_e3/legs/`): L1 22,941 / L2 3,998 / L3 9,000 (DALL·E 3, Midjourney v5,
  Firefly) / L4 115,377 (AUDITS PowerPaint + TGIF ps-sp).
- **48 ladder fits + heads** (`tb_e3/fits/*.pt`), phase-2 heads (`tb_e3/phase2/`), FARE
  column + 8/255 battery (`tb_e3/hardening/`). Backups:
  `_runs/backups/tb_e3-results-20260901-*.tar.gz`.
- Tooling: `src/trustfake/curation/` + `src/curate.py` + `jobs/tb_e3/` (all lint-clean,
  794 tests green).

## The verdicts (snapshots/2026-09-01-tb-e3-curation-ladder.md has the tables)

- **H1 mixed, leg-resolved**: matching buys L2 +0.087 / L3 +0.031 (30x seed noise; bigger at
  matched size) and costs L4 -0.114 — nuisance overlap and paired-contrast mass are competing
  resources. **H2**: strict-drop mapping (tie rule; decisions leaf written). **H3**: k-center
  coreset beats random on 3/3 shifted legs. **H4**: no invariance objective adopted; both
  repair ~+0.09 of C2's L4 damage (directed follow-up).
- Hardening cells: FARE4-B/16's clean tax survives curation on every leg; the winner's
  confidence axis HELD under the 400-query black-box attack at 8/255 but PGD white-box
  demolishes it — the argument for a custom 8/255 encoder fine-tune stands.

## TB-E4 (overnight 2026-09-02): executed to completion

Spec `specs/tb-e4-pairs-and-training.md`; numbers `snapshots/2026-09-02-tb-e4-pairs-and-training.md`.
H5 not adopted; **H6 ADOPTED — the full fine-tune on the pair-fixed composition takes every
shifted leg (L2 0.786 / L3 0.977 / L4 0.732, all project bests)**, but the 8/255 battery
shows it trades away the frozen probe's black-box confidence stability (fd_auroc 0.708 ->
0.517 under query_underconf). Best clean-shift model: `_runs/out/tb_e4/armB/best.pt`. Best
confidence-robust model: the TB-E3 C3a frozen probe. Nothing dominates both axes yet.

## Next steps, in leverage order

1. **Merge the branch** (PR from `claude/tb-e3-curation-ladder-255c6d` — now carries TB-E3 AND TB-E4).
1b. **Close the axis trade**: harden Arm B (partial-freeze or adversarial fine-tune at
   8/255) or budget-balance the composition so one model holds both the shifted legs and
   the confidence axis — the sentence "nothing dominates both axes" is the next spec.
2. **Tampered-axis curation** — the L4 result says tampered needs paired contrasts, not
   nuisance matching: design the C2' variant that matches within-pair instead of dropping
   pairs (new spec).
3. **EV-AT / hardening on the winning composition** (talk to the Track A colleague first;
   new table per the separate-tables decision). The 8/255 defender still requires a custom
   encoder fine-tune.
4. **Backbone column** (L/14, DINOv2) over the same frozen legs and arms — cheap re-embeds.
5. RAISE-1k (form) would upgrade L3's negatives; SAGI-D admission note: its PowerPaint rows
   are already excluded (L4 conflict).

## How to resume, from the box

```bash
cd ~/Desktop/FILES/PROJECTS/trustfake/framework
git fetch && git checkout claude/tb-e3-curation-ladder-255c6d
set -a; . ./.env; set +a
PYTHONPATH=$PWD/src .venv/bin/python src/curate.py ladder   # resumes: every fit is a marker
```

Chain logs: `${LOGS_PATH}/tb_e3/` (fetch/, embed/, ladder.log, phase2.log, harden_fare.log,
attacks.log). Every stage is marker-gated and resumable; a step that succeeded with a bad
number is re-done by deleting its marker/JSON.
