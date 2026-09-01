---
type: decision
title: Unmappable binary fakes are dropped, not multi-task-folded (H2)
status: current
as_of: 2026-09-01
source: "TB-E3 official run, H2 verdict (verdicts.json), pre-registered tie rule"
tags: [track-b, curation, labels, tb-e3]
links: [2026-09-01-tb-e3-curation-ladder, specs/tb-e3-curation-ladder]
---

**Decision.** When a source dataset's fake labels do not state the modality (fully-generated
vs locally-edited), those rows are DROPPED from 3-class training (C3a), not trained through a
binary side-objective (C3b). This is the project's mapping policy for any future combined
datamodule — including the colleague repo's `combined.py`, whose fold-into-synthetic default
is the naive C0 behaviour the ladder measured.

**Why.** The pre-registered rule: the C3 variant that wins on tampered detection under shift
becomes policy, ties go to strict-drop. The measured contrast was exactly zero (C3a == C3b to
4 dp on every leg) because the admitted shortlist has zero unmappable rows — every dataset
declares its modality per-image. So the tie rule fires, and the cheaper, simpler policy wins.

**Standing caveat.** The contrast is structural, not empirical: if a binary-fake environment
(e.g. a FakeClue-like source) is ever admitted, H2 must be re-run — the zero delta says
nothing about pools that actually contain unmappable rows.
