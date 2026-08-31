---
type: gotcha
title: Splitting FakeClue by row leaks FaceForensics++ identities
status: current
as_of: 2026-08-31
source: "measured on the real test split, 5000 rows"
tags: [data, fakeclue, leakage]
links: [fakeclue-dataset-properties]
---

**Trap.** FakeClue's 1168 deepfake rows come from only **666 FF++ identities** — mean 3.15 rows
per identity — and **194 identities appear under BOTH labels**, the same face present as real and
as fake. A row-level calib/test split therefore puts the same person on both sides, often on both
sides of the label, and what gets measured is identity memorisation rather than detection.
Measured: a naive split leaves **393 identities in both roles**.

FF++ frame directories encode this: a real clip is `.../frames/<id>`, a manipulated one is
`.../frames/<src>_<tgt>`, so one fake row belongs to *two* identities and can collide with real
rows of either. Identities also chain transitively (`a_b` and `b_c` bind a to c).

**Fix.** `assign_groups` runs union-find over identities (parent directory outside FF++) and deals
whole components. Measured after: **0 identities in both roles**.

**Second-order trap:** components are large and label-pure (`genimage/fake` alone is 916 rows), so
dealing them whole wrecks the class prior — 87.5% fake in calib against 53.7% in test, which
breaks any threshold fitted on calib. The deal is therefore stratified by label and greedy
largest-first by deficit. Where one component holds most of a label the prior **cannot** be
matched without breaking the firewall; that case warns rather than silently skewing.
