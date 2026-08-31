---
type: gotcha
title: limit_test takes a prefix, so the underlying order must be random
status: current
as_of: 2026-08-31
source: "caught smoke-testing the datamodules on real data on the box"
tags: [data, evaluation]
links: [fakeclue-splits-leak-ffpp-identities]
---

**Trap.** `limit_test` is documented across this repo as a prefix whose capped set is *"nested
inside the full one and describes the same population"*. That is only true when the underlying
order is random. FakeClue's json is clustered by category (every deepfake row, then every
satellite row...), and `assign_groups` originally returned indices sorted — so a capped test split
was **single-class**. Measured: the first 64 test rows were all one label. An AUROC over one class
is NaN, and a NaN is not something anyone notices missing in a table of fifteen conditions.

**Fix.** Both new datamodules shuffle within each role before the cap, seeded by `manifest_seed`
so the capped set stays a protocol constant. Shuffling changes order, never membership, so the
identity firewall is untouched. Verified after: FakeClue capped test came back
`{fake: 125, real: 75}`, So-Fake-OOD `{real, synthetic, tampered}` all present.

**Generalise:** any new dataset loader that supports a prefix cap needs the same treatment.
