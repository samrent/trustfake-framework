---
type: decision
title: OOD evaluation takes its thresholds from the in-domain calib split
status: current
as_of: 2026-08-31
source: "implemented as the calib_datamodule Hydra group, 2026-08-31"
tags: [evaluation, calibration, so-fake-ood]
links: [2026-08-31-track-b-first-results]
---

**Decision.** `SoFakeOODDataModule.calib_dataloader` **raises**. Temperature and the WP4 moderation
policy are fitted on SID-Set's in-domain calib split and applied unchanged to the shifted set, via
the `calib_datamodule` Hydra group (`calib_datamodule=sid_set`). `src/test.py` logs when the calib
source differs from the evaluation datamodule.

**Why.** Selective prediction and conformal risk control rest on exchangeability; distribution
shift breaks it. A threshold refitted *after* the shift papers over exactly the breakage the OOD
condition exists to expose. It is also the honest deployment model: you calibrate on the data you
had and serve on whatever arrives.

**Alternatives rejected.** *Fit on the OOD data* — hides the effect being measured. *Skip
calibration entirely (T=1.0)* — silently comparable to nothing, since every other condition in the
table is calibrated. *Let the datamodule quietly serve a calib split* — the refusal is what forces
the caller to state which calib split the thresholds came from, which a reader of the table needs.

**Cost, recorded honestly:** the refusal initially broke both OOD legs of the chain, because the
guard was written before the path it prescribes existed. Guard and mechanism have to ship together.
