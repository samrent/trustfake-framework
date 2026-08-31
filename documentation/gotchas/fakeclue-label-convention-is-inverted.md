---
type: gotcha
title: FakeClue's label convention is inverted relative to this project
status: current
as_of: 2026-08-31
source: "read from lingcco/FakeClue data_json/test.json"
tags: [data, fakeclue]
links: [fakeclue-dataset-properties]
---

**Trap.** FakeClue's own json uses `0 = fake, 1 = real`. This project uses `0 = real`
everywhere — `real_class=0`, `p_fake = 1 - P(real)`, and every per-class row. Loading FakeClue's
raw labels returns detection AUROCs, moderation decisions and recall rows that look plausible and
are **exactly backwards**. Nothing raises.

**Fix.** `trustfake.data.fake_clue` remaps at read time and keeps `FAKE_CLUE_FAKE_ID` /
`FAKE_CLUE_REAL_ID` as module constants so the mapping is auditable. Two tests pin both
directions. If you write another loader for this dataset, remap there too.
