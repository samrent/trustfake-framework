# decisions

- [ood-thresholds-come-from-in-domain-calib.md](ood-thresholds-come-from-in-domain-calib.md) — a shifted eval must not fit its own thresholds; that is the failure being measured
- [track-a-and-track-b-are-separate-tables.md](track-a-and-track-b-are-separate-tables.md) — a different backbone is a different model; the two comparisons never share a table
- [tb-e3-mapping-is-strict-drop.md](tb-e3-mapping-is-strict-drop.md) — unmappable binary fakes are dropped (H2 tie rule); re-run H2 if a binary-fake env is admitted
