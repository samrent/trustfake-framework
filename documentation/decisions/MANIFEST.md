# decisions

- [ood-thresholds-come-from-in-domain-calib.md](ood-thresholds-come-from-in-domain-calib.md) — a shifted eval must not fit its own thresholds; that is the failure being measured
- [track-a-and-track-b-are-separate-tables.md](track-a-and-track-b-are-separate-tables.md) — a different backbone is a different model; the two comparisons never share a table
- [depth-head-lives-in-the-model-group.md](depth-head-lives-in-the-model-group.md) — Track C's head is an opt-in ResNet kwarg on a separate forward path; three scorings are three eval runs; what the attack sees is a named setting
