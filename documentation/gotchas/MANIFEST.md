# gotchas

Every one of these produces a plausible wrong number rather than an error.

- [fakeclue-label-convention-is-inverted.md](fakeclue-label-convention-is-inverted.md) — FakeClue says 0=fake, this project says 0=real; raw labels invert every detection metric
- [fakeclue-splits-leak-ffpp-identities.md](fakeclue-splits-leak-ffpp-identities.md) — 393 identities land in both roles on a naive split; 194 appear under both labels
- [limit-test-prefix-needs-a-shuffled-order.md](limit-test-prefix-needs-a-shuffled-order.md) — capped splits were single-class, and an AUROC over one class is NaN
- [freezing-a-backbone-must-not-use-no-grad.md](freezing-a-backbone-must-not-use-no-grad.md) — no_grad also kills input gradients, so PGD reports perfect robustness on an attackable model
- [binary-fold-goes-after-checkpoint-load.md](binary-fold-goes-after-checkpoint-load.md) — wrapping first prefixes state_dict keys with `inner.` and the load silently mismatches
- [clip-normalization-lives-in-the-model.md](clip-normalization-lives-in-the-model.md) — a datamodule Normalize on top double-normalizes with no error and no shape change
- [gpu-concurrency-is-negative-on-this-box.md](gpu-concurrency-is-negative-on-this-box.md) — two jobs are slower than one; run evaluations sequentially
- [fakeclue-train-and-test-share-every-identity.md](fakeclue-train-and-test-share-every-identity.md) — 100% of test identities appear in train; the published splits are contaminated
