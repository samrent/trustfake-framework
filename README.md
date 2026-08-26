# TrustFake

## Requirements

- Git
- Python 3.11+ (for the `uv` setup)
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) (for the `docker` setup)

## Installation

For installation, you can either use [`uv`](https://github.com/astral-sh/uv) for a local environment, or [`docker`](https://www.docker.com/) for a fully isolated one.

**Step 1: Clone the repository**
```bash
git clone git@github.com:NicolasSournac/TrustFake.git
cd TrustFake
```

**Step 2: Configure environment variables**
```bash
cp .env.example .env
```
This file defines `DATA_PATH`, `OUTPUT_PATH`, `LOGS_PATH` and `CONFIGS_PATH`.

### Option A - Recommended: Using `docker` (requires Docker and Docker Compose)

```bash
# Build the docker image
make build

# Start the dev container in the background
make up

# Open a shell inside the container
make shell
```

### Option B: Using `uv` (requires Python 3.11+)

```bash
# Install uv if you haven't already
pip install uv

# Create the locked project environment
uv sync --frozen

# Activate the environment
source .venv/bin/activate
```

You're all set. Run any command directly, e.g.:
```bash
uv run python -c "import torch; print(torch.__version__)"
```

## Usage

### Training and evaluation

Pipelines are configured with [Hydra](https://hydra.cc/) from the YAML files in [`configs/training/`](configs/training/). `experiment.name` is required; every other config value (model, optimizer, dataset, etc.) can be overridden on the command line.

**Train a model:**
```bash
python src/train.py experiment.name=my_experiment
```
This trains the model defined in [`configs/training/train_config.yaml`](configs/training/train_config.yaml) (ResNet-18 on SID_Set by default) and writes checkpoints/logs under `${OUTPUT_PATH}/my_experiment/<model>/s<seed>/`.

**Evaluate a trained model:**
```bash
python src/test.py experiment.name=my_experiment
```
This loads the best checkpoint saved by the matching training run (same `experiment.name`, model and seed) and reports classification, failure-detection, selective-classification and calibration metrics (AURC/AUGRC/E-AURC as block-size-weighted means over distinct operating points, tie blocks collapsed -- see [`selective_classification.py`](src/trustfake/metrics/evaluation/selective_classification.py) for the convention). Add `+attack=fgsm` to also evaluate robustness under an adversarial attack:
```bash
python src/test.py experiment.name=my_experiment +attack=fgsm
```

See [`jobs/train_resnet18.sh`](jobs/train_resnet18.sh) for a full example. Prefix commands with `uv run` if you're using the `uv` setup instead of Docker.

### Data splits

Splits are governed by a shard-level manifest ([`src/trustfake/data/manifest.py`](src/trustfake/data/manifest.py)): **fit** comes from `train-*` parquet shards, **calib** and **test** from disjoint `validation-*` shards, and the model-selection slice (what Lightning sees as `val`) is carved from fit at row level. Early stopping and checkpointing therefore never see the rows that `src/test.py` reports on, and post-hoc quantities (e.g. a temperature) get their own `calib` split. The shard assignment is a function of `datamodule.manifest_seed` -- a project constant, deliberately independent of `experiment.seed`, so the reported split never moves with the training seed. Choose the shard budget with `datamodule.profile` (`smoke | full | sweep | train | train_holdout | all | all_holdout`). `all` uses every train shard on disk rather than a fixed count, so it does not silently stop covering the dataset when shards are added; `all_holdout` reserves six of them, sealed and disjoint from fit. `sweep` is small everywhere, for comparing many configurations rather than reporting any of them.

The official SID-Set test split is withheld by the dataset authors; everything called "test" here is carved from the validation split. Reports must say so (`trustfake.data.SPLIT_PROVENANCE`).

The datamodule reads the parquet shards directly from `${DATA_PATH}/sid_set` -- fetch them with [`jobs/download_sidset.sh`](jobs/download_sidset.sh).

### Long runs: resume and precision

Training resumes from `last.ckpt` when one exists under the experiment's
output directory. On by default, because a full-dataset adversarial arm is
13+ hours on a single card and an interruption at hour 12 should cost an
epoch rather than the run:

```bash
resume: false            # force a fresh start
ckpt_path: /path/to.ckpt # resume a specific checkpoint instead of the newest
```

Mixed precision is `bf16-mixed` (`trainer.trainer.precision`). Training here
is GPU-bound -- measured at 90-99% utilisation on a 3090 -- so it is worth
roughly 1.7x. bf16 rather than fp16 deliberately: no loss scaling to tune,
and no silent overflow in an adversarial inner loop, where gradients are
taken with respect to the *input* and sit outside the range weight gradients
occupy. The evidential head is computed in fp32 regardless, since it is an
`exp -> log -> softmax` chain where an overflow surfaces far from its cause.

Measured throughput is ~415 img/s for standard training, so the full dataset
(~210k images, 12 epochs) is ~1.7h for `standard`, ~13.5h for PGD-7 and ~15h
for TRADES.

### Comparing methods: `src/sweep.py`

Runs a set of arms under one protocol and ranks them on **confidence
resilience** -- mean failure-detection AUROC under `ace_uint8` and
`overconf`, subject to a clean-accuracy floor -- rather than on robust
accuracy, which is the crowded axis this project does not try to win.

```bash
python src/sweep.py --strategy ACE --epochs 12 --profile train --subsample 1000
python src/sweep.py --rank-only          # re-rank what is already on disk
python src/sweep.py --strategy E --only ev_at   # run one arm by name
```

Four grids, from `SWEEP-STRATEGIES.md`:

| grid | arms |
|---|---|
| `A` | `standard`, `pgd_at`, `trades` |
| `B` | `at_kl`, `mart` -- the hybrids that document recommends skipping |
| `C` | `at_conf`, `conf_reg` -- the confidence-targeted defences |
| `E` | EV-AT and its ablation ladder (see below) |

Every arm is pinned at eps = 8/255 with 7 inner steps. `--subsample N` caps
the test split to its first N rows -- a prefix, so a capped set is nested in
the full one and the two describe the same population, which is what lets
Square (~7h per arm on the full split) sit in the same table as a clean
evaluation. `calib` is never capped: every condition must be read against
thresholds fitted on the same calibration data.

Training is skipped when a checkpoint exists, so adding evaluation conditions
does not pay for the expensive half twice.

**The EV-AT ladder** (grid `E`) isolates one component per rung, which is the
question an independent harness can ask that a method's own authors cannot
easily ask of their own paper:

| arm | what it isolates |
|---|---|
| `ev_only` | evidential head, **no adversary** |
| `ev_at_b0` | adversary, REA off (`beta=0`) |
| `ev_at` | full: adversary + REA |
| `ev_at_awp` | + weight-space perturbation |
| `ev_at_kl` / `ev_at_l2` | same `beta`, different discrepancy |

`pgd_at` from grid `A` is the fourth corner: adversarial but not evidential.

Sweep numbers **rank**; they do not report. Re-run the winner at
`--profile train` (or `all`) with the cap removed before quoting anything.

### Calibration

Temperature scaling is fitted on the `calib` split (NLL-minimising, one scalar
`T`) and frozen before scoring, as the calibration baseline the harness
benchmarks against. `calib` comes from validation shards disjoint from `test`,
so this cannot touch the reported split. Reported **ECE / NLL / Brier** (clean
and adversarial) reflect the fitted `T`; disable with `calibrate=false` to
report the raw model.

`T > 0` cannot change the argmax, so accuracy is untouched -- but unlike a
2-class model (where MSP is monotone in the single logit margin and temperature
cannot reorder samples), with three classes temperature *can* change the
confidence ranking. Calibration is therefore a live variable here, which is
what makes it a meaningful baseline for a selective-classification method.

### Jupyter notebooks

Start a Jupyter Lab server with access to the project environment and the [`notebooks/`](notebooks/) folder. This allows you to run and edit the notebooks directly in your browser.

[`selective_risk_under_attack.ipynb`](notebooks/selective_risk_under_attack.ipynb)
is the demonstrator for what this project is about: it trains a small detector
on synthetic data (no dataset download, about a minute on CPU) and shows a
confidence attack leaving accuracy bit-identical while the risk-coverage curve
inverts and the frozen WP4 policy degrades on both axes at once. Start there if
you want the argument before the API.

It then walks the rest of the harness on the same toy model: the twenty-attack
taxonomy and why `uses_labels` decides whether two rows are comparable; the
witness guarantee that stops a minimum-norm attack reporting robustness a
fixed-budget attack refutes; PGD-AT trained through the harness's own pipe, so
the "does a label-axis defence repair the confidence axis" question is asked
rather than asserted; and the SID-Set protocol -- the `width == height -> fake`
trap, the geometry controls, the shard-level firewall, and the profile ladder
from `smoke` to `all`. The dataset section degrades gracefully when the shards
are not present, so the whole notebook still runs on a fresh clone.

**Docker:**
```bash
make jupyter
```
Then open [http://localhost:8800](http://localhost:8800) in your browser.

**uv:**
```bash
uv run jupyter lab
```

### Code quality

Lint and format the codebase inside the dev container:
```bash
make dev-ruff-check    # Check for lint errors
make dev-ruff-fix      # Check and auto-fix
make dev-ruff-format   # Format the code
```

### Other useful commands

```bash
make help        # List all available commands
make logs        # Follow all container logs
make env-info     # Display the user/UID/GID used by the containers
make down      # Stop and remove the containers once you're done
```

## Adversarial attacks

Attacks live in [`src/trustfake/attacks/`](src/trustfake/attacks/) and implement the `AdversarialAttack` interface defined in [`abc.py`](src/trustfake/attacks/abc.py): given a `TrustFakeWrapper` model and a batch of clean, unnormalized inputs, they return a perturbed batch of the same shape, within `eps` of the input and clipped to `[clip_min, clip_max]`.

**To add a new attack:**

1. Create a class in `src/trustfake/attacks/` that subclasses `AdversarialAttack`, implements `name` and `__call__`, and keeps its perturbation within `self.eps` (use the inherited `self._clamp()` to enforce `clip_min`/`clip_max`). Use `fgsm.py` as a reference implementation. An attack that produces metadata (per-sample epsilon, an accept-check forward) overrides `run` instead, returns an `AttackResult`, and implements `__call__` as `self.run(...).perturbed` -- see `ace.py`; the evaluation pipe then scores the accept-check forward directly instead of re-running the model.
2. Declare its taxonomy by overriding the class attributes that differ from the defaults — `family`, `direction`, `uses_labels`, `norm`, `minimum_norm` (see `AttackFamily` in `abc.py`). The defaults describe a fixed-budget, label-free, L∞ prediction attack, so only the attacks that differ have to say so. This is what lets a results table be grouped by threat family and lets a reader tell which rows assumed an attacker who holds the ground truth.
3. Export it from [`src/trustfake/attacks/__init__.py`](src/trustfake/attacks/__init__.py). `attack_registry()` picks it up from there automatically.
4. Add a Hydra config for it under [`configs/training/attack/`](configs/training/attack/) (see `fgsm.yaml`), so it can be selected with `+attack=<name>` when running `src/test.py`. Encode any parameter that changes the threat model into the attack's `name` (`ace` vs `ace_uint8`), not just the config filename -- `name` is what keys the log directory and the metric prefixes, so two configs sharing one silently overwrite each other. `tests/attacks/test_config_registry.py` enforces this.
5. Register it in `ATTACKS` in [`tests/attacks/test_attack_contracts.py`](tests/attacks/test_attack_contracts.py) (see below) to get it covered by the validation suite. A minimum-norm attack also needs a test that it actually minimises — the contract battery only checks that it stays inside a budget, and a stalled min-norm attack passes that while reporting robustness the model does not have.

**To validate an implementation:** [`tests/attacks/test_attack_contracts.py`](tests/attacks/test_attack_contracts.py) runs the same battery of checks against every attack in its `ATTACKS` list — that it stays within its `eps` L∞ ball, stays within `[clip_min, clip_max]`, doesn't mutate the input tensor or the model's weights, restores the model's training mode, returns a detached output, actually perturbs the input, is a no-op at `eps=0`, and is deterministic for a fixed model/input. Run it with:
```bash
make dev-test-attacks
```
or, if you're using the `uv` setup instead of Docker:
```bash
uv run pytest tests/attacks -v
```
These checks catch the common ways an attack implementation goes wrong, but are not a substitute for attack-specific tests (e.g. checking that FGSM actually moves in the sign-of-gradient direction).

### Available attacks

Selected with `+attack=<name>` when running `src/test.py`. Two families: a
**confidence-targeted** attack collapses selective risk while leaving accuracy
untouched (label preservation is a constraint), whereas a
**prediction-targeted** attack collapses it as a side effect of destroying
accuracy. Reporting both under one "robustness" heading is what makes the
first one invisible, so the family is data on the attack
(`trustfake.attacks.attack_registry()` catalogues the classes; `describe(attack)` reports one configured instance), not prose in this table.

`eps` is an L∞ budget except for the **minimum-norm** attacks (DeepFool, C&W,
BB, PDPGD, FAB), where it is a cap on the *result* and the quantity to report
is `AttackResult.l2_norm` — the norm the attack actually needed — together
with `AttackResult.success`, since a minimum-norm attack that fails on a
sample returns it unperturbed rather than inventing a perturbation.

| name | family | reference | notes |
|---|---|---|---|
| `fgsm` | prediction | Goodfellow et al. 2015 | single step |
| `bim` | prediction | Kurakin et al. 2017 | iterative, no random start |
| `pgd` | prediction | Madry et al. 2018 | random start, seeded |
| `pgd_l2` | prediction | Madry et al. 2018 | fixed budget in **L2**, not L∞ |
| `deepfool` | prediction | Moosavi-Dezfooli et al. 2016 | min-norm; `eps` is an L2 cap |
| `cw` | prediction | Carlini & Wagner 2017 | L2; `eps` is an L2 cap |
| `bb` | prediction | Brendel & Bethge 2019 | min-norm; boundary walk, adaptive trust region |
| `pdpgd` | prediction | Matyasko & Chau 2021 | min-norm; primal-dual proximal, L∞ or L2 |
| `tr` | prediction | Yao et al. 2019 | trust-region, adaptive step |
| `a3` | prediction | Liu et al. 2022 | adaptive init + online discarding |
| `apgd` | prediction | Croce & Hein 2020 | via `autoattack` |
| `fab` | prediction | Croce & Hein 2020 | min-norm; via `autoattack` |
| `square` | prediction | Andriushchenko et al. 2020 | query-based; via `autoattack` |
| `autoattack` | prediction | Croce & Hein 2020 | ensemble; class-count-safe composition |
| `uncertainty_fgsm` | uncertainty | Disrupting Deep Uncertainty Estimation | label-free; attacks the uncertainty score |
| `ace` | confidence | Galil & El-Yaniv 2021 | per-sample eps search, accept test |
| `ace_uint8` | confidence | Galil & El-Yaniv 2021 | ACE on the 1/255 grid: the file-upload threat model |
| `param_ace` | confidence | Buerger et al. 2024 (arXiv:2405.13922) | (η,ω)-ACE family |
| `overconf` | confidence | Ledda et al. 2025 | label-free, label-preserving |
| `underconf` | confidence | Ledda et al. 2025 | label-free; toward max entropy |
| `evidence_pgd` | evidence | EV-AT (arXiv:2607.03075) | maximises Dirichlet drift; needs `wrapper=evidential` |

`bb`, `pdpgd` and `a3` are **native reimplementations**. Each is otherwise
only available from a research repository that is not on PyPI (or, for BB, via
a heavyweight `foolbox` dependency), and a vendored attack that cannot be
verified offline is the worst kind of dependency here: an attack that is
subtly weak does not fail, it reports robustness the model does not have.
Where an implementation departs from its reference the module docstring says
so and a test pins the consequence — see
`tests/attacks/test_min_norm_attacks.py`, which cross-checks the three
minimum-norm attacks against each other precisely because a stalled one still
returns a plausible number.

The `autoattack`-package wrappers drive the model through a logits adapter and
seed their randomised components for determinism; the ensemble excludes the
targeted stages, which read 3rd/4th-largest logits and cannot run on a
few-class detector (see the wrapper docstring). Attacks with an accept test or
a min-norm search return an `AttackResult` carrying the per-sample effective
epsilon and the accept-check logits, which the evaluation pipe scores directly.

**Ground truth is opt-in.** `fgsm`, `bim`, `pgd` and `pgd_l2` attack the
model's *own* clean prediction by default, which is the realisable threat
model — an attacker in production does not hold the labels. `use_labels=true`
switches them to the supplied ground truth, a strictly stronger attacker and
therefore a different row in a results table, not a variant of the same one.
`describe(attack)` records which mode a row was produced in -- read it off the instance you ran, not off the class default, because for a parameterised attack the two can differ on exactly that field.

### Common corruptions

Selected with `+corruption=<name>`, mutually exclusive with `+attack=`. The
keystone benchmark reports clean + adversarial + common-corruption, and a
detector that survives an L∞ ball but not a JPEG re-encode is not deployable:
re-encoding is what every platform does to every image it serves.

| name | condition |
|---|---|
| `jpeg` | JPEG re-encode (`jpeg_q40`) |
| `webp` | WebP re-encode (`webp_q80`) |
| `downscale` | downscale and back up (`downscale_2`) |
| `gaussian_noise` | additive Gaussian noise |
| `gaussian_blur` | Gaussian blur |

A corruption is a distributional-shift condition, not a bounded perturbation,
so `eps` is `inf` and the L∞ contract does not apply — they are deliberately
kept out of the eps-ball battery in `tests/attacks/`. Corruptions are applied
to the model input (post-resize), the ImageNet-C convention; see
`src/trustfake/corruptions/abc.py` for why, and for what that does *not*
model.

## Methods and training pipelines

The training pipeline is chosen with `experiment.training_pipe`:

| pipe | axis | what | key config |
|---|---|---|---|
| `standard` | — | ordinary training | — |
| `pgd_at` | label | PGD adversarial training (Madry 2018) | `adv_eps`, `adv_steps`, `adv_warmup_epochs` |
| `trades` | label | TRADES (Zhang 2019) | `trades_beta`, `adv_eps`, `adv_steps` |
| `at_kl` | label | AT + consistency KL (hybrid) | `at_kl_beta`, `adv_eps`, `adv_steps` |
| `mart` | label | MART (Wang 2020) | `mart_beta`, `adv_eps`, `adv_steps` |
| `at_conf` | **confidence** | AT against the *confidence* attack | `adv_eps`, `adv_steps` |
| `conf_reg` | **confidence** | penalty on confident mistakes, no adversary | `lambda_reg` |
| `evidential_adversarial` | evidence | Evidential Adversarial Training (EV-AT) | `beta`, `rea_mode`, `ikl_ema`, `adv_eps`, `adv_steps` |

The **axis** column is the one that matters. Every classical arm defends the
*label* axis: it assumes the adversary wants to change the prediction. But the
attack that breaks a moderation layer does not change the prediction at all —
ACE and the over-confidence attack leave accuracy bit-identical and move only
the confidence attached to it. `at_conf` and `conf_reg` are the two arms aimed
at that, and they are the project's own contribution rather than ports:

- **`at_conf`** runs the over-confidence attack as its *inner maximisation* —
  freeze the model's own prediction, then inflate confidence in it — and asks
  the outer cross-entropy to be right on those inputs anyway. Where the model
  would have been confidently wrong, the two disagree and the gradient is large.
- **`conf_reg`** adds `λ·mean(max_k p_k · 1[wrong])`: a direct penalty on
  confident mistakes, with no inner attack, so it costs one forward per step.
  It optimises the confidence *ranking* the moderation layer reads, which no
  accuracy-based objective can express.

Both are **non-adaptive** results unless an attacker is subsequently optimised
against them, and must be reported that way — a defence evaluated only against
the attack it was trained on is the standard way robustness claims dissolve
(Athalye et al. 2018). A negative result is still a result here: the question
of whether confidence adversarial training survives an adaptive attack is open.

**AWP** (Adversarial Weight Perturbation, Wu et al. 2020) is a *modifier*, not
an arm: `awp_gamma > 0` composes it with every pipe except `standard`,
including EV-AT, where the weight adversary attacks `L_EV + β·L_REA` rather
than a cross-entropy proxy. Report it as an ablation on top of an arm, never
as an arm. Ordering is the subtle part — the gradient is taken at `w + v` and
applied to `w` — and `tests/pipes/test_awp.py` pins it, because getting it
wrong yields a run that trains fine with no AWP in it.

**Model selection.** Selecting a defence arm on clean macro-F1 selects it on
exactly what it deliberately trades away. Set `robust_val_steps > 0` to also
compute `val_robust_accuracy` (one PGD run per validation batch) and
`selection_metric: val_robust_accuracy` so the checkpoint and early-stopping
callbacks use it. Off by default because it is not free.

**EV-AT** (arXiv:2607.03075) is the evidential method the harness benchmarks
against the MSP and temperature-scaling baselines. The backbone becomes an
evidential head (`wrapper=evidential`): evidence `e = softplus(logits)`,
Dirichlet `α = e + 1`, posterior mean `π̄ = α/S`, and the selective score is
the posterior entropy `u = H[Cat(π̄)]`. Training minimises the evidential loss
`L_EV` (`loss=evidential`) plus `β·L_REA`, where `L_REA` aligns the clean and
adversarial posteriors in log-Dirichlet space (IKL by default) and the
adversarial examples come from the evidence-targeted adversary. A full run:

```bash
python src/train.py experiment.name=ev_at \
  experiment.training_pipe=evidential_adversarial \
  wrapper=evidential loss=evidential \
  uncertainty_score=evidential_predictive_entropy
```

Report robustness at a forensic epsilon: an 8/255 ball can erase the
small-amplitude, high-frequency evidence a deepfake detector relies on and
collapse adversarial training, so lower `adv_eps` (e.g. `0.00784` = 2/255) or
ramp it with `adv_warmup_epochs`. Adversarial-training arms must be matched on
optimiser steps (same `max_epochs`/schedule), not wall-clock, to compare method
rather than budget.

## Selective moderation (WP4)

`src/test.py` fits a two-threshold moderation policy on the clean calib split
(minimise review rate subject to a residual-risk SLA) and freezes it, then
reports per condition — clean and adversarial — `coverage`, `review_rate`,
`residual_risk`, `missed_fake_rate` and `false_flag_rate`, plus an
uncertainty-gated two-axis variant. Two thresholds on `p(fake)` (= `1 − P(real)`
for the 3-class detector) rather than one on confidence, because moderation
costs are asymmetric. Residual risk is `NaN`, never `0.0`, on an empty
auto-decide zone, and an infeasible SLA degrades to review-everything. Tune with
`moderation_sla` and `moderation_review_budget`; disable with `moderate=false`.

## Trivial baselines and verification

`python src/baselines.py` reports the metadata floor every detector accuracy
must be read against — chiefly `width == height → fake`, computed from the
original image bytes over a split's shards. `trustfake.data.verify` checks the
split manifest is reproducible (deterministic, firewall-consistent) and that a
finished run left a checkpoint and saved config behind.

## Baselines

Pretrained baseline results are available [here](https://transfer.multitel.be/index.php/s/FrcitTqbe9Z48wG). Download the `out` folder and place it in the root directory of the project.

## Development

For development, if you use Docker and VSCode, you can attach a vscode instance to the running container. This allows you to edit files directly in the container environment as the project is mounted as a volume. You can use the following vscode extensions:
- [Contrainer Tools](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers)
- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

To contribute to the project and ensure code quality, you can setup pre-commit hooks.

If you use a virtual environment, you can run:
```bash
uv run pre-commit install
```
If you use Docker, you can run:
```bash
pre-commit install
```

For code formatting, we use [Ruff](https://docs.astral.sh/ruff/). An extension is also available for VSCode: [Ruff for VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). If you are not using the extension, Ruff is also part of the `pyproject.toml` dependencies, so you can run it from the command line:

```bash
uvx ruff check src
```
and
```bash
uvx ruff format src
```
to check and format the code, respectively. You can also automatically fix minor issues with:

```bash
uvx ruff check src --fix
```

Please refer to the `pyproject.toml` to see the linter configuration.

## Repository structure

```
configs/training/    Hydra configs: train_config.yaml / eval_config.yaml plus one
                      subfolder per component (model, datamodule, optimizer,
                      scheduler, loss, trainer, callbacks, attack, uncertainty_score)
src/train.py          Training entrypoint
src/test.py           Evaluation entrypoint
src/trustfake/
  data/                Datamodules (e.g. SID_Set) and visualization helpers
  models/
    torch/             Plain PyTorch model architectures (e.g. ResNet)
    wrapper/            Wraps a model with normalization, loss and uncertainty score
  attacks/             Adversarial attacks (e.g. FGSM) used during evaluation
  corruptions/         Common-corruption evaluation conditions (jpeg, webp, ...)
  losses/              Evidential loss and the log-Dirichlet discrepancy
  metrics/             Classification, failure-detection, calibration,
                        selective-classification and WP4 moderation metrics
  pipes/               Lightning modules driving training and evaluation
    train/             One module per defence arm, plus the AWP modifier
  pydantic/            Schemas validating model outputs
  instantiator.py      Builds all components from a Hydra config
  logging.py           Project-wide loguru setup
jobs/                 Example shell scripts running a full train + eval pipeline
notebooks/            Jupyter notebooks
docker/               Dockerfile and entrypoint for the dev container
```

A pipeline run (`train.py` / `test.py`) reads a Hydra config, uses `Instantiator` to build the datamodule/model/optimizer/etc. from it, then runs the corresponding module from `pipes/` on that setup.