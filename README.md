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
This loads the best checkpoint saved by the matching training run (same `experiment.name`, model and seed) and reports classification, failure-detection and selective-classification metrics (AURC/AUGRC/E-AURC as block-size-weighted means over distinct operating points, tie blocks collapsed -- see [`selective_classification.py`](src/trustfake/metrics/evaluation/selective_classification.py) for the convention). Add `+attack=fgsm` to also evaluate robustness under an adversarial attack:
```bash
python src/test.py experiment.name=my_experiment +attack=fgsm
```

See [`jobs/train_resnet18.sh`](jobs/train_resnet18.sh) for a full example. Prefix commands with `uv run` if you're using the `uv` setup instead of Docker.

### Data splits

Splits are governed by a shard-level manifest ([`src/trustfake/data/manifest.py`](src/trustfake/data/manifest.py)): **fit** comes from `train-*` parquet shards, **calib** and **test** from disjoint `validation-*` shards, and the model-selection slice (what Lightning sees as `val`) is carved from fit at row level. Early stopping and checkpointing therefore never see the rows that `src/test.py` reports on, and post-hoc quantities (e.g. a temperature) get their own `calib` split. The shard assignment is a function of `datamodule.manifest_seed` -- a project constant, deliberately independent of `experiment.seed`, so the reported split never moves with the training seed. Choose the shard budget with `datamodule.profile` (`smoke | full | train | train_holdout`).

The official SID-Set test split is withheld by the dataset authors; everything called "test" here is carved from the validation split. Reports must say so (`trustfake.data.SPLIT_PROVENANCE`).

The datamodule reads the parquet shards directly from `${DATA_PATH}/sid_set` -- fetch them with [`jobs/download_sidset.sh`](jobs/download_sidset.sh).

### Jupyter notebooks

Start a Jupyter Lab server with access to the project environment and the [`notebooks/`](notebooks/) folder. This allows you to run and edit the notebooks directly in your browser.

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
2. Export it from [`src/trustfake/attacks/__init__.py`](src/trustfake/attacks/__init__.py).
3. Add a Hydra config for it under [`configs/training/attack/`](configs/training/attack/) (see `fgsm.yaml`), so it can be selected with `+attack=<name>` when running `src/test.py`.
4. Register it in `ATTACKS` in [`tests/attacks/test_attack_contracts.py`](tests/attacks/test_attack_contracts.py) (see below) to get it covered by the validation suite.

**To validate an implementation:** [`tests/attacks/test_attack_contracts.py`](tests/attacks/test_attack_contracts.py) runs the same battery of checks against every attack in its `ATTACKS` list — that it stays within its `eps` L∞ ball, stays within `[clip_min, clip_max]`, doesn't mutate the input tensor or the model's weights, restores the model's training mode, returns a detached output, actually perturbs the input, is a no-op at `eps=0`, and is deterministic for a fixed model/input. Run it with:
```bash
make dev-test-attacks
```
or, if you're using the `uv` setup instead of Docker:
```bash
uv run pytest tests/attacks -v
```
These checks catch the common ways an attack implementation goes wrong, but are not a substitute for attack-specific tests (e.g. checking that FGSM actually moves in the sign-of-gradient direction).

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
  metrics/             Classification, failure-detection and uncertainty metrics
  pipes/               Lightning modules driving training and evaluation
  pydantic/            Schemas validating model outputs
  instantiator.py      Builds all components from a Hydra config
  logging.py           Project-wide loguru setup
jobs/                 Example shell scripts running a full train + eval pipeline
notebooks/            Jupyter notebooks
docker/               Dockerfile and entrypoint for the dev container
```

A pipeline run (`train.py` / `test.py`) reads a Hydra config, uses `Instantiator` to build the datamodule/model/optimizer/etc. from it, then runs the corresponding module from `pipes/` on that setup.