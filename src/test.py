import os
from pathlib import Path

import hydra
import lightning
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from trustfake.instantiator import (
    config_parsing,
    save_experiment_config,
    select_evaluation_condition,
)
from trustfake.logging import add_handler, get_logger
from trustfake.metrics.calibration import calibrate_temperature
from trustfake.metrics.moderation import (
    ModerationPolicy,
    fit_thresholds,
    fit_uncertainty_gate,
)
from trustfake.models.torch import BinaryFoldClassifier
from trustfake.models.wrapper import (
    BaseWrapper,
    EvidentialWrapper,
    MCDropoutWrapper,
)
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput
from trustfake.utils import resolve_device

logger = get_logger("eval")

WRAPPERS = {
    "base": BaseWrapper,
    "mc_dropout": MCDropoutWrapper,
    "evidential": EvidentialWrapper,
}


@hydra.main(
    config_path="../configs/training", config_name="eval_config", version_base=None
)
def run_eval_pipe(cfg: DictConfig):
    """
    Main function to run the evaluation pipelines using Hydra configuration.

    Args:
        config: Hydra configuration object containing all necessary parameters.
    """
    add_handler(
        sink=Path(HydraConfig.get().runtime.output_dir)
        / f"{HydraConfig.get().job.name}.log"
    )

    raw_cfg = cfg
    exp_cfg, cfg = config_parsing(cfg)

    logger.info(f"Output directories set to: {cfg['output_dir']}")
    seed = exp_cfg["seed"]
    if seed is None:
        msg = "Seed is not set in the configuration"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.debug(f"Reset random state with seed {seed}")
    lightning.seed_everything(seed)

    data_path = Path(os.environ.get("DATA_PATH"))
    if not data_path.exists():
        msg = (
            f"Data path {data_path} does not exist. Please check the DATA_PATH "
            "environment variable."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    datamodule = cfg["datamodule"]

    trainer: lightning.Trainer = cfg["trainer"]
    save_dir = Path(trainer.default_root_dir)

    try:
        # Look for .ckpt files
        all_ckpts = list(save_dir.rglob("*.ckpt"))
        all_ckpts.sort()
    except Exception as e:
        logger.exception(f"Error accessing checkpoints in {save_dir}: {e}")
        raise RuntimeError from e

    if not all_ckpts:
        msg = f"No checkpoints found in {save_dir}. Please check the training process."
        logger.error(msg)
        raise FileNotFoundError(msg)

    best_model_path = next(ckpt for ckpt in all_ckpts if ckpt.name != "last.ckpt")

    logger.info(f"Best model checkpoint found at: {best_model_path}")

    wrapper_name = cfg.get("wrapper", "base")
    wrapper_cls = WRAPPERS.get(wrapper_name)
    if wrapper_cls is None:
        msg = f"Unknown wrapper '{wrapper_name}'. Available wrappers: {list(WRAPPERS)}"
        logger.error(msg)
        raise ValueError(msg)

    # The single evaluation condition reported beside clean: an attack
    # (+attack=...) or a corruption (+corruption=...), never both.
    condition = select_evaluation_condition(cfg)
    if condition is not None:
        logger.info(f"Evaluation condition: {condition.name} ({condition.family})")

    wrapper_kwargs = {}
    if wrapper_cls is MCDropoutWrapper:
        wrapper_kwargs["num_samples"] = cfg.get("num_samples", 20)
    if wrapper_cls is EvidentialWrapper:
        wrapper_kwargs["evidence_activation"] = cfg.get(
            "evidence_activation", "softplus"
        )

    # PyTorch model
    module_best = wrapper_cls(
        normalization_layer=datamodule.normalization_layer,
        model=cfg["model"],
        loss_fn=cfg["loss"],
        uncertainty_score=cfg["uncertainty_score"],
        **wrapper_kwargs,
    )
    # `real_class` is needed here as well as by the moderation policy below.
    # The detection AUROC folds every non-real class to "fake", so leaving it
    # at the constructor default would have `detection_auroc` measuring class 0
    # while every moderation indicator in the SAME table measured the
    # configured class -- a disagreement inside one report, and one that stays
    # invisible for as long as the configured value happens to be 0.
    real_class = cfg.get("real_class", 0)
    try:
        eval_module = ClassificationEvaluationModule.load_from_checkpoint(
            checkpoint_path=best_model_path,
            model=module_best,
            model_output_schema_cls=ClassificationModelOutput,
            num_classes=datamodule.num_classes,
            attack=condition,
            real_class=real_class,
        )
    except Exception as e:
        logger.exception(f"Error loading model from checkpoint {best_model_path}: {e}")
        raise RuntimeError from e

    # Fold a multi-class model onto a binary benchmark (FakeClue). Applied
    # AFTER the checkpoint loads, never before: wrapping the module first
    # would prefix every state_dict key with `inner.` and the load would fail
    # to match. p_fake = 1 - P(real), the repo-wide definition.
    if cfg.get("binary_fold", False):
        if datamodule.num_classes != 2:
            msg = (
                f"binary_fold=true but the datamodule reports "
                f"{datamodule.num_classes} classes; the fold produces exactly 2"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            "binary_fold: collapsing the model onto 2 classes "
            f"(real_class={real_class}). This MERGES synthetic and tampered -- "
            "the per-modality breakout is not available under this fold."
        )
        eval_module.model.model = BinaryFoldClassifier(
            eval_module.model.model, real_class=real_class
        )

    # Temperature scaling: fit on the calib split and freeze before scoring.
    # calib comes from validation shards disjoint from test (see
    # trustfake.data.manifest), so this cannot touch the reported split.
    if cfg.get("calibrate", True):
        datamodule.setup()
        calib_loader = getattr(datamodule, "calib_dataloader", None)
        if calib_loader is None:
            logger.warning(
                "Datamodule has no calib_dataloader; skipping temperature "
                "scaling (temperature stays 1.0)."
            )
        else:
            temperature = calibrate_temperature(
                eval_module.model, calib_loader(), device=resolve_device()
            )
            eval_module.model.temperature = temperature
            logger.info(f"Applied fitted temperature T = {temperature:.4f}")
    else:
        logger.info("Calibration disabled (calibrate=false); temperature = 1.0")

    # WP4 selective moderation: fit the policy on the clean calib split (after
    # temperature) and freeze it. calib is shard-disjoint from test.
    if cfg.get("moderate", True):
        calib_loader = getattr(datamodule, "calib_dataloader", None)
        if calib_loader is None:
            logger.warning("No calib_dataloader; skipping moderation fitting.")
        else:
            import numpy as np

            datamodule.setup()
            device = resolve_device()
            probs_list, targets_list, unc_list = [], [], []
            eval_module.model.to(device).eval()
            with torch.no_grad():
                for inputs, targets in calib_loader():
                    _, probs, _, uncertainty = eval_module.model(inputs.to(device))
                    probs_list.append(probs.detach().cpu().numpy())
                    targets_list.append(targets.detach().cpu().numpy())
                    unc_list.append(uncertainty.detach().cpu().numpy())
            probs = np.concatenate(probs_list)
            targets = np.concatenate(targets_list)
            uncertainty = np.concatenate(unc_list)
            p_fake = 1.0 - probs[:, real_class]
            y_binary = (targets != real_class).astype(int)
            # sla_missed_fake is the second, asymmetric SLA: residual risk
            # counts both error directions, so a policy can meet it while
            # auto-allowing a large share of the fakes. Left null when the
            # deployment has no missed-fake commitment to state.
            sla_missed_fake = cfg.get("moderation_sla_missed_fake", None)
            t_low, t_high = fit_thresholds(
                p_fake,
                y_binary,
                sla_residual_risk=cfg.get("moderation_sla", 0.05),
                sla_missed_fake=sla_missed_fake,
            )
            t_unc = fit_uncertainty_gate(
                uncertainty,
                clean_review_budget=cfg.get("moderation_review_budget", 0.10),
            )
            eval_module.moderation_policy = ModerationPolicy(
                t_low=t_low, t_high=t_high, real_class=real_class, t_unc=t_unc
            )
            missed_fake_note = (
                "not set" if sla_missed_fake is None else f"<= {sla_missed_fake:.0%}"
            )
            logger.info(
                f"Moderation policy fitted on clean calib: t_low={t_low:.4f}, "
                f"t_high={t_high:.4f}, t_unc={t_unc:.4f}, "
                f"SLA residual risk <= {cfg.get('moderation_sla', 0.05):.0%}, "
                f"SLA missed fake {missed_fake_note}"
            )
            if eval_module.moderation_policy.infeasible:
                logger.warning(
                    "Moderation SLA infeasible: policy degrades to review-everything."
                )

    logger.info("Starting testing...")
    trainer.test(eval_module, datamodule=datamodule)
    logger.info("Testing completed.")

    save_experiment_config(trainer, raw_cfg)


if __name__ == "__main__":
    run_eval_pipe()
