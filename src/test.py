import os
from pathlib import Path

import hydra
import lightning
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from trustfake.instantiator import config_parsing, save_experiment_config
from trustfake.logging import add_handler, get_logger
from trustfake.metrics.calibration import calibrate_temperature
from trustfake.metrics.moderation import (
    ModerationPolicy,
    fit_thresholds,
    fit_uncertainty_gate,
)
from trustfake.models.wrapper import (
    BaseWrapper,
    EvidentialWrapper,
    MCDropoutWrapper,
)
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

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
    try:
        eval_module = ClassificationEvaluationModule.load_from_checkpoint(
            checkpoint_path=best_model_path,
            model=module_best,
            model_output_schema_cls=ClassificationModelOutput,
            num_classes=datamodule.num_classes,
            attack=cfg["attack"],
        )
    except Exception as e:
        logger.exception(f"Error loading model from checkpoint {best_model_path}: {e}")
        raise RuntimeError from e

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
                eval_module.model, calib_loader(), device=eval_module.device
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
            probs_list, targets_list, unc_list = [], [], []
            eval_module.model.eval()
            with torch.no_grad():
                for inputs, targets in calib_loader():
                    _, probs, _, uncertainty = eval_module.model(
                        inputs.to(eval_module.device)
                    )
                    probs_list.append(probs.detach().cpu().numpy())
                    targets_list.append(targets.detach().cpu().numpy())
                    unc_list.append(uncertainty.detach().cpu().numpy())
            probs = np.concatenate(probs_list)
            targets = np.concatenate(targets_list)
            uncertainty = np.concatenate(unc_list)
            real_class = cfg.get("real_class", 0)
            p_fake = 1.0 - probs[:, real_class]
            y_binary = (targets != real_class).astype(int)
            t_low, t_high = fit_thresholds(
                p_fake, y_binary, sla_residual_risk=cfg.get("moderation_sla", 0.05)
            )
            t_unc = fit_uncertainty_gate(
                uncertainty,
                clean_review_budget=cfg.get("moderation_review_budget", 0.10),
            )
            eval_module.moderation_policy = ModerationPolicy(
                t_low=t_low, t_high=t_high, real_class=real_class, t_unc=t_unc
            )
            logger.info(
                f"Moderation policy fitted on clean calib: t_low={t_low:.4f}, "
                f"t_high={t_high:.4f}, t_unc={t_unc:.4f}, "
                f"SLA residual risk <= {cfg.get('moderation_sla', 0.05):.0%}"
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
