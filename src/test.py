import os
from pathlib import Path

import hydra
import lightning
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from trustfake.instantiator import config_parsing, save_experiment_config
from trustfake.logging import add_handler, get_logger
from trustfake.metrics.calibration import calibrate_temperature
from trustfake.models.wrapper import BaseWrapper, MCDropoutWrapper
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

logger = get_logger("eval")

WRAPPERS = {
    "base": BaseWrapper,
    "mc_dropout": MCDropoutWrapper,
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

    logger.info("Starting testing...")
    trainer.test(eval_module, datamodule=datamodule)
    logger.info("Testing completed.")

    save_experiment_config(trainer, raw_cfg)


if __name__ == "__main__":
    run_eval_pipe()
