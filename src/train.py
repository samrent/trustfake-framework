import os
from pathlib import Path

import hydra
import lightning
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from trustfake.instantiator import (
    CallbacksHandler,
    config_parsing,
    save_experiment_config,
)
from trustfake.logging import add_handler, get_logger
from trustfake.models.wrapper import (
    BaseWrapper,
    EvidentialWrapper,
    MCDropoutWrapper,
)
from trustfake.pipes.train import (
    EvidentialAdversarialTrainingModule,
    StandardTrainingModule,
)

logger = get_logger("training-pipe")

WRAPPERS = {
    "base": BaseWrapper,
    "mc_dropout": MCDropoutWrapper,
    "evidential": EvidentialWrapper,
}

TRAINING_PIPES = {
    "standard": StandardTrainingModule,
    "evidential_adversarial": EvidentialAdversarialTrainingModule,
}


@hydra.main(
    config_path="../configs/training", config_name="train_config", version_base=None
)
def run_train_pipe(cfg: DictConfig) -> None:
    """
    Main function to run the training pipe using Hydra configuration.

    Args:
        config: Hydra configuration object containing all necessary parameters.
    """

    add_handler(
        sink=Path(HydraConfig.get().runtime.output_dir)
        / f"{HydraConfig.get().job.name}.log"
    )

    seed = cfg["experiment"]["seed"]
    if seed is None:
        msg = "Seed is not set in the configuration"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.debug(f"Reset random state with seed {seed}")
    lightning.seed_everything(seed)

    raw_cfg = cfg
    exp_cfg, cfg = config_parsing(cfg)

    logger.info(f"Output directories set to: {cfg['output_dir']}")

    data_path = Path(os.environ.get("DATA_PATH"))
    if not data_path.exists():
        msg = (
            f"Data path {data_path} does not exist. Please check the DATA_PATH "
            "environment variable."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    datamodule = cfg["datamodule"]

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

    module = wrapper_cls(
        normalization_layer=datamodule.normalization_layer,
        model=cfg["model"],
        loss_fn=cfg["loss"],
        uncertainty_score=cfg["uncertainty_score"],
        **wrapper_kwargs,
    )

    pipe_name = exp_cfg.get("training_pipe", "standard")
    pipe_cls = TRAINING_PIPES.get(pipe_name)
    if pipe_cls is None:
        msg = f"Unknown training_pipe '{pipe_name}'. Available: {list(TRAINING_PIPES)}"
        logger.error(msg)
        raise ValueError(msg)

    pipe_kwargs = {}
    if pipe_cls is EvidentialAdversarialTrainingModule:
        pipe_kwargs = {
            "beta": cfg.get("beta", 1.0),
            "divergence_mode": cfg.get("rea_mode", "ikl"),
            "adv_eps": cfg.get("adv_eps", 8 / 255),
            "adv_steps": cfg.get("adv_steps", 10),
        }

    training_module = pipe_cls(
        model=module,
        num_classes=datamodule.num_classes,
        optimizer=cfg["optimizer"],
        scheduler=cfg.get("scheduler", None),
        **pipe_kwargs,
    )

    trainer: lightning.Trainer = cfg["trainer"]

    logger.info("Starting training...")
    trainer.fit(training_module, datamodule=datamodule)
    logger.success("Training completed.")

    ckpt_cb: ModelCheckpoint | None = CallbacksHandler.get_callback(
        cfg["callbacks"], "model_checkpoint"
    )
    if ckpt_cb is not None:
        best_model_path = ckpt_cb.best_model_path
        logger.info(f"Best model saved at: {best_model_path}")
    else:
        logger.warning("No model checkpoint callback found")

    save_experiment_config(trainer, raw_cfg)


if __name__ == "__main__":
    run_train_pipe()
