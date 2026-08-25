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
    ConfidenceAdversarialTrainingModule,
    ConfidenceRegularisedTrainingModule,
    EvidentialAdversarialTrainingModule,
    HybridAdversarialTrainingModule,
    MARTTrainingModule,
    PGDAdversarialTrainingModule,
    StandardTrainingModule,
    TRADESTrainingModule,
)

logger = get_logger("training-pipe")

WRAPPERS = {
    "base": BaseWrapper,
    "mc_dropout": MCDropoutWrapper,
    "evidential": EvidentialWrapper,
}

TRAINING_PIPES = {
    "standard": StandardTrainingModule,
    # Label-axis defences: the adversary tries to change the prediction.
    "pgd_at": PGDAdversarialTrainingModule,
    "trades": TRADESTrainingModule,
    "at_kl": HybridAdversarialTrainingModule,
    "mart": MARTTrainingModule,
    # Confidence-axis defences: the adversary tries to change the confidence
    # attached to an unchanged prediction -- the failure this harness measures.
    "at_conf": ConfidenceAdversarialTrainingModule,
    "conf_reg": ConfidenceRegularisedTrainingModule,
    # Evidential.
    "evidential_adversarial": EvidentialAdversarialTrainingModule,
}

# Arms sharing the adversarial-training scaffold (inner PGD, eps warm-up,
# robust validation).
_ADVERSARIAL_PIPES = (
    PGDAdversarialTrainingModule,
    TRADESTrainingModule,
    HybridAdversarialTrainingModule,
    MARTTrainingModule,
    ConfidenceAdversarialTrainingModule,
)
# `beta` weights a different term in each arm (TRADES: KL against natural CE;
# AT+KL: consistency KL against adversarial CE; MART: misclassification-weighted
# KL). One shared key would make three incomparable settings look like one.
_BETA_KEYS = {
    TRADESTrainingModule: "trades_beta",
    HybridAdversarialTrainingModule: "at_kl_beta",
    MARTTrainingModule: "mart_beta",
}
# Every arm except `standard` can take a weight-space inner maximisation.
_AWP_CAPABLE = _ADVERSARIAL_PIPES + (
    ConfidenceRegularisedTrainingModule,
    EvidentialAdversarialTrainingModule,
)


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
            "ikl_ema": cfg.get("ikl_ema", 0.9),
            "adv_eps": cfg.get("adv_eps", 8 / 255),
            "adv_steps": cfg.get("adv_steps", 10),
        }
    elif pipe_cls in _ADVERSARIAL_PIPES:
        pipe_kwargs = {
            "eps": cfg.get("adv_eps", 8 / 255),
            "steps": cfg.get("adv_steps", 10),
            "eps_warmup_epochs": cfg.get("adv_warmup_epochs", 0),
        }
        beta_key = _BETA_KEYS.get(pipe_cls)
        if beta_key is not None:
            pipe_kwargs["beta"] = cfg.get(beta_key, 6.0)
    elif pipe_cls is ConfidenceRegularisedTrainingModule:
        pipe_kwargs = {"lambda_reg": cfg.get("lambda_reg", 1.0)}

    if pipe_cls in _AWP_CAPABLE:
        pipe_kwargs["awp_gamma"] = cfg.get("awp_gamma", 0.0)
        pipe_kwargs["awp_warmup_epochs"] = cfg.get("awp_warmup_epochs", 0)

    # EVERY pipe, not just the adversarial ones. If the classical baselines
    # could be selected on robustness and EV-AT / conf_reg / standard could
    # not, the comparison would be between selection protocols rather than
    # between methods -- and asking for it on those arms used to die with
    # "Early stopping conditioned on metric val_robust_accuracy which is not
    # available", one epoch into the run.
    pipe_kwargs["robust_val_steps"] = cfg.get("robust_val_steps", 0)
    pipe_kwargs["robust_val_eps"] = cfg.get("robust_val_eps", None)

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
