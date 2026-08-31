"""
This module provides utility functions and classes for instantiating components
in TrustFake pipelines, including model, loss function, optimizer, scheduler,
trainer, and callbacks. It also includes a configuration parser for Hydra-based
training and evaluation setups.
"""

from pathlib import Path

import lightning
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic import BaseModel, model_validator
from torch import nn

from trustfake.logging import get_logger

logger = get_logger("pipe-init")

__all__ = [
    "ExperimentConfig",
    "Instantiator",
    "CallbacksHandler",
    "config_parsing",
    "save_experiment_config",
    "select_evaluation_condition",
]

# Each wrapper drives `uncertainty_score.update` differently.
WRAPPER_UNCERTAINTY_SCORE_TARGETS: dict[str, str] = {
    "base": "trustfake.metrics.uncertainty.probs.MultiClassMaxProbability",
    "mc_dropout": "trustfake.metrics.uncertainty.mc_dropout.MCDropoutPredictiveEntropy",
    "evidential": (
        "trustfake.metrics.uncertainty.evidential.EvidentialPredictiveEntropy"
    ),
}


class ExperimentConfig(BaseModel):
    seed: int
    name: str = MISSING
    wrapper: str = "base"
    uncertainty_score: str | None = None

    @model_validator(mode="after")
    def check_wrapper_uncertainty_score_pairing(self) -> "ExperimentConfig":
        expected_target = WRAPPER_UNCERTAINTY_SCORE_TARGETS.get(self.wrapper)
        if expected_target is None:
            msg = (
                f"Unknown wrapper '{self.wrapper}'. Available wrappers: "
                f"{list(WRAPPER_UNCERTAINTY_SCORE_TARGETS)}."
            )
            raise ValueError(msg)

        if self.uncertainty_score != expected_target:
            msg = (
                f"wrapper '{self.wrapper}' requires uncertainty_score "
                f"'{expected_target}', got '{self.uncertainty_score}'. Check the "
                "'wrapper' and 'uncertainty_score' entries in your config."
            )
            raise ValueError(msg)

        return self


class Instantiator:
    """
    Utility class to instantiate components from the configuration files.

    This class is designed to be used with Hydra training and evaluation configurations.
    It instantiate the following components:
    - Model
    - Loss function
    - Optimizer
    - Scheduler
    - Trainer
    - Callbacks
    - Attacks
    - Corruptions
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def instantiate_all(self):
        datamodule = self.instantiate_datamodule()
        calib_datamodule = self.instantiate_calib_datamodule()
        model = self.instantiate_model()
        uncertainty_score = self.instantiate_uncertainty_score()
        optimizer = None
        if hasattr(self.cfg, "optimizer"):
            optimizer = self.instantiate_optimizer(model)
        if hasattr(self.cfg, "scheduler"):
            if optimizer is None:
                msg = (
                    "A 'scheduler' is configured but no 'optimizer' is defined. "
                    "Please provide an 'optimizer' configuration."
                )
                logger.error(msg)
                raise ConfigAttributeError(msg)
            scheduler = self.instantiate_scheduler(optimizer)
        if hasattr(self.cfg, "loss"):
            loss = self.instantiate_loss()
        if hasattr(self.cfg, "attack"):
            attack = self.instantiate_attack()
        if hasattr(self.cfg, "corruption"):
            corruption = self.instantiate_corruption()
        if hasattr(self.cfg, "callbacks"):
            callbacks = self.instantiate_callbacks()
            if callbacks:
                cb_list = [cb for _, cb in callbacks.items()]
        else:
            cb_list = []
        if hasattr(self.cfg, "trainer"):
            trainer = self.instantiate_trainer(cb_list)
        return {
            "datamodule": datamodule,
            "calib_datamodule": calib_datamodule,
            "model": model,
            "optimizer": optimizer if hasattr(self.cfg, "optimizer") else None,
            "scheduler": scheduler if hasattr(self.cfg, "scheduler") else None,
            "uncertainty_score": uncertainty_score,
            "loss": loss if hasattr(self.cfg, "loss") else None,
            "attack": attack if hasattr(self.cfg, "attack") else None,
            "corruption": corruption if hasattr(self.cfg, "corruption") else None,
            "trainer": trainer if hasattr(self.cfg, "trainer") else None,
            "callbacks": callbacks if hasattr(self.cfg, "callbacks") else None,
        }

    def get_instatiated_cfg(self) -> DictConfig:
        """
        Returns the instantiated configuration with all components.
        This method resolves the configuration and instantiates all components
        defined in the configuration file.
        """
        cfg = OmegaConf.to_container(self.cfg, resolve=True)
        instances = self.instantiate_all()
        cfg.update(instances)
        return cfg

    def instantiate_calib_datamodule(self):
        """An optional SEPARATE datamodule to fit calibration on.

        Returns None unless a `calib_datamodule` group is selected, in which
        case temperature and the moderation policy are fitted on its calib
        split rather than the evaluation datamodule's. That is the only
        correct arrangement for a shifted evaluation set: fitting on the
        shifted data hides the exchangeability violation the condition is
        there to measure.
        """
        cfg = getattr(self.cfg, "calib_datamodule", None)
        if cfg is None:
            return None
        built = instantiate(cfg)
        factory = built.get("datamodule", None) if built else None
        if factory is None:
            return None
        logger.info("Calibration will be fitted on a separate calib datamodule")
        return factory()

    def instantiate_datamodule(self):
        logger.debug("Instantiating datamodule")

        try:
            datamodule_cfg = instantiate(self.cfg.datamodule)
        except ConfigAttributeError as e:
            msg = f"'datamodule' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        datamodule = datamodule_cfg.get("datamodule", None)
        if datamodule is None:
            msg = (
                "'datamodule' key is missing in the datamodule configuration file. "
                "Please provide a 'datamodule' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return datamodule()

    def instantiate_model(self) -> nn.Module:
        logger.debug("Instantiating model")

        try:
            model = instantiate(self.cfg.model)
        except ConfigAttributeError as e:
            msg = f"'model' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        model = model.get("model", None)
        if model is None:
            msg = (
                "'model' key is missing in the model configuration file. "
                "Please provide a 'model' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return model

    def instantiate_loss(self) -> nn.Module:
        logger.debug("Instantiating loss function")
        try:
            loss = instantiate(self.cfg.loss)
        except ConfigAttributeError as e:
            msg = f"'loss' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        loss = loss.get("loss", None)
        if loss is None:
            msg = (
                "'loss' key is missing in the loss configuration file. "
                "Please provide a 'loss' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return loss

    def instantiate_attack(self):
        logger.debug("Instantiating attack")
        try:
            attack = instantiate(self.cfg.attack)
        except ConfigAttributeError as e:
            msg = f"'attack' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        attack = attack.get("attack", None)
        if attack is None:
            msg = (
                "'attack' key is missing in the attack configuration file. "
                "Please provide an 'attack' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return attack

    def instantiate_corruption(self):
        """Instantiate the common-corruption evaluation condition.

        A sibling of `instantiate_attack`, not a branch of it: a corruption
        and an attack are consumed by the same evaluation slot but they are
        different threat models (see `trustfake.corruptions.abc`), so they
        get separate config groups and a run declares exactly one.
        """
        logger.debug("Instantiating corruption")
        try:
            corruption = instantiate(self.cfg.corruption)
        except ConfigAttributeError as e:
            msg = f"'corruption' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        corruption = corruption.get("corruption", None)
        if corruption is None:
            msg = (
                "'corruption' key is missing in the corruption configuration "
                "file. Please provide a 'corruption' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return corruption

    def instantiate_optimizer(
        self, model: nn.Module | lightning.LightningModule
    ) -> torch.optim.Optimizer:
        logger.debug("Instantiating optimizer")
        try:
            optimizer = instantiate(self.cfg.optimizer)

        except ConfigAttributeError as e:
            msg = f"'optimizer' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        optimizer = optimizer.get("optimizer", None)
        if optimizer is None:
            msg = (
                "'optimizer' key is missing in the optimizer configuration file. "
                "Please provide an 'optimizer' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return optimizer(model.parameters())

    def instantiate_uncertainty_score(self):
        logger.debug("Instantiating uncertainty score")
        try:
            uncertainty_score = instantiate(self.cfg.uncertainty_score)
        except ConfigAttributeError as e:
            msg = (
                "'uncertainty_score' attribute is missing in the configuration "
                f"file: {e}"
            )
            logger.exception(msg)
            raise e

        uncertainty_score = uncertainty_score.get("uncertainty_score", None)
        if uncertainty_score is None:
            msg = (
                "'uncertainty_score' key is missing in the uncertainty score "
                "configuration file. Please provide an 'uncertainty_score' "
                "configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return uncertainty_score

    def instantiate_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        logger.debug("Instantiating scheduler")
        try:
            scheduler = instantiate(self.cfg.scheduler)
        except ConfigAttributeError:
            logger.warning(
                "No scheduler defined in the configuration file. "
                "Consider setting one via the `scheduler` key in the config "
                "file. You can ignore this warning if you don't want to use "
                "a scheduler."
            )
            return

        scheduler = scheduler.get("scheduler", None)
        if scheduler is None:
            msg = (
                "'scheduler' key is missing in the scheduler configuration file. "
                "Please provide a 'scheduler' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return scheduler(optimizer=optimizer)

    def instantiate_trainer(
        self, callbacks: list[lightning.Callback]
    ) -> lightning.Trainer:
        logger.debug("Instantiating trainer")

        try:
            trainer_cfg = OmegaConf.to_container(self.cfg.trainer, resolve=True)

            trainer_cfg = instantiate(trainer_cfg)
        except ConfigAttributeError as e:
            msg = f"'trainer' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        try:
            trainer = trainer_cfg.trainer
        except ConfigAttributeError as e:
            msg = (
                "'trainer' key is missing in the trainer configuration file. "
                "Please provide a 'trainer' configuration."
            )
            logger.exception(msg)
            raise e

        try:
            loggers = trainer_cfg.get("logger", [])
            logger_list = [logger for _, logger in loggers.items()]
        except ConfigAttributeError:
            logger_list = []

        return trainer(logger=logger_list, callbacks=callbacks)

    def instantiate_callbacks(self) -> dict[str, lightning.Callback]:
        logger.debug("Instantiating callbacks")

        try:
            callbacks = instantiate(self.cfg.callbacks)
            return callbacks
        except ConfigAttributeError:
            return []


class CallbacksHandler:
    """
    Handler for callbacks defined in the hydra configuration files.

    This class provides a class method to retrieve a specific callback by name.
    Handled callbacks include:
    - model_checkpoint
    - early_stopping
    """

    HANDLED_CALLBACKS = {
        "model_checkpoint",
        "early_stopping",
    }

    @classmethod
    def get_callback(
        cls, callbacks: DictConfig, callback_name: str
    ) -> nn.Module | None:
        """
        Get a specific callback by name.
        """
        if callback_name not in cls.HANDLED_CALLBACKS:
            msg = f"Callback '{callback_name}' is not handled."
            logger.error(msg)
            raise ValueError(msg)

        try:
            return callbacks[callback_name]
        except KeyError:
            msg = f"Callback '{callback_name}' is not defined in the configuration."
            logger.warning(msg)
            return None


def config_parsing(cfg: DictConfig) -> tuple[DictConfig, DictConfig]:
    """
    Parses the training configuration and returns the experiment configuration
    and the instantiated configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object.

    Returns:
        tuple: A tuple containing the experiment configuration and the instantiated
            configuration.
    """
    experiment = cfg.get("experiment", None)
    if experiment is None:
        msg = (
            "Experiment configuration is missing. Please provide 'experiment' "
            "in the config."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    try:
        ExperimentConfig(
            **experiment,
            wrapper=cfg.get("wrapper", "base"),
            uncertainty_score=cfg.get("uncertainty_score", {})
            .get("uncertainty_score", {})
            .get("_target_", None),
        )
    except Exception as e:
        msg = f"Validation problem with the experiment configuration: {e}"
        logger.exception(msg)
        raise e

    try:
        experiment_cfg = OmegaConf.to_container(experiment, resolve=True)
    except Exception as e:
        msg = f"Failed to parse experiment configuration: {e}"
        logger.exception(msg)
        raise e

    instantiator = Instantiator(cfg)
    cfg = instantiator.get_instatiated_cfg()
    return experiment_cfg, cfg


def select_evaluation_condition(cfg: dict | DictConfig):
    """Pick the single evaluation condition a test run reports beside clean.

    An attack and a corruption occupy the same slot in
    `ClassificationEvaluationModule` but are different threat models -- a
    worst-case, eps-bounded perturbation against a distributional shift with
    no budget at all (see `trustfake.corruptions.abc`). The pipe keys its
    metrics, its confusion matrix and its storage file by the condition's
    name, so configuring both would not produce two columns: it would
    quietly report one of them and drop the other. Refusing is the only
    behaviour that cannot be misread.

    Args:
        cfg: The instantiated configuration.

    Returns:
        The attack, the corruption, or None for a clean-only run.

    Raises:
        ValueError: If both an attack and a corruption are configured.
    """
    attack = cfg.get("attack")
    corruption = cfg.get("corruption")
    if attack is not None and corruption is not None:
        msg = (
            f"Both an attack ('{attack.name}') and a corruption "
            f"('{corruption.name}') are configured; they are mutually exclusive "
            "evaluation conditions and one run reports one of them. Pass "
            "exactly one of '+attack=...' or '+corruption=...'."
        )
        logger.error(msg)
        raise ValueError(msg)
    return attack if attack is not None else corruption


def save_experiment_config(
    trainer: lightning.Trainer,
    cfg: DictConfig,
    filename: str = "experiment_config.yaml",
) -> None:
    """
    Save the fully resolved experiment configuration inside each of the
    trainer's logger directories (e.g. `test_lightning_logs/version_x`), so
    the parameters used for a run stay attached to its logged metrics.

    Args:
        trainer: The Lightning trainer whose loggers should receive the config.
        cfg: The raw (pre-instantiation) Hydra configuration for the run.
        filename: Name of the config file written into each logger's directory.
    """
    for trainer_logger in trainer.loggers:
        log_dir = Path(trainer_logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / filename)
        logger.info(f"Saved experiment configuration to: {log_dir / filename}")
