import os
from abc import ABC

import lightning as pl
import numpy as np
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    ConfusionMatrix,
    MulticlassROC,
)

from trustfake.attacks import AdversarialAttack
from trustfake.logging import get_logger
from trustfake.metrics.calibration import get_calibration_metrics
from trustfake.metrics.evaluation import (
    get_detection_metrics,
    get_failure_detection_metrics,
    get_multiclass_classification_metrics,
    get_selective_classification_metrics,
)
from trustfake.metrics.moderation import ModerationPolicy
from trustfake.models.wrapper import TrustFakeWrapper
from trustfake.pydantic.model_output_schema import (
    ClassificationModelOutput,
)

logger = get_logger("evaluation")

__all__ = ["ClassificationEvaluationModule"]


class ClassificationEvaluationModule(ABC, pl.LightningModule):
    """
    Abstract evaluation module for classification pipelines.

    This class defines the common structure for evaluating classification models and
        implements the test step and metric logging.

    Subclasses must implement the `standard_metrics` property to specify the metrics to
      be used for evaluation as well as choosing corresponding model output schema
      for validation.
    """

    def __init__(
        self,
        model: TrustFakeWrapper,
        model_output_schema_cls: ClassificationModelOutput,
        num_classes: int,
        attack: AdversarialAttack | None = None,
        moderation_policy: ModerationPolicy | None = None,
        real_class: int = 0,
    ):
        """
        Args:
            model (TrustFakeWrapper): The model to evaluate.
            model_output_schema_cls (type(ClassificationModelOutput)):
            Pydantic model class for model output validation.
            The model output is four tensors:
                - logits: Raw model outputs (logits).
                - probs: Predicted class probabilities.
                - preds: Predicted class labels.
                - uncertainty: Uncertainty score.
            num_classes (int): The number of classes in the classification task.
            attack (AdversarialAttack | None): Optional adversarial attack to apply
                during evaluation.
            moderation_policy (ModerationPolicy | None): Frozen WP4 policy the
                deployment indicators are reported against, per condition.
            real_class (int): Index of the real class; every other class folds
                to fake for the binary detection AUROC. SID-Set uses 0.
        """
        super().__init__()
        self.model = model
        self._model_output_schema_cls = model_output_schema_cls
        self._num_classes = num_classes
        self._real_class = real_class
        self.attack = attack
        self.moderation_policy = moderation_policy

        # Classification metrics
        self.nat_classification_metrics = self.classification_metrics.clone(
            prefix="nat_"
        )
        self.nat_cm = self.confusion_matrix.clone()
        self.nat_roc_curve = self.roc_curve.clone()

        # Failure detection metrics
        self.nat_fd_metrics = self.fd_metrics.clone(prefix="nat_")

        # Detection metrics -- the AUROC of the TASK (does p(fake) rank fakes
        # above reals), which is a different question from fd_auroc (does
        # uncertainty rank the model's errors above its correct answers).
        # Reporting one under the other's name is the cheap embarrassing
        # error, so they are separate collections consuming separate inputs.
        self.nat_detection_metrics = self.detection_metrics.clone(prefix="nat_")

        # Selective classification metrics
        self.nat_selective_classification_metrics = (
            self.selective_classification_metrics.clone(prefix="nat_")
        )

        # Calibration metrics (ECE / NLL / Brier)
        self.nat_calibration_metrics = self.calibration_metrics.clone(prefix="nat_")

        self._storage: dict[str, dict[str, list]] = {
            "nat": {
                "uncertainties": [],
                "errors": [],
                "probs": [],
                "targets": [],
            },
        }

        if self.attack is not None:
            prefix = f"{self.attack.name}_"
            self.adv_classification_metrics = self.classification_metrics.clone(
                prefix=prefix
            )
            self.adv_cm = self.confusion_matrix.clone()
            self.adv_roc_curve = self.roc_curve.clone()
            self.adv_fd_metrics = self.fd_metrics.clone(prefix=prefix)
            self.adv_detection_metrics = self.detection_metrics.clone(prefix=prefix)
            self.adv_selective_classification_metrics = (
                self.selective_classification_metrics.clone(prefix=prefix)
            )
            self.adv_calibration_metrics = self.calibration_metrics.clone(prefix=prefix)
            self._storage[f"{self.attack.name}"] = {
                "uncertainties": [],
                "errors": [],
                "probs": [],
                "targets": [],
            }

    def training_step(self, batch, batch_idx):
        """
        Training step is not used in evaluation module.
        """
        msg = "Training step is not applicable for evaluation module."
        logger.error(msg)
        raise NotImplementedError(msg)

    def validation_step(self, batch, batch_idx):
        """
        Validation step is not used in evaluation module.
        """
        msg = "Validation step is not applicable for evaluation module."
        logger.error(msg)
        raise NotImplementedError(msg)

    def on_test_start(self):
        """
        Called at the start of the test phase to move metrics to the appropriate device.
        """
        self.nat_classification_metrics.to(self.device)
        self.nat_cm.to(self.device)
        self.nat_roc_curve.to(self.device)
        self.nat_fd_metrics.to(self.device)
        self.nat_detection_metrics.to(self.device)
        self.nat_selective_classification_metrics.to(self.device)
        self.nat_calibration_metrics.to(self.device)
        if self.attack is not None:
            self.adv_classification_metrics.to(self.device)
            self.adv_cm.to(self.device)
            self.adv_roc_curve.to(self.device)
            self.adv_fd_metrics.to(self.device)
            self.adv_detection_metrics.to(self.device)
            self.adv_selective_classification_metrics.to(self.device)
            self.adv_calibration_metrics.to(self.device)

    def _run_model(self, inputs: torch.Tensor):
        logits, probs, preds, uncertainty = self.model(inputs)
        return self._model_output_schema_cls(
            logits=logits, probs=probs, preds=preds, uncertainty=uncertainty
        )

    def _evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        classification_metrics: MetricCollection,
        failure_detection_metrics: MetricCollection,
        selective_classification_metrics: MetricCollection,
        calibration_metrics: MetricCollection,
        detection_metrics: MetricCollection,
        cm: Metric,
        roc_curve: Metric,
        storage_key: str,
        output: ClassificationModelOutput | None = None,
    ):
        """
        Runs the model on `inputs` (unless a precomputed `output` is given,
            e.g. derived from an attack's accept-check forward), updates the
            given metric collections against `targets`, and stores the
            per-sample errors/uncertainty under `self._storage[storage_key]`.
        """
        if output is None:
            output = self._run_model(inputs)
        classification_metrics.update(output.preds.long(), targets.long())
        cm.update(output.preds.long(), targets.long())
        roc_curve.update(output.probs, targets.long())

        errors = output.preds.long() != targets.long()
        failure_detection_metrics.update(output.uncertainty, errors)

        selective_classification_metrics.update(
            output.probs, targets.long(), output.uncertainty
        )
        calibration_metrics.update(output.probs, targets.long())
        detection_metrics.update(output.probs, targets.long())

        # --- Storage for post-hoc analysis ---
        self._storage[storage_key]["uncertainties"].append(
            output.uncertainty.detach().cpu()
        )
        self._storage[storage_key]["errors"].append(errors.detach().cpu())
        self._storage[storage_key]["probs"].append(output.probs.detach().cpu())
        self._storage[storage_key]["targets"].append(targets.detach().cpu())

    def test_step(self, batch: list[torch.Tensor, torch.Tensor], batch_idx):
        """
        Test step for evaluation.
        This method should be overridden by subclasses to implement custom evaluation
            logic.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.

        Returns:
        """
        inputs, targets = batch[0], batch[1]

        self._evaluate(
            inputs,
            targets,
            self.nat_classification_metrics,
            self.nat_fd_metrics,
            self.nat_selective_classification_metrics,
            self.nat_calibration_metrics,
            self.nat_detection_metrics,
            self.nat_cm,
            self.nat_roc_curve,
            storage_key="nat",
        )

        if self.attack is not None:
            with torch.enable_grad():
                result = self.attack.run(self.model, inputs, targets)

            # Prefer the logits from the attack's accept-check forward: a
            # re-forward on the perturbed batch is not always bit-identical
            # (batch-shape-dependent backends), so a boundary sample can
            # flip and misreport quantities the attack guarantees. Falls
            # back to a re-forward when the wrapper cannot derive its
            # uncertainty from a single forward (e.g. MC dropout).
            adv_output = None
            if result.accepted_logits is not None:
                derived = self.model.outputs_from_logits(result.accepted_logits)
                if derived is not None:
                    logits, probs, preds, uncertainty = derived
                    adv_output = self._model_output_schema_cls(
                        logits=logits,
                        probs=probs,
                        preds=preds,
                        uncertainty=uncertainty,
                    )
                else:
                    logger.warning(
                        f"{self.attack.name} supplied accepted logits, but "
                        f"{type(self.model).__name__} cannot derive its "
                        "uncertainty from a single forward; re-running the "
                        "model on the perturbed batch instead."
                    )

            self._evaluate(
                result.perturbed,
                targets,
                self.adv_classification_metrics,
                self.adv_fd_metrics,
                self.adv_selective_classification_metrics,
                self.adv_calibration_metrics,
                self.adv_detection_metrics,
                self.adv_cm,
                self.adv_roc_curve,
                storage_key=self.attack.name,
                output=adv_output,
            )
            self._log_attack_metadata(result)

    def _log_attack_metadata(self, result) -> None:
        """
        Log what the attack itself measured: label preservation against the
        clean prediction (from the accept-check forward when present -- for
        ACE this is exactly 1.0 by construction), and the mean per-sample
        L_inf perturbation actually applied.
        """
        prefix = f"{self.attack.name}_"
        if result.accepted_logits is not None and result.clean_preds is not None:
            preservation = (
                (result.accepted_logits.argmax(dim=1) == result.clean_preds)
                .float()
                .mean()
            )
            self.log(prefix + "label_preservation", preservation)
        if result.effective_eps is not None:
            self.log(prefix + "effective_eps_mean", result.effective_eps.mean())

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to compute and log metrics.
        """
        nat_classification_metrics = self.nat_classification_metrics.compute()
        self.log_dict(nat_classification_metrics)
        self.nat_classification_metrics.reset()
        fig, _ = self.nat_cm.plot()
        fig.savefig(self.logger.experiment.log_dir + "/nat_confusion_matrix.png")
        fig, _ = self.nat_roc_curve.plot()
        fig.savefig(self.logger.experiment.log_dir + "/nat_roc_curve.png")
        self.nat_cm.reset()
        self.nat_roc_curve.reset()
        nat_fd_metrics = self.nat_fd_metrics.compute()
        self.log_dict(nat_fd_metrics)
        self.nat_fd_metrics.reset()
        nat_selective_classification_metrics = (
            self.nat_selective_classification_metrics.compute()
        )
        self.log_dict(nat_selective_classification_metrics)
        self.nat_selective_classification_metrics.reset()
        nat_calibration_metrics = self.nat_calibration_metrics.compute()
        self.log_dict(nat_calibration_metrics)
        self.nat_calibration_metrics.reset()
        self.log_dict(self.nat_detection_metrics.compute())
        self.nat_detection_metrics.reset()

        if self.attack is not None:
            adv_classification_metrics = self.adv_classification_metrics.compute()
            self.log_dict(adv_classification_metrics)
            self.adv_classification_metrics.reset()
            fig, _ = self.adv_cm.plot()
            fig.savefig(
                self.logger.experiment.log_dir
                + f"/{self.attack.name}_confusion_matrix.png"
            )
            fig, _ = self.adv_roc_curve.plot()
            fig.savefig(
                self.logger.experiment.log_dir + f"/{self.attack.name}_roc_curve.png"
            )
            self.adv_cm.reset()
            self.adv_roc_curve.reset()
            adv_fd_metrics = self.adv_fd_metrics.compute()
            self.log_dict(adv_fd_metrics)
            self.adv_fd_metrics.reset()
            adv_selective_classification_metrics = (
                self.adv_selective_classification_metrics.compute()
            )
            self.log_dict(adv_selective_classification_metrics)
            self.adv_selective_classification_metrics.reset()
            adv_calibration_metrics = self.adv_calibration_metrics.compute()
            self.log_dict(adv_calibration_metrics)
            self.adv_calibration_metrics.reset()
            self.log_dict(self.adv_detection_metrics.compute())
            self.adv_detection_metrics.reset()

        # Selective moderation (WP4): apply the frozen policy per condition.
        self._log_moderation()

        # Save storage to disk for post-hoc analysis
        self._save_storage_to_disk()
        # Clear storage after saving to disk to free up memory
        for key in self._storage:
            for field in self._storage[key]:
                self._storage[key][field] = []

    def _log_moderation(self) -> None:
        """Apply the frozen moderation policy to each stored condition and log
        the deployment indicators. The uncertainty gate (two-axis rule) is
        reported alongside when the policy carries a t_unc."""
        if self.moderation_policy is None:
            return
        for key, store in self._storage.items():
            if not store["probs"]:
                continue
            probs = torch.cat(store["probs"]).numpy()
            targets = torch.cat(store["targets"]).numpy()
            uncertainty = torch.cat(store["uncertainties"]).numpy()
            prefix = "nat" if key == "nat" else self.attack.name
            r = self.moderation_policy.evaluate(probs, targets)
            for name in (
                "coverage",
                "review_rate",
                "residual_risk",
                "missed_fake_rate",
                "false_flag_rate",
            ):
                value = r[name]
                if value is not None and not np.isnan(value):
                    self.log(f"{prefix}_moderation_{name}", float(value))
            if self.moderation_policy.t_unc is not None:
                r2 = self.moderation_policy.evaluate(
                    probs, targets, uncertainty, use_gate=True
                )
                for name in ("review_rate", "residual_risk"):
                    value = r2[name]
                    if value is not None and not np.isnan(value):
                        self.log(f"{prefix}_moderation_2axis_{name}", float(value))

    def _save_storage_to_disk(self):
        """
        Save the stored indices, targets, and model outputs to disk
        at the end of the test epoch.
        """
        if not self.logger:
            logger.warning("Logger not found. Skipping storage saving.")
            return
        log_dir = self.logger.log_dir
        if log_dir is None:
            logger.warning("Logger log_dir is None. Skipping storage saving.")
            return
        os.makedirs(log_dir, exist_ok=True)
        for key in self._storage:
            uncertainties = torch.cat(self._storage[key]["uncertainties"], dim=0)
            errors = torch.cat(self._storage[key]["errors"], dim=0)
            path = os.path.join(log_dir, f"storage_{key}.pt")
            torch.save({"uncertainties": uncertainties, "errors": errors}, path)

    @property
    def classification_metrics(self) -> MetricCollection:
        """
        Abstract property to define the classification metrics for evaluation.
        Subclasses must implement this property to return a MetricCollection
            containing the desired metrics for evaluation.

        Returns:
            MetricCollection: A collection of metrics to be used for evaluation.
        """
        return get_multiclass_classification_metrics(num_classes=self._num_classes)

    @property
    def confusion_matrix(self) -> Metric:
        """
        Property that define the confusion matrix metric for evaluation.


        Returns:
            Metric: A metric instance for computing the confusion matrix.
        """
        return ConfusionMatrix(
            task="multiclass", num_classes=self._num_classes, normalize="true"
        )

    @property
    def roc_curve(self) -> Metric:
        """
        Property that define the ROC curve metric for evaluation.

        Returns:
            Metric: A metric instance for computing the ROC curve.
        """
        return MulticlassROC(num_classes=self._num_classes)

    @property
    def fd_metrics(self) -> MetricCollection:
        """
        Property that define the failure detection metrics for evaluation.

        Returns:
            MetricCollection: A collection of metrics to be used for failure detection.
        """
        return get_failure_detection_metrics()

    @property
    def calibration_metrics(self) -> MetricCollection:
        """
        Property that defines the calibration metrics (ECE / NLL / Brier).

        Returns:
            MetricCollection: A collection of calibration metrics.
        """
        return get_calibration_metrics(num_classes=self._num_classes)

    @property
    def detection_metrics(self) -> MetricCollection:
        """
        Property that defines the detection-task metrics (fake vs real).

        Returns:
            MetricCollection: AUROC of ranking fakes above reals by p(fake).
        """
        return get_detection_metrics(real_class=self._real_class)

    @property
    def selective_classification_metrics(self) -> MetricCollection:
        """
        Property that define the selective classification metrics for evaluation.

        Returns:
            MetricCollection: A collection of metrics to
            be used for selective classification.
        """
        return get_selective_classification_metrics()
