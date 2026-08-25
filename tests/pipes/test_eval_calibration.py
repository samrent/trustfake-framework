"""End-to-end check that the evaluation pipe reports calibration metrics and
that temperature scaling flows through it. Runs a real (tiny) Lightning
Trainer on synthetic tensors -- no dataset, no network, CPU only.

This is the integration test for the eval-pipe metric threading: if any of
the calibration collections is mis-wired, the logged keys go missing here.
"""

import lightning as L  # noqa: N812
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.attacks import FGSM
from trustfake.metrics.moderation import ModerationPolicy
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

NUM_CLASSES = 3


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, NUM_CLASSES)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _wrapper():
    torch.manual_seed(0)
    return BaseWrapper(
        normalization_layer=nn.Identity(),
        model=_TinyNet(),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )


def _loader():
    torch.manual_seed(1)
    x = torch.rand(24, 3, 8, 8)
    y = torch.randint(0, NUM_CLASSES, (24,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _run(tmp_path, attack=None, temperature=1.0, moderation_policy=None):
    module = _wrapper()
    module.temperature = temperature
    eval_module = ClassificationEvaluationModule(
        model=module,
        model_output_schema_cls=ClassificationModelOutput,
        num_classes=NUM_CLASSES,
        attack=attack,
        moderation_policy=moderation_policy,
    )
    logger = L.pytorch.loggers.CSVLogger(save_dir=str(tmp_path))
    trainer = L.Trainer(
        accelerator="cpu",
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,  # attacks need grad
    )
    trainer.test(eval_module, dataloaders=_loader(), verbose=False)
    return trainer.callback_metrics


def test_calibration_metrics_are_reported(tmp_path):
    metrics = _run(tmp_path)
    for key in ("nat_ece", "nat_nll", "nat_brier"):
        assert key in metrics, f"missing {key} in {list(metrics)}"
        assert torch.isfinite(torch.as_tensor(metrics[key]))


def test_adversarial_calibration_metrics_are_reported(tmp_path):
    metrics = _run(tmp_path, attack=FGSM(eps=0.05))
    for key in ("nat_ece", "fgsm_ece", "fgsm_nll", "fgsm_brier"):
        assert key in metrics, f"missing {key} in {list(metrics)}"


def test_temperature_changes_calibration_but_not_accuracy(tmp_path):
    """The eval-pipe-level statement of the calibration property: raising T
    moves ECE/NLL/Brier while accuracy is untouched."""
    cold = _run(tmp_path / "t1", temperature=1.0)
    warm = _run(tmp_path / "t3", temperature=3.0)
    assert float(cold["nat_accuracy"]) == float(warm["nat_accuracy"])
    assert float(cold["nat_nll"]) != float(warm["nat_nll"])
    assert float(cold["nat_ece"]) != float(warm["nat_ece"])


def test_moderation_indicators_reported(tmp_path):
    """With a frozen policy, the eval pipe reports the WP4 indicators, and the
    two-axis (uncertainty-gated) variant when the policy carries a t_unc."""
    policy = ModerationPolicy(t_low=0.3, t_high=0.7, real_class=0, t_unc=0.5)
    metrics = _run(tmp_path, moderation_policy=policy)
    assert "nat_moderation_coverage" in metrics
    assert "nat_moderation_review_rate" in metrics
    assert "nat_moderation_false_flag_rate" in metrics
    assert "nat_moderation_2axis_review_rate" in metrics


def test_no_policy_means_no_moderation_keys(tmp_path):
    metrics = _run(tmp_path)  # moderation_policy=None
    assert not any("moderation" in k for k in metrics)
