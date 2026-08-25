"""The design constraint, asserted: a corruption is scored by the EXISTING
evaluation pipe with no changes to it.

`ClassificationEvaluationModule` knows only `attack.name` and
`attack.run(model, inputs, targets) -> AttackResult`, so a corruption that
honours those two is a drop-in second condition. If that ever stops being
true, the failure would otherwise surface as a missing metric key in a real
run -- days of GPU time later.
"""

import lightning as L  # noqa: N812
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.corruptions import JPEGCompression
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


def test_corruption_is_reported_as_its_own_condition(tmp_path):
    corruption = JPEGCompression(quality=30)
    eval_module = ClassificationEvaluationModule(
        model=_wrapper(),
        model_output_schema_cls=ClassificationModelOutput,
        num_classes=NUM_CLASSES,
        attack=corruption,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        logger=L.pytorch.loggers.CSVLogger(save_dir=str(tmp_path)),
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,
    )
    trainer.test(eval_module, dataloaders=_loader(), verbose=False)
    metrics = trainer.callback_metrics

    # Clean and corrupted rows, side by side and separately keyed -- the
    # parameter is in the prefix, so a second quality cannot overwrite this.
    assert any(key.startswith("nat_") for key in metrics)
    assert any(key.startswith("jpeg_q30_") for key in metrics)
    assert "jpeg_q30_effective_eps_mean" in metrics
    # No accept-check forward, so no label-preservation claim is made.
    assert "jpeg_q30_label_preservation" not in metrics


def test_corruption_metrics_are_not_the_clean_ones(tmp_path):
    """A round-trip that silently did nothing would produce an identical
    corrupted row -- indistinguishable from a robust detector."""
    corruption = JPEGCompression(quality=5)
    eval_module = ClassificationEvaluationModule(
        model=_wrapper(),
        model_output_schema_cls=ClassificationModelOutput,
        num_classes=NUM_CLASSES,
        attack=corruption,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        logger=L.pytorch.loggers.CSVLogger(save_dir=str(tmp_path)),
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,
    )
    trainer.test(eval_module, dataloaders=_loader(), verbose=False)

    assert trainer.callback_metrics["jpeg_q5_effective_eps_mean"].item() > 0.0
