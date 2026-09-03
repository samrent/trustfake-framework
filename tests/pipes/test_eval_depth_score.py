"""The evaluation pipe with the depth-consistency wrapper, end to end.

A real (tiny) Lightning test loop on synthetic tensors, with the stub
teacher: the failure-detection and selective metrics must appear for the
clean and the attacked condition, the on-disk storage schema must be the
historical two-key one, the teacher must have scored the PERTURBED batch,
and the attack-scoring mode must change only what the attack sees. CPU,
no data, no network.
"""

from __future__ import annotations

import lightning as L  # noqa: N812
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trustfake.attacks import FGSM, QueryConfidence
from trustfake.depth import FakeDepthTeacher
from trustfake.metrics.uncertainty import (
    CombinedDepthScore,
    DepthConsistencyScore,
    MultiClassMaxProbability,
)
from trustfake.models.wrapper import BaseWrapper, DepthConsistencyWrapper
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput

NUM_CLASSES = 3
SIZE = 16


class _TwoHeaded(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(4 * SIZE * SIZE, NUM_CLASSES)
        self.depth_head = nn.Conv2d(4, 1, 3, padding=1, stride=2)

    def forward(self, x):
        return self.fc(self.trunk(x).flatten(1))

    def forward_with_depth(self, x):
        f = self.trunk(x)
        return self.fc(f.flatten(1)), self.depth_head(f)


class _SpyTeacher(FakeDepthTeacher):
    def __init__(self):
        super().__init__(output_size=SIZE // 2, input_size=SIZE, multiple=1)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return super().forward(x)


def _wrapper(score, teacher, **kw):
    torch.manual_seed(0)
    cls = BaseWrapper if isinstance(score, MultiClassMaxProbability) else None
    if cls is BaseWrapper:
        return BaseWrapper(
            normalization_layer=nn.Identity(),
            model=_TwoHeaded(),
            loss_fn=nn.CrossEntropyLoss(),
            uncertainty_score=score,
        )
    return DepthConsistencyWrapper(
        normalization_layer=nn.Identity(),
        model=_TwoHeaded(),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=score,
        teacher=teacher,
        **kw,
    )


def _loader(n=24):
    torch.manual_seed(1)
    x = torch.rand(n, 3, SIZE, SIZE)
    y = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _run(tmp_path, score, attack=None, **kw):
    teacher = _SpyTeacher()
    module = _wrapper(score, teacher, **kw)
    if isinstance(score, CombinedDepthScore):
        with torch.no_grad():
            msp, res = module.score_components(next(iter(_loader()))[0])
        score.fit_reference(msp, res)
    eval_module = ClassificationEvaluationModule(
        model=module,
        model_output_schema_cls=ClassificationModelOutput,
        num_classes=NUM_CLASSES,
        attack=attack,
    )
    logger = L.pytorch.loggers.CSVLogger(save_dir=str(tmp_path))
    trainer = L.Trainer(
        accelerator="cpu",
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,
    )
    trainer.test(eval_module, dataloaders=_loader(), verbose=False)
    return trainer.callback_metrics, teacher, logger.log_dir


def test_depth_score_flows_through_the_metrics_and_the_storage(tmp_path):
    metrics, teacher, log_dir = _run(tmp_path, DepthConsistencyScore(), FGSM(eps=0.05))
    for key in (
        "nat_fd_auroc",
        "nat_aurc",
        "fgsm_fd_auroc",
        "fgsm_aurc",
        "nat_accuracy",
    ):
        assert key in metrics, f"missing {key}"
    stored = torch.load(f"{log_dir}/storage_nat.pt")
    assert set(stored) == {"uncertainties", "errors"}
    assert stored["uncertainties"].shape == (24,)
    assert torch.isfinite(stored["uncertainties"]).all()
    assert teacher.inputs, "the teacher was never consulted"


def test_white_box_mode_runs_the_teacher_in_the_attack_and_rescores_the_batch(
    tmp_path,
):
    """Default mode: the attack itself calls the wrapper (teacher included),
    then the pipe re-forwards the perturbed batch, so the last teacher input
    of every batch is the attacked one -- not the clean one."""
    _, teacher, _ = _run(tmp_path, DepthConsistencyScore(), FGSM(eps=0.05))
    # per batch: clean forward, the attack's forward(s), the re-forward
    assert len(teacher.inputs) > 2 * 3
    clean_batches = list(_loader())
    last = teacher.inputs[-1]
    x_last = clean_batches[-1][0]
    assert not torch.equal(last, x_last)
    assert (last - x_last).abs().max() <= 0.05 + 1e-6


def test_transfer_mode_skips_the_teacher_inside_the_attack(tmp_path):
    """The attack sees 1 - max prob; the teacher runs exactly twice per
    batch (clean scoring, perturbed scoring)."""
    _, teacher, _ = _run(
        tmp_path, DepthConsistencyScore(), FGSM(eps=0.05), attack_scoring="transfer"
    )
    assert len(teacher.inputs) == 2 * 3
    clean_batches = list(_loader())
    last = teacher.inputs[-1]
    x_last = clean_batches[-1][0]
    assert not torch.equal(last, x_last)  # still scored on the attacked pixels


def test_query_attack_navigates_by_the_depth_score_in_white_box_mode(tmp_path):
    """QueryConfidence reads the wrapper's 4th element; in white-box mode
    that IS the depth residual, so the run is the adaptive gradient-free
    attack on the score -- it must simply complete with finite numbers."""
    attack = QueryConfidence(eps=0.05, n_queries=6, direction="under", seed=0)
    metrics, teacher, _ = _run(tmp_path, DepthConsistencyScore(), attack)
    assert "query_underconf_fd_auroc" in metrics
    assert len(teacher.inputs) > 2 * 3


def test_combined_score_runs_end_to_end(tmp_path):
    metrics, _, log_dir = _run(tmp_path, CombinedDepthScore(weight=0.5), FGSM(eps=0.05))
    assert "fgsm_fd_auroc" in metrics
    stored = torch.load(f"{log_dir}/storage_fgsm.pt")
    u = stored["uncertainties"]
    assert (u >= 0).all() and (u <= 1).all()


def test_base_wrapper_path_is_untouched_by_the_attacking_hook(tmp_path):
    """`ClassificationEvaluationModule` enters the wrapper's `attacking()`
    only if it has one; a BaseWrapper has none and everything is as before."""
    metrics, _, _ = _run(tmp_path, MultiClassMaxProbability(), FGSM(eps=0.05))
    assert "fgsm_fd_auroc" in metrics and "nat_fd_auroc" in metrics
