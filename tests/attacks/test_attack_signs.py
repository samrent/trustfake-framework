"""Per-attack sign contracts: what each attack SHOULD move, asserted.

A wrong attack implementation usually still runs, still perturbs, and still
passes shape checks -- what gives it away is a metric moving the wrong way.
A prediction-targeted attack must move accuracy down; a confidence-targeted
attack must leave accuracy untouched while failure detection and selective
risk degrade. These are checked on a small trained model with a real error
pattern, deterministically.
"""

import pytest
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAUROC

from trustfake.attacks import ACE, FGSM, UncertaintyFGSM
from trustfake.metrics.evaluation.selective_classification import aurc_from_scores
from trustfake.metrics.uncertainty.probs import MultiClassMaxProbability
from trustfake.models.wrapper import BaseWrapper

NUM_CLASSES = 4
INPUT_SHAPE = (3, 8, 8)
N_TRAIN, N_EVAL = 256, 512  # eval = train + fresh, so errors exist
EPS = 0.05


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]
        self.linear = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


@pytest.fixture(scope="module")
def fitted():
    """A linear model trained to memorize 256 random points, evaluated on
    those plus 256 fresh ones: high but imperfect accuracy, so both correct
    and wrong predictions exist and failure detection is measurable."""
    torch.manual_seed(0)
    x_train = torch.rand(N_TRAIN, *INPUT_SHAPE)
    y_train = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
    x_fresh = torch.rand(N_EVAL - N_TRAIN, *INPUT_SHAPE)
    y_fresh = torch.randint(0, NUM_CLASSES, (N_EVAL - N_TRAIN,))

    model = BaseWrapper(
        normalization_layer=_Identity(),
        model=_TinyClassifier(),
        loss_fn=nn.CrossEntropyLoss(),
        uncertainty_score=MultiClassMaxProbability(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        logits, _, _, _ = model(x_train)
        nn.functional.cross_entropy(logits, y_train).backward()
        opt.step()
    model.eval()

    x = torch.cat([x_train, x_fresh])
    y = torch.cat([y_train, y_fresh])
    with torch.no_grad():
        _, _, preds, uncertainty = model(x)
    acc = (preds == y).float().mean().item()
    # the fixture is only useful if errors AND correct predictions exist
    assert 0.3 < acc < 0.95, f"fixture drifted: clean accuracy {acc:.3f}"
    return model, x, y, preds, uncertainty


def _scores(model, inputs):
    with torch.no_grad():
        _, _, preds, uncertainty = model(inputs)
    return preds, uncertainty


def test_fgsm_moves_accuracy_down(fitted):
    """The prediction family's defining sign: accuracy drops."""
    model, x, y, preds_clean, _ = fitted
    perturbed = FGSM(eps=EPS)(model, x, y)
    preds_adv, _ = _scores(model, perturbed)
    acc_clean = (preds_clean == y).float().mean().item()
    acc_adv = (preds_adv == y).float().mean().item()
    assert acc_adv < acc_clean


def test_uncertainty_fgsm_moves_uncertainty_up(fitted):
    """UncertaintyFGSM's defining sign: uncertainty rises. It makes no
    label-preservation claim (maximizing uncertainty may flip predictions).

    Probed at a small eps: it is a single sign-step, and on a sharply
    fitted model a large step overshoots into ANOTHER class's
    high-confidence region, sending 1 - maxprob back down (measured on this
    fixture: 88% of rows up at eps=0.01, 17% at eps=0.2). The gradient
    direction is the claim under test; the step size is not."""
    model, x, y, _, unc_clean = fitted
    perturbed = UncertaintyFGSM(eps=0.01)(model, x, y)
    _, unc_adv = _scores(model, perturbed)
    assert unc_adv.mean().item() > unc_clean.mean().item()
    assert (unc_adv > unc_clean).float().mean().item() > 0.5


def test_ace_signs(fitted):
    """The confidence family's defining signs, all on one run: accuracy
    EXACTLY unchanged (accept test), confidence down where the model is
    right and up where it is wrong, therefore failure-detection AUROC down
    and AURC up. An accuracy-based regression test proves nothing here --
    that is the point of the attack."""
    model, x, y, preds_clean, unc_clean = fitted
    result = ACE(eps=EPS).run(model, x, y)
    preds_adv, unc_adv = _scores(model, result.perturbed)

    # accuracy: bit-identical predictions, not merely equal accuracy
    assert torch.equal(preds_adv, preds_clean)

    correct = (preds_clean == y).float()
    errors = 1.0 - correct
    # direction: uncertainty up where correct, down where wrong (on average)
    assert (
        unc_adv[correct.bool()].mean().item() > unc_clean[correct.bool()].mean().item()
    )
    assert unc_adv[errors.bool()].mean().item() < unc_clean[errors.bool()].mean().item()

    # failure detection: uncertainty must rank errors ABOVE correct rows;
    # after ACE it ranks them lower
    auroc = BinaryAUROC()
    fd_clean = auroc(unc_clean, errors.long()).item()
    auroc.reset()
    fd_adv = auroc(unc_adv, errors.long()).item()
    assert fd_adv < fd_clean

    # selective risk: abstention gets WORSE than not abstaining at all
    aurc_clean = aurc_from_scores(unc_clean.numpy(), errors.numpy())
    aurc_adv = aurc_from_scores(unc_adv.numpy(), errors.numpy())
    assert aurc_adv > aurc_clean
