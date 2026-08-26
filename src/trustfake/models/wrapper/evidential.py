import torch
from torch import Tensor
from torch.nn.functional import relu, softplus

from trustfake.models.wrapper.abc import TrustFakeWrapper

__all__ = ["EvidentialWrapper", "EVIDENCE_ACTIVATIONS"]

# Non-negative maps from raw network output to evidence e >= 0.
EVIDENCE_ACTIVATIONS = {
    "softplus": softplus,
    "exp": lambda z: torch.exp(z.clamp(max=20.0)),  # clamp guards overflow
    "relu": relu,
}


class EvidentialWrapper(TrustFakeWrapper):
    """Evidential deep-learning head (arXiv:2607.03075, Sec. 3.2).

    The backbone output is turned into a non-negative evidence vector
    ``e = activation(logits)``; the Dirichlet concentration is
    ``alpha = e + 1`` with strength ``S = sum_c alpha_c``. The returned
    probabilities are the Dirichlet posterior mean ``pi_bar = alpha / S``,
    equal to ``softmax(log alpha)`` (Eq. 5), and the prediction is its
    argmax. The uncertainty score is computed by the configured
    ``uncertainty_score`` on ``pi_bar`` -- with EvidentialPredictiveEntropy
    that is the paper's selective score ``u(x) = H[Cat(pi_bar)]`` (Eq. 1).

    The raw backbone logits are returned unchanged so attacks and losses can
    act on them (an evidence-targeted attack works in log-alpha space, which
    it derives from these). Temperature scaling, if set, divides the
    log-concentration (the pseudo-logit eta = log alpha) before the softmax,
    a valid post-hoc calibration of pi_bar that leaves the argmax untouched.

    Args:
        evidence_activation: one of EVIDENCE_ACTIVATIONS ("softplus" default).
    """

    def __init__(self, *args, evidence_activation: str = "softplus", **kwargs):
        super().__init__(*args, **kwargs)
        if evidence_activation not in EVIDENCE_ACTIVATIONS:
            raise ValueError(
                f"Unknown evidence_activation '{evidence_activation}'. "
                f"Available: {list(EVIDENCE_ACTIVATIONS)}"
            )
        self.evidence_activation = evidence_activation

    def evidence(self, logits: Tensor) -> Tensor:
        """Non-negative evidence e = activation(logits)."""
        return EVIDENCE_ACTIVATIONS[self.evidence_activation](logits)

    def dirichlet(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Return (alpha, eta) from raw logits: alpha = e + 1, eta = log alpha."""
        alpha = self.evidence(logits) + 1.0
        return alpha, torch.log(alpha)

    def loss_input(self, logits: Tensor) -> Tensor:
        """`EvidentialLoss` consumes alpha, not logits."""
        return self.dirichlet(logits)[0]

    def _from_logits(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Logits -> (logits, posterior mean, prediction, entropy), in fp32.

        The evidential head is computed at full precision even under autocast,
        and deliberately so. The chain is evidence(logits) -> alpha -> log
        alpha -> softmax -> entropy, and it starts with an exponential: in
        bf16, with 8 bits of mantissa, a large logit overflows to inf, alpha
        becomes inf, and the softmax then yields nan. It surfaces as
        "uncertainty contains non-finite values" a long way from the cause.

        Autocast is left ON for the backbone, which is where the time
        actually goes -- this head is a handful of elementwise ops on a
        (B, C) tensor, so making it exact costs nothing measurable and buys
        back the arm. Not hypothetical: enabling bf16 broke evidential
        training on the clean (non-adversarial) arm while EV-AT itself kept
        running, so the failure was invisible on the flagship configuration
        and only appeared on the ablation baseline.
        """
        self.uncertainty_score = self.uncertainty_score.to(logits.device)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits32 = logits.float()
            _, eta = self.dirichlet(logits32)
            probs = torch.softmax(eta / self.temperature, dim=1)  # pi_bar
            preds = torch.argmax(probs, dim=1)
            uncertainty = self.uncertainty_score(probs)
        self.uncertainty_score.reset()
        return logits, probs, preds, uncertainty

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.normalization_layer(x)
        logits = self.model(x)
        return self._from_logits(logits)

    def outputs_from_logits(
        self, logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._from_logits(logits)
