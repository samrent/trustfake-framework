import torch

from trustfake.attacks.abc import (
    AdversarialAttack,
    AttackDirection,
    AttackFamily,
    AttackResult,
)
from trustfake.models.wrapper import TrustFakeWrapper

__all__ = ["ACE"]


class ACE(AdversarialAttack):
    """
    Attack on Confidence Estimation (ACE).

    A confidence-targeted attack: it never tries to change the prediction,
    only the confidence attached to it -- down where the model is right, up
    where it is wrong -- so accuracy is unchanged by construction while
    every confidence-ranked quantity (failure detection, selective risk)
    degrades. One gradient of the top-class probability gives the direction;
    each sample then takes the largest step within its budget that leaves
    the argmax unchanged, halving the per-sample epsilon on rejection.

    A step is accepted by re-checking the argmax, so label preservation is
    exact by construction -- as measured on the accept-check forward, whose
    logits are returned in `AttackResult.accepted_logits` and must be the
    reported ones (see `AttackResult`).

    `quantize` decides the threat model, and it is not a detail: at small
    eps the mean perturbation can be below one 8-bit quantisation step, so
    rounding back to the pixel grid (what saving a file does) erases it.
    Unquantised ACE describes an attacker with post-decode tensor access;
    `quantize=True` describes one who can only upload a file. Report which
    one a row is. With `quantize=True` the grid snap can exceed `eps` by up
    to half a grey level when `eps` is not grid-aligned.

    Args:
        eps (float): Maximum L_inf perturbation, per sample, in input units.
        clip_min (float): Minimum valid value for a perturbed input.
        clip_max (float): Maximum valid value for a perturbed input.
        iters (int): Epsilon-halving attempts per sample.
        quantize (bool): Snap candidate steps to the 1/255 pixel grid.
    """

    family = AttackFamily.CONFIDENCE
    direction = AttackDirection.BOTH
    # Ground truth chooses the direction per sample; without it ACE can
    # only push confidence down (see `run`).
    uses_labels = True

    def __init__(
        self,
        eps: float = 0.005,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        iters: int = 15,
        quantize: bool = False,
    ):
        super().__init__(eps=eps, clip_min=clip_min, clip_max=clip_max)
        self.iters = iters
        self.quantize = quantize

    @property
    def name(self) -> str:
        return "ace"

    def run(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AttackResult:
        was_training = model.training
        model.eval()

        with torch.enable_grad():
            x = inputs.clone().detach().requires_grad_(True)
            logits, probs, preds, _ = model(x)
            # Confidence of the predicted class; one gradient serves every
            # halving attempt.
            kappa = probs.gather(1, preds[:, None]).squeeze(1)
            grad = torch.autograd.grad(kappa.sum(), x)[0]
        eta = grad.sign()

        # Down-confidence where the model is right, up where it is wrong.
        # Without ground truth every sample counts as right (pure
        # down-confidence) -- ACE needs labels to use its up direction.
        expand = (-1,) + (1,) * (inputs.ndim - 1)
        if targets is None:
            direction = torch.full_like(kappa, -1.0).view(expand)
        else:
            direction = torch.where(preds == targets, -1.0, 1.0).view(expand)

        out = inputs.clone()
        out_logits = logits.detach().clone()
        eps_i = torch.full((inputs.shape[0],), float(self.eps), device=inputs.device)
        pending = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)

        for _ in range(self.iters):
            idx = pending.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            step = direction[idx] * eps_i[idx].view(expand) * eta[idx]
            cand = self._clamp(inputs[idx] + step)
            if self.quantize:
                # Snap to what a saved file can hold.
                cand = self._clamp(torch.round(cand * 255.0) / 255.0)
            with torch.no_grad():
                cand_logits, _, cand_preds, _ = model(cand)
            ok = cand_preds == preds[idx]
            out[idx[ok]] = cand[ok]
            out_logits[idx[ok]] = cand_logits[ok]
            pending[idx[ok]] = False
            eps_i[idx[~ok]] *= 0.5

        model.train(was_training)
        return AttackResult(
            perturbed=out.detach(),
            effective_eps=(out - inputs).abs().flatten(1).amax(dim=1).detach(),
            clean_preds=preds.detach(),
            accepted_logits=out_logits.detach(),
        )

    def __call__(
        self,
        model: TrustFakeWrapper,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.run(model, inputs, targets).perturbed
