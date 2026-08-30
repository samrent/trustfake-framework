"""CLIP-family zero-shot classifier -- seam 1 of the confidence-axis port.

The confidence-axis apparatus in this repo (the attack battery, the
failure-detection metrics, the selective layer, the ranking protocol) is
written against `TrustFakeWrapper`'s contract -- `x -> (logits, probs, preds,
uncertainty)` -- and against nothing else. It does not know what task it is
scoring. So porting the whole audit onto a CLIP-family vision encoder does
not need a new harness: it needs a module that turns an encoder into that
contract, which is what this file is.

The construction is the standard zero-shot head. Class names are rendered
through prompt templates, encoded by the text tower, averaged per class and
L2-normalized into one **prototype** per class. At inference the text tower is
gone: the image embedding is L2-normalized and dotted against the frozen
prototypes, so `logits = logit_scale * cos(f(x), P)`. That makes the head a
fixed linear map on the unit sphere -- a *disclosed* confound rather than a
hidden one, which is the argument for starting here rather than at the
representation level.

Two things about this file are load-bearing and easy to get wrong:

**Normalization lives here, not in the datamodule.** Everywhere else in the
framework the datamodule owns normalization and the wrapper applies it before
the model. A CLIP encoder is not free to be normalized any way you like: it
was trained under its own mean/std, and feeding it ImageNet statistics
silently shifts every embedding, which moves accuracy AND the confidence
distribution without raising anything. Preprocessing is part of *this*
encoder, so it is applied inside this module, from buffers that travel with
the checkpoint. The L_inf ball is unaffected -- attacks perturb the wrapper's
input, which is still raw [0, 1] pixels, exactly as `TrustFakeWrapper`
requires. **Configure the datamodule with `normalization_layer: null` when
using this model**, or the input is normalized twice and every number is
quietly wrong.

**`logit_scale` is a confidence-axis variable, not a detail.** CLIP ships a
learned scale of ~100, which saturates the softmax: max-probability sits at
~1.0 for nearly every input, so `1 - MSP` collapses toward zero and the
failure-detection AUROC is then computed over a degenerate signal. That is
measuring the temperature, not the model. Fit the temperature on the calib
split (`trustfake.metrics.calibration.fit_temperature`, whose (1e-2, 1e2)
bounds already cover this scale) and hand it to the wrapper before reporting
any Phi. The default here is CLIP's native scale so that clean accuracy
reproduces the published zero-shot numbers; the calibrated scale is a
reported quantity, not a default.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from trustfake.logging import get_logger

__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "OPENAI_CLIP_LOGIT_SCALE",
    "CLIPZeroShotClassifier",
    "build_text_prototypes",
    "clip_zeroshot",
]

logger = get_logger("clip")

#: OpenAI CLIP preprocessing statistics, shared by the open_clip ports of the
#: OpenAI checkpoints and by the LAION-trained ViT-B/32 weights. A SigLIP or
#: InternViT lineage uses different values -- pass them explicitly.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

#: exp(learned logit_scale) for the released CLIP checkpoints.
OPENAI_CLIP_LOGIT_SCALE: float = 100.0


class CLIPZeroShotClassifier(nn.Module):
    """A frozen vision encoder plus frozen text prototypes, as an `x -> logits`
    module that drops into `BaseWrapper` unchanged.

    Args:
        visual: The image tower. Any module mapping ``(B, 3, H, W)`` to
            ``(B, D)`` embeddings -- `open_clip`'s `model.visual`, or a stub
            in tests. Not required to return unit-norm embeddings; this module
            normalizes.
        prototypes: ``(C, D)`` class prototypes from the text tower. Stored
            L2-normalized as a buffer, so a checkpoint round-trips without
            needing the text tower to exist.
        logit_scale: Multiplies the cosine similarities. See the module
            docstring -- this sets the confidence distribution.
        normalize_input: Apply the encoder's own mean/std inside the forward.
            Leave True unless the caller has already normalized, which for
            this framework would mean a double-normalized input.
        mean, std: Preprocessing statistics for `visual`.
        freeze: Set `requires_grad_(False)` on the encoder. Seam 1 audits a
            pretrained encoder rather than training one, and input gradients
            (what FGSM/PGD need) are unaffected by frozen parameters.
    """

    def __init__(
        self,
        visual: nn.Module,
        prototypes: Tensor,
        logit_scale: float = OPENAI_CLIP_LOGIT_SCALE,
        normalize_input: bool = True,
        mean: Sequence[float] = CLIP_MEAN,
        std: Sequence[float] = CLIP_STD,
        freeze: bool = True,
    ):
        super().__init__()
        if prototypes.ndim != 2:
            raise ValueError(
                f"prototypes must be (C, D), got shape {tuple(prototypes.shape)}"
            )
        if logit_scale <= 0:
            raise ValueError(f"logit_scale must be > 0, got {logit_scale}")

        self.visual = visual
        self.logit_scale = float(logit_scale)
        self.normalize_input = normalize_input

        # Prototypes are stored already-normalized so the forward is one
        # matmul, and as a buffer so `state_dict` carries them: the text tower
        # is a build-time dependency, never a load-time one.
        self.register_buffer(
            "prototypes", F.normalize(prototypes.detach().float(), dim=-1)
        )
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std).view(1, -1, 1, 1))

        if freeze:
            self.visual.requires_grad_(False)

    @property
    def num_classes(self) -> int:
        return int(self.prototypes.shape[0])

    def encode(self, x: Tensor) -> Tensor:
        """Unit-norm image embeddings for raw ``[0, 1]`` input.

        Exposed separately because seam 2 -- the representation-level port,
        where confidence is read off the feature geometry instead of a head --
        needs exactly this and none of the rest.
        """
        if self.normalize_input:
            x = (x - self.pixel_mean) / self.pixel_std
        return F.normalize(self.visual(x).float(), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.logit_scale * self.encode(x) @ self.prototypes.T


def build_text_prototypes(
    model: object,
    tokenizer: object,
    classnames: Sequence[str],
    templates: Sequence[str] = ("a photo of a {}.",),
    device: str | torch.device = "cpu",
) -> Tensor:
    """Encode ``classnames`` through ``templates`` into ``(C, D)`` prototypes.

    Prompt ensembling, as in the CLIP paper: each class is rendered through
    every template, the embeddings are L2-normalized, averaged, and normalized
    again. The averaging is over *normalized* embeddings, so no single verbose
    template dominates by magnitude.

    `model` and `tokenizer` are the `open_clip` objects; they are parameters
    rather than a name so the caller controls the checkpoint and so tests can
    pass fakes.
    """
    prototypes = []
    with torch.no_grad():
        for name in classnames:
            prompts = [t.format(name) for t in templates]
            tokens = tokenizer(prompts).to(device)  # type: ignore[operator]
            embeddings = model.encode_text(tokens).float()  # type: ignore[attr-defined]
            embeddings = F.normalize(embeddings, dim=-1)
            prototypes.append(F.normalize(embeddings.mean(0), dim=-1))
    return torch.stack(prototypes)


def clip_zeroshot(
    classnames: Sequence[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    templates: Sequence[str] = ("a photo of a {}.",),
    logit_scale: float | None = None,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> CLIPZeroShotClassifier:
    """Build a zero-shot classifier from an `open_clip` checkpoint.

    `open_clip` is imported here rather than at module scope, the same way the
    AutoAttack wrappers import their package: it is a heavy optional
    dependency and importing this module must not require it -- the tests
    construct `CLIPZeroShotClassifier` directly with a stub encoder and never
    touch the network.

    Args:
        classnames: One name per class, in label order. For the forgery task
            that order is (real, synthetic, tampered); for an arbitrary
            transfer set it is whatever the datamodule's label mapping says.
        model_name, pretrained: `open_clip.create_model_and_transforms` args.
        templates: Prompt templates; `{}` is filled with the class name.
        logit_scale: Overrides the checkpoint's learned scale. `None` keeps
            the checkpoint's own value -- see the module docstring on why this
            is a reported quantity.
        device: Where prototypes are built. The returned module is on CPU
            unless the caller moves it; Lightning handles placement.
        freeze: Freeze the encoder parameters.
    """
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    prototypes = build_text_prototypes(
        model, tokenizer, classnames, templates, device=device
    )
    scale = (
        float(model.logit_scale.exp().item()) if logit_scale is None else logit_scale
    )
    logger.info(
        "clip_zeroshot: %s/%s, %d classes, %d templates, logit_scale=%.2f",
        model_name,
        pretrained,
        len(classnames),
        len(templates),
        scale,
    )
    if logit_scale is None and scale > 50:
        logger.warning(
            "logit_scale=%.1f is the checkpoint's native scale: max-probability "
            "will saturate and 1-MSP will be near-degenerate. Fit a temperature "
            "on the calib split before reporting failure detection.",
            scale,
        )

    return CLIPZeroShotClassifier(
        visual=model.visual,
        prototypes=prototypes.cpu(),
        logit_scale=scale,
        freeze=freeze,
    )
