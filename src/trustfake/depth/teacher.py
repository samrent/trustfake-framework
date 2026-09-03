"""The frozen monocular-depth teacher (Depth Anything V2, small) -- Track C.

The teacher is a measurement instrument, not a model under test. It runs
twice in this repo: OFFLINE, once, to precompute the depth targets the
auxiliary head trains against (`src/precompute_depth.py`), and ONLINE at
evaluation, inside `DepthConsistencyWrapper`, to give the depth-consistency
score its reference on the very input the classifier saw. Both paths go
through this one module so they agree on preprocessing, resolution and the
output frame; a residual between two maps produced by different pipelines
would measure the pipelines.

Three things are load-bearing and easy to get wrong:

**Preprocessing lives here, not in the datamodule** (the CLIP gotcha). The
teacher takes the same raw [0, 1] pixels the wrapper takes -- the space every
attack perturbs in -- and applies its own resize and ImageNet normalisation
internally, from buffers. The datamodule's `normalization_layer` never
touches it, so nothing is normalised twice and the online reference sees
exactly the attacked tensor.

**The resize rule is the checkpoint's own.** Depth Anything V2 is a ViT with
14-pixel patches trained at 518; its HF image processor scales the image as
little as possible toward 518 and rounds each side to a multiple of 14
(`dpt_resize_hw`). A 224x224 input becomes 518x518. Feeding 224 directly
works (16 x 14 = 224) but is a different instrument; the input size is a
recorded setting, written into the target store's manifest and refused on
mismatch at training time.

**Frozen means `requires_grad_(False)`, never `no_grad`.** Whether a call
builds a graph is the CALLER's decision: the precompute runs under
`inference_mode`, while the evaluation wrapper keeps the gradient through the
teacher so gradient attacks on the depth score see the whole score, not the
student half only (see `freezing-a-backbone-must-not-use-no-grad`).

`transformers` is imported lazily in `load_depth_teacher`, exactly as
`open_clip` is in `trustfake.models.torch.clip`; the test suite uses
`FakeDepthTeacher` and never needs it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from trustfake.logging import get_logger
from trustfake.losses.depth import normalize_depth, resize_depth

__all__ = [
    "DEPTH_ANYTHING_V2_SMALL",
    "DEPTH_ANYTHING_V2_SMALL_REVISION",
    "DEFAULT_DEPTH_SIZE",
    "DEFAULT_TEACHER_INPUT_SIZE",
    "PATCH_MULTIPLE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DepthTeacher",
    "FakeDepthTeacher",
    "dpt_resize_hw",
    "load_depth_teacher",
]

logger = get_logger("depth-teacher")

#: Hub id and pinned revision of the small V2 checkpoint (~99 MB safetensors).
DEPTH_ANYTHING_V2_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_ANYTHING_V2_SMALL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"

#: The checkpoint's own preprocessing: scale toward 518, sides a multiple of
#: 14 (the ViT-S/14 patch), ImageNet statistics.
DEFAULT_TEACHER_INPUT_SIZE = 518
PATCH_MULTIPLE = 14
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Grid the targets are stored at and the ResNet head emits at a 224 input.
DEFAULT_DEPTH_SIZE = 112

#: The only frame this repo stores or compares depth in (see losses.depth).
DEPTH_FRAME = "median_mad"


def _round_to_multiple(value: float, multiple: int, min_val: int) -> int:
    rounded = int(round(value / multiple) * multiple)
    return max(rounded, min_val)


def dpt_resize_hw(
    height: int,
    width: int,
    target: int = DEFAULT_TEACHER_INPUT_SIZE,
    multiple: int = PATCH_MULTIPLE,
    keep_aspect_ratio: bool = True,
) -> tuple[int, int]:
    """The DPT image processor's resize rule, in integers.

    With `keep_aspect_ratio` the side whose scale is closest to 1 sets the
    scale for both ("scale as little as possible"); each side is then rounded
    to the nearest multiple of `multiple`, never below one patch. 224x224
    -> 518x518; 480x640 -> 518x686.
    """
    if height <= 0 or width <= 0:
        msg = f"height and width must be positive, got {height}x{width}"
        raise ValueError(msg)
    scale_h = target / height
    scale_w = target / width
    if keep_aspect_ratio:
        if abs(1 - scale_w) < abs(1 - scale_h):
            scale_h = scale_w
        else:
            scale_w = scale_h
    return (
        _round_to_multiple(scale_h * height, multiple, multiple),
        _round_to_multiple(scale_w * width, multiple, multiple),
    )


class DepthTeacher(nn.Module):
    """A frozen depth backbone plus its own preprocessing and output frame.

    Args:
        backbone: Any module mapping normalised pixels (B, 3, H', W') to a
            depth map -- a tensor of shape (B, H', W') or (B, 1, H', W'), or
            an object carrying `.predicted_depth` (the HF output). The real
            one comes from `load_depth_teacher`; tests inject a stub.
        output_size: Grid of the returned maps.
        input_size: Target of the checkpoint's resize rule.
        multiple: Patch multiple of the checkpoint's resize rule.
        keep_aspect_ratio: See `dpt_resize_hw`.
        mean, std: Normalisation statistics applied INSIDE the module.
        eps: MAD floor of the output frame.
        autocast: Run the backbone under float16 autocast on CUDA. OFF by
            default: fp16 is fine for the no-grad precompute (which opts in),
            but a gradient taken THROUGH an fp16 teacher underflows -- the
            per-pixel gradient of the residual is ~1e-8, below fp16's
            subnormal floor -- so a white-box attack on the depth score
            would silently see only the student half while the config said
            "teacher included". The online teacher therefore runs fp32. The
            map is always returned as float32 in the median/MAD frame.
        name: Recorded in `describe()`, so a target store can say which
            teacher produced it.
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_size: int = DEFAULT_DEPTH_SIZE,
        input_size: int = DEFAULT_TEACHER_INPUT_SIZE,
        multiple: int = PATCH_MULTIPLE,
        keep_aspect_ratio: bool = True,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        eps: float = 1e-6,
        autocast: bool = False,
        name: str = "custom",
        revision: str | None = None,
    ):
        super().__init__()
        if output_size <= 0 or input_size <= 0 or multiple <= 0:
            msg = "output_size, input_size and multiple must be positive"
            raise ValueError(msg)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.output_size = int(output_size)
        self.input_size = int(input_size)
        self.multiple = int(multiple)
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.eps = float(eps)
        self.autocast = bool(autocast)
        self.name = name
        self.revision = revision
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std).view(1, 3, 1, 1))
        self.eval()

    def train(self, mode: bool = True) -> DepthTeacher:
        """A frozen instrument never leaves eval mode, whatever its owner
        does: no BatchNorm statistic and no dropout mask of the teacher may
        depend on the batch it is asked about."""
        return super().train(False)

    @property
    def frame(self) -> str:
        return DEPTH_FRAME

    def describe(self) -> dict[str, Any]:
        """The settings a target store records, and a training run checks."""
        return {
            "teacher": self.name,
            "revision": self.revision,
            "input_size": self.input_size,
            "multiple": self.multiple,
            "keep_aspect_ratio": self.keep_aspect_ratio,
            "output_size": self.output_size,
            "frame": self.frame,
            "precision": "fp16-autocast" if self.autocast else "fp32",
        }

    def preprocess(self, x: Tensor) -> Tensor:
        """Raw [0, 1] pixels -> the backbone's input (resized, normalised)."""
        if x.ndim != 4 or x.shape[1] != 3:
            msg = f"expected a (B, 3, H, W) batch, got shape {tuple(x.shape)}"
            raise ValueError(msg)
        size = dpt_resize_hw(
            x.shape[-2],
            x.shape[-1],
            self.input_size,
            self.multiple,
            self.keep_aspect_ratio,
        )
        if size != tuple(x.shape[-2:]):
            # Bicubic like the checkpoint's processor; antialias matters only
            # when shrinking, and is a no-op otherwise. No clamp afterwards,
            # matching the HF pipeline.
            x = F.interpolate(
                x.float(),
                size=size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (x.float() - self.pixel_mean) / self.pixel_std

    def raw_depth(self, x: Tensor) -> Tensor:
        """Teacher-resolution relative inverse depth, (B, 1, H', W') float32."""
        pixels = self.preprocess(x)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.autocast and pixels.is_cuda,
        ):
            out = self.backbone(pixels)
        depth = getattr(out, "predicted_depth", out)
        if not isinstance(depth, Tensor):
            msg = (
                "the depth backbone must return a tensor or an object with a "
                f"`predicted_depth` tensor, got {type(out).__name__}"
            )
            raise ValueError(msg)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4 or depth.shape[1] != 1:
            msg = f"the depth backbone returned shape {tuple(depth.shape)}"
            raise ValueError(msg)
        return depth.float()

    def forward(self, x: Tensor) -> Tensor:
        """Raw [0, 1] pixels -> (B, 1, output_size, output_size), float32,
        per-image zero-median / unit-MAD."""
        depth = resize_depth(self.raw_depth(x), self.output_size)
        return normalize_depth(depth, self.eps)


class _LuminanceBlur(nn.Module):
    """Stub backbone: blurred luminance, so the 'depth' is a deterministic,
    differentiable, parameter-free function of the pixels."""

    def __init__(self, blur: int = 3):
        super().__init__()
        self.blur = blur

    def forward(self, x: Tensor) -> Tensor:
        lum = x.mean(dim=1, keepdim=True)
        return F.avg_pool2d(lum, self.blur, stride=1, padding=self.blur // 2)


class FakeDepthTeacher(DepthTeacher):
    """A `DepthTeacher` over a stub backbone, for tests and for exercising the
    precompute script without `transformers`. Same preprocessing rule, same
    frame, same shapes as the real one; no weights, no network. Tests pass a
    small `input_size`/`multiple` so tiny images are not blown up to 518.
    """

    def __init__(self, output_size: int = DEFAULT_DEPTH_SIZE, blur: int = 3, **kw):
        kw.setdefault("name", "fake_luminance_blur")
        kw.setdefault("input_size", DEFAULT_TEACHER_INPUT_SIZE)
        super().__init__(_LuminanceBlur(blur), output_size=output_size, **kw)


def load_depth_teacher(
    hub_id: str = DEPTH_ANYTHING_V2_SMALL,
    revision: str | None = DEPTH_ANYTHING_V2_SMALL_REVISION,
    cache_dir: str | None = None,
    local_files_only: bool | None = None,
    device: str | torch.device = "cpu",
    **teacher_kwargs: Any,
) -> DepthTeacher:
    """Build the real teacher from the Hub (or the local HF cache).

    `transformers` is imported here and nowhere else. `local_files_only=None`
    leaves the decision to the standard `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`
    environment variables; `cache_dir=None` uses the standard HF cache.
    """
    try:
        from transformers import AutoModelForDepthEstimation
    except ImportError as e:  # pragma: no cover - environment-dependent
        msg = (
            "The depth teacher needs `transformers` (Depth Anything V2). It is "
            "not part of the locked environment; install it into the venv with "
            "`uv pip install transformers`."
        )
        raise ImportError(msg) from e

    kwargs: dict[str, Any] = {"revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    backbone = AutoModelForDepthEstimation.from_pretrained(hub_id, **kwargs)
    backbone.eval()
    teacher = DepthTeacher(backbone, name=hub_id, revision=revision, **teacher_kwargs)
    n_params = sum(p.numel() for p in backbone.parameters())
    logger.info(
        f"Depth teacher {hub_id}@{revision}: {n_params / 1e6:.1f}M params, "
        f"input {teacher.input_size} (x{teacher.multiple}), output "
        f"{teacher.output_size}, frame {teacher.frame}"
    )
    return teacher.to(device)
