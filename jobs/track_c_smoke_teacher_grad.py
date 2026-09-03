"""GPU smoke for the white-box depth score: does the gradient really flow
through the teacher on THIS device?

The CPU test suite proves it in fp32 only. On the box the teacher runs where
the config says (fp32 online, per the fp16-underflow finding); this script
checks, on one batch, that the input gradient of the depth score with
`teacher_grad=True` differs from the student-half gradient and that the
teacher half is non-zero on a non-trivial fraction of pixels. Random-init
student, real teacher, no data. Exit code 1 on failure.

Usage:  .venv/bin/python jobs/track_c_smoke_teacher_grad.py [--input-size 518]
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

from trustfake.depth import DEFAULT_TEACHER_INPUT_SIZE, load_depth_teacher
from trustfake.metrics.uncertainty import DepthConsistencyScore
from trustfake.models.torch import resnet18
from trustfake.models.wrapper import DepthConsistencyWrapper
from trustfake.utils import resolve_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-size", type=int, default=DEFAULT_TEACHER_INPUT_SIZE)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    device = resolve_device()
    teacher = load_depth_teacher(input_size=args.input_size, device=device)

    torch.manual_seed(0)
    model = resnet18(num_classes=3, depth_head=True).to(device).eval()
    grads = {}
    for teacher_grad in (True, False):
        wrapper = (
            DepthConsistencyWrapper(
                normalization_layer=nn.Identity(),
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                uncertainty_score=DepthConsistencyScore(),
                teacher=teacher,
                teacher_grad=teacher_grad,
            )
            .to(device)
            .eval()
        )
        torch.manual_seed(1)
        x = torch.rand(args.batch, 3, 224, 224, device=device, requires_grad=True)
        with torch.enable_grad():
            wrapper(x)[3].sum().backward()
        grads[teacher_grad] = x.grad.detach().float().cpu()

    diff = grads[True] - grads[False]
    nonzero = float((diff != 0).float().mean())
    finite = bool(torch.isfinite(grads[True]).all())
    print(
        f"device={device} teacher_input={args.input_size} "
        f"|grad_full|={grads[True].abs().mean():.3e} |grad_student|="
        f"{grads[False].abs().mean():.3e} teacher-half nonzero fraction={nonzero:.3f} "
        f"finite={finite}"
    )
    if not finite or nonzero < 0.5:
        print("FAIL: the teacher half of the white-box gradient is missing")
        return 1
    print("OK: white-box gradient reaches the input through the teacher")
    return 0


if __name__ == "__main__":
    sys.exit(main())
