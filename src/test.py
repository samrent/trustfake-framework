import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import lightning
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from trustfake.depth import (
    DEFAULT_TEACHER_INPUT_SIZE,
    DEPTH_ANYTHING_V2_SMALL,
    DEPTH_ANYTHING_V2_SMALL_REVISION,
)
from trustfake.instantiator import (
    config_parsing,
    save_experiment_config,
    select_evaluation_condition,
)
from trustfake.logging import add_handler, get_logger
from trustfake.metrics.calibration import calibrate_temperature
from trustfake.metrics.moderation import (
    ModerationPolicy,
    fit_thresholds,
    fit_uncertainty_gate,
)
from trustfake.metrics.uncertainty import CombinedDepthScore
from trustfake.models.torch import BinaryFoldClassifier
from trustfake.models.wrapper import (
    BaseWrapper,
    DepthConsistencyWrapper,
    EvidentialWrapper,
    MCDropoutWrapper,
)
from trustfake.pipes import ClassificationEvaluationModule
from trustfake.pydantic.model_output_schema import ClassificationModelOutput
from trustfake.utils import resolve_device

logger = get_logger("eval")

WRAPPERS = {
    "base": BaseWrapper,
    "mc_dropout": MCDropoutWrapper,
    "evidential": EvidentialWrapper,
    "depth": DepthConsistencyWrapper,
}


def _depth_wrapper_kwargs(cfg) -> dict:
    """Teacher settings for `wrapper=depth`, read with inert defaults. The
    input size must be the one the training targets were precomputed with,
    or the score compares two instruments."""
    return {
        "teacher_name": cfg.get("depth_teacher", DEPTH_ANYTHING_V2_SMALL),
        "teacher_revision": cfg.get(
            "depth_teacher_revision", DEPTH_ANYTHING_V2_SMALL_REVISION
        ),
        "teacher_input_size": cfg.get(
            "depth_teacher_input_size", DEFAULT_TEACHER_INPUT_SIZE
        ),
        "teacher_grad": cfg.get("depth_teacher_grad", True),
        "attack_scoring": cfg.get("depth_attack_scoring", "white_box"),
    }


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks with ties given their mean rank (Spearman's convention), so a
    constant vector ranks constant and correlates with nothing."""
    order = x.argsort()
    sorted_x = x[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_x, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(0) - counts
    mean_rank = starts.double() + (counts.double() - 1) / 2
    ranks = torch.empty(x.numel(), dtype=torch.float64)
    ranks[order] = mean_rank[inverse]
    return ranks


def spearman_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """|Spearman rho| between two 1-D score vectors; NaN when either is
    constant (undefined, never 0.0)."""
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    if a.numel() < 2 or a.numel() != b.numel():
        return float("nan")
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).item()
    if denom == 0.0:
        return float("nan")
    return abs(float((ra * rb).sum() / denom))


def calib_depth_pass(eval_module, calib_loader, device, trainer) -> dict:
    """The calib-split pass every depth-aware scoring needs (Track C).

    Two things happen here, both on the IN-DOMAIN calib split -- the same
    source temperature and the moderation gate come from -- and ordered after
    temperature scaling (the max-probability component depends on T) and
    before the moderation fit (the gate must be fitted on the final score):

    1. `CombinedDepthScore` gets its calib reference (`fit_reference`).
    2. Validity gate G2 (TODO section 4, the sigma seam): |Spearman rho|
       between the depth residual and 1 - max prob on calib. A residual
       that is 1 - MSP relabelled (|rho| >= 0.98) is not an independent
       uncertainty producer, however good its AUROC looks. Logged, and
       written to `depth_calib_gate.json` beside the metrics so a collator
       can read it without parsing logs.
    """
    score = eval_module.model.uncertainty_score
    msp_list, residual_list = [], []
    eval_module.model.to(device).eval()
    with torch.no_grad():
        for inputs, _ in calib_loader:
            msp, residual = eval_module.model.score_components(inputs.to(device))
            msp_list.append(msp.detach().cpu())
            residual_list.append(residual.detach().cpu())
    msp = torch.cat(msp_list)
    residual = torch.cat(residual_list)
    if isinstance(score, CombinedDepthScore):
        score.fit_reference(msp, residual)
        logger.info(
            f"CombinedDepthScore reference fitted on {msp.numel()} calib rows "
            f"(weight={score.weight})"
        )
    gate = {
        "n_calib": int(msp.numel()),
        "spearman_abs_residual_vs_msp": spearman_abs(residual, msp),
        "residual_mean": float(residual.mean()) if residual.numel() else float("nan"),
        "residual_std": float(residual.std()) if residual.numel() > 1 else float("nan"),
        "residual_finite": bool(torch.isfinite(residual).all()),
        "degeneracy_threshold": 0.98,
    }
    logger.info(
        "Depth score on calib -- GATE G2 |rho(residual, 1-MSP)| = "
        f"{gate['spearman_abs_residual_vs_msp']:.4f} (>= 0.98 means the "
        f"residual is MSP relabelled); residual mean {gate['residual_mean']:.4f}, "
        f"std {gate['residual_std']:.4f}, n = {gate['n_calib']}"
    )
    import json

    for trainer_logger in trainer.loggers:
        log_dir = Path(trainer_logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "depth_calib_gate.json").write_text(json.dumps(gate, indent=2))
    return gate


@hydra.main(
    config_path="../configs/training", config_name="eval_config", version_base=None
)
def run_eval_pipe(cfg: DictConfig):
    """
    Main function to run the evaluation pipelines using Hydra configuration.

    Args:
        config: Hydra configuration object containing all necessary parameters.
    """
    add_handler(
        sink=Path(HydraConfig.get().runtime.output_dir)
        / f"{HydraConfig.get().job.name}.log"
    )

    raw_cfg = cfg
    exp_cfg, cfg = config_parsing(cfg)

    logger.info(f"Output directories set to: {cfg['output_dir']}")
    seed = exp_cfg["seed"]
    if seed is None:
        msg = "Seed is not set in the configuration"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.debug(f"Reset random state with seed {seed}")
    lightning.seed_everything(seed)

    data_path = Path(os.environ.get("DATA_PATH"))
    if not data_path.exists():
        msg = (
            f"Data path {data_path} does not exist. Please check the DATA_PATH "
            "environment variable."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    datamodule = cfg["datamodule"]

    trainer: lightning.Trainer = cfg["trainer"]
    save_dir = Path(trainer.default_root_dir)

    try:
        # Look for .ckpt files
        all_ckpts = list(save_dir.rglob("*.ckpt"))
        all_ckpts.sort()
    except Exception as e:
        logger.exception(f"Error accessing checkpoints in {save_dir}: {e}")
        raise RuntimeError from e

    if not all_ckpts:
        msg = f"No checkpoints found in {save_dir}. Please check the training process."
        logger.error(msg)
        raise FileNotFoundError(msg)

    best_model_path = next(ckpt for ckpt in all_ckpts if ckpt.name != "last.ckpt")

    logger.info(f"Best model checkpoint found at: {best_model_path}")

    wrapper_name = cfg.get("wrapper", "base")
    wrapper_cls = WRAPPERS.get(wrapper_name)
    if wrapper_cls is None:
        msg = f"Unknown wrapper '{wrapper_name}'. Available wrappers: {list(WRAPPERS)}"
        logger.error(msg)
        raise ValueError(msg)

    # The single evaluation condition reported beside clean: an attack
    # (+attack=...) or a corruption (+corruption=...), never both.
    condition = select_evaluation_condition(cfg)
    if condition is not None:
        logger.info(f"Evaluation condition: {condition.name} ({condition.family})")

    wrapper_kwargs = {}
    if wrapper_cls is MCDropoutWrapper:
        wrapper_kwargs["num_samples"] = cfg.get("num_samples", 20)
    if wrapper_cls is EvidentialWrapper:
        wrapper_kwargs["evidence_activation"] = cfg.get(
            "evidence_activation", "softplus"
        )
    if wrapper_cls is DepthConsistencyWrapper:
        wrapper_kwargs = _depth_wrapper_kwargs(cfg)

    # PyTorch model
    module_best = wrapper_cls(
        normalization_layer=datamodule.normalization_layer,
        model=cfg["model"],
        loss_fn=cfg["loss"],
        uncertainty_score=cfg["uncertainty_score"],
        **wrapper_kwargs,
    )
    # `real_class` is needed here as well as by the moderation policy below.
    # The detection AUROC folds every non-real class to "fake", so leaving it
    # at the constructor default would have `detection_auroc` measuring class 0
    # while every moderation indicator in the SAME table measured the
    # configured class -- a disagreement inside one report, and one that stays
    # invisible for as long as the configured value happens to be 0.
    real_class = cfg.get("real_class", 0)
    try:
        eval_module = ClassificationEvaluationModule.load_from_checkpoint(
            checkpoint_path=best_model_path,
            model=module_best,
            model_output_schema_cls=ClassificationModelOutput,
            num_classes=datamodule.num_classes,
            attack=condition,
            real_class=real_class,
        )
    except Exception as e:
        logger.exception(f"Error loading model from checkpoint {best_model_path}: {e}")
        raise RuntimeError from e

    # Fold a multi-class model onto a binary benchmark (FakeClue). Applied
    # AFTER the checkpoint loads, never before: wrapping the module first
    # would prefix every state_dict key with `inner.` and the load would fail
    # to match. p_fake = 1 - P(real), the repo-wide definition.
    if cfg.get("binary_fold", False):
        if getattr(module_best, "consumes_depth", False):
            # The fold replaces `wrapper.model` and hides `forward_with_depth`;
            # the depth score would have nothing to compute against.
            msg = (
                "binary_fold=true with a depth-aware uncertainty score: the fold "
                "hides the depth path. Score the folded model with "
                "uncertainty_score=multiclass_max_probability instead."
            )
            logger.error(msg)
            raise ValueError(msg)
        if datamodule.num_classes != 2:
            msg = (
                f"binary_fold=true but the datamodule reports "
                f"{datamodule.num_classes} classes; the fold produces exactly 2"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            "binary_fold: collapsing the model onto 2 classes "
            f"(real_class={real_class}). This MERGES synthetic and tampered -- "
            "the per-modality breakout is not available under this fold."
        )
        eval_module.model.model = BinaryFoldClassifier(
            eval_module.model.model, real_class=real_class
        )

    # Where calibration is fitted. Normally the evaluation datamodule's own
    # calib split -- shard-disjoint from test, so it cannot touch the reported
    # rows. But when the evaluation set is SHIFTED (So-Fake-OOD), fitting there
    # would hide the exchangeability violation the condition exists to expose,
    # so `calib_datamodule=sid_set` supplies in-domain thresholds instead, as
    # a deployment would have.
    calib_source = cfg.get("calib_datamodule") or datamodule
    calib_source_name = type(calib_source).__name__
    if calib_source is not datamodule:
        logger.info(
            f"Calibration source: {calib_source_name} (SEPARATE from the "
            f"evaluation datamodule {type(datamodule).__name__}). Report which "
            "calib split the thresholds came from."
        )
        calib_source.setup()

    # Temperature scaling: fit on the calib split and freeze before scoring.
    if cfg.get("calibrate", True):
        datamodule.setup()
        calib_loader = getattr(calib_source, "calib_dataloader", None)
        if calib_loader is None:
            logger.warning(
                "Datamodule has no calib_dataloader; skipping temperature "
                "scaling (temperature stays 1.0)."
            )
        else:
            # A depth-aware wrapper scores by probability here: temperature
            # needs logits only, and the combined score has no calib
            # reference yet (it is fitted right after this, and would raise).
            probability_only = getattr(
                eval_module.model, "probability_only", nullcontext
            )
            with probability_only():
                temperature = calibrate_temperature(
                    eval_module.model, calib_loader(), device=resolve_device()
                )
            eval_module.model.temperature = temperature
            logger.info(f"Applied fitted temperature T = {temperature:.4f}")
    else:
        logger.info("Calibration disabled (calibrate=false); temperature = 1.0")

    # Track C: every depth-aware scoring needs a calib pass -- the combined
    # score for its reference, all of them for the degeneracy gate -- from
    # the same in-domain source, after T and before the moderation gate.
    if getattr(eval_module.model, "consumes_depth", False):
        calib_loader = getattr(calib_source, "calib_dataloader", None)
        if calib_loader is None:
            msg = (
                "a depth-aware uncertainty score needs an in-domain calib split "
                "(the combined score fits its reference there, and every depth "
                "score is gated there); select calib_datamodule=sid_set."
            )
            logger.error(msg)
            raise ValueError(msg)
        datamodule.setup()
        calib_depth_pass(eval_module, calib_loader(), resolve_device(), trainer)

    # WP4 selective moderation: fit the policy on the clean calib split (after
    # temperature) and freeze it. calib is shard-disjoint from test.
    if cfg.get("moderate", True):
        calib_loader = getattr(calib_source, "calib_dataloader", None)
        if calib_loader is None:
            logger.warning("No calib_dataloader; skipping moderation fitting.")
        else:
            import numpy as np

            datamodule.setup()
            device = resolve_device()
            probs_list, targets_list, unc_list = [], [], []
            eval_module.model.to(device).eval()
            with torch.no_grad():
                for inputs, targets in calib_loader():
                    _, probs, _, uncertainty = eval_module.model(inputs.to(device))
                    probs_list.append(probs.detach().cpu().numpy())
                    targets_list.append(targets.detach().cpu().numpy())
                    unc_list.append(uncertainty.detach().cpu().numpy())
            probs = np.concatenate(probs_list)
            targets = np.concatenate(targets_list)
            uncertainty = np.concatenate(unc_list)
            p_fake = 1.0 - probs[:, real_class]
            y_binary = (targets != real_class).astype(int)
            # sla_missed_fake is the second, asymmetric SLA: residual risk
            # counts both error directions, so a policy can meet it while
            # auto-allowing a large share of the fakes. Left null when the
            # deployment has no missed-fake commitment to state.
            sla_missed_fake = cfg.get("moderation_sla_missed_fake", None)
            t_low, t_high = fit_thresholds(
                p_fake,
                y_binary,
                sla_residual_risk=cfg.get("moderation_sla", 0.05),
                sla_missed_fake=sla_missed_fake,
            )
            t_unc = fit_uncertainty_gate(
                uncertainty,
                clean_review_budget=cfg.get("moderation_review_budget", 0.10),
            )
            eval_module.moderation_policy = ModerationPolicy(
                t_low=t_low, t_high=t_high, real_class=real_class, t_unc=t_unc
            )
            missed_fake_note = (
                "not set" if sla_missed_fake is None else f"<= {sla_missed_fake:.0%}"
            )
            logger.info(
                f"Moderation policy fitted on clean calib: t_low={t_low:.4f}, "
                f"t_high={t_high:.4f}, t_unc={t_unc:.4f}, "
                f"SLA residual risk <= {cfg.get('moderation_sla', 0.05):.0%}, "
                f"SLA missed fake {missed_fake_note}"
            )
            if eval_module.moderation_policy.infeasible:
                logger.warning(
                    "Moderation SLA infeasible: policy degrades to review-everything."
                )

    logger.info("Starting testing...")
    trainer.test(eval_module, datamodule=datamodule)
    logger.info("Testing completed.")

    save_experiment_config(trainer, raw_cfg)


if __name__ == "__main__":
    run_eval_pipe()
