---
type: gotcha
title: ACE never reads the uncertainty score — a white-box ACE cell is the transfer cell relabelled
status: current
as_of: 2026-09-04
source: "Track C first run: depth_wb and depth_tr identical to four decimals under ace_uint8 on both depth arms; confirmed in src/trustfake/attacks/ace.py"
tags: [track-c, attacks, confidence-axis, depth]
links: [snapshots/2026-09-04-track-c-first-results, specs/track-c-depth-auxiliary]
---

# ACE never reads the uncertainty score

`trustfake.attacks.ACE` takes its one gradient from the predicted class's probability
(`probs.gather(1, preds)`) and halves ε against the model's logits. It never calls the wrapper's
uncertainty score, so `depth_attack_scoring=white_box` changes nothing for it: the Track C
`ace_uint8__depth_wb` cells reproduced `ace_uint8__depth_tr` to four decimals on both depth
arms (Φ 0.6070 / 0.5792, aurc 0.1226 / 0.2319). A reader who takes the `_wb` column as "the
adaptive number" reads a transfer attack as an adaptive one.

**The number this protects:** the one family where the depth residual beats max-probability
(Φ 0.61 vs 0.16 under ACE on `standard_depth`) is a family where the attack never targeted the
residual. It says the residual is insensitive to an MSP-only perturbation at ε=0.005, not that it
survives an attack on itself.

**Fix:** `jobs/track_c_depth.sh` schedules `_wb` cells for the `query_*` attacks only (they read
the score through `attacking()`); ACE gets `_tr`. An ACE that ascends the residual would be a
new attack class, the same follow-up as the gradient adaptive attack in the spec.
