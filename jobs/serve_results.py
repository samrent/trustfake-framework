"""Live results table over HTTP, for reading a sweep from another machine.

Stdlib only. Rebuilds the page from the metrics CSVs on every GET, so it is
always current with no cron, no cache and no state -- the CSV logs on disk
are the single source of truth. Serve with:

    set -a; . ./.env; set +a
    python jobs/serve_results.py --port 8793 --strategy ACE

Bind is 0.0.0.0 on the assumption of a private (tailnet) host; there is no
auth, so do not expose it beyond one.
"""

from __future__ import annotations

import argparse
import datetime
import html
import math
import os
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sweep import (  # noqa: E402
    SUBSAMPLE_ROWS,
    _latest_test_metrics,
    build_grid,
    confidence_resilience,
)

COLUMNS = (
    ("clean", "nat_accuracy_top1"),
    ("PGD", "pgd_accuracy_top1"),
    ("AutoAttack", "autoattack_accuracy_top1"),
    ("Square", "square_accuracy_top1"),
    ("fd-AUROC", "nat_fd_auroc"),
    ("@ACE", "ace_uint8_fd_auroc"),
    ("@overconf", "overconf_fd_auroc"),
    ("resilience", None),
    ("rr clean", "nat_moderation_residual_risk"),
    ("rr @ACE", "ace_uint8_moderation_residual_risk"),
    ("n_op", "nat_n_operating_points"),
)

FINDINGS = (
    "Confidence attacks leave accuracy bit-identical while failure-detection "
    "AUROC inverts (undefended: 0.843 clean, 0.134 under ACE). Below 0.5, "
    "abstention actively selects errors.",
    "Gradient masking in the evidential head: PGD reports 0.54 robust "
    "accuracy on ev_only while gradient-free Square reports 0.004. Any "
    "robustness number for an evidential model from a PGD-family attack is "
    "suspect; read AutoAttack / Square.",
    "The evidence-targeted adversary does not remove the masking "
    "(ev_at_b0: PGD 0.48, AutoAttack 0.014).",
    "The over-confidence attack can DELETE the evidential selective signal, "
    "not just invert it: on ev_at_b0 the uncertainty score collapses to one "
    "distinct value (n_op = 1), leaving a single operating point -- accept "
    "everything. Shown as 'deleted' below.",
    "Classical label-axis training substantially repairs the confidence "
    "axis (resilience 0.35 -> 0.72-0.76), and holds the frozen-gate "
    "residual risk under ACE at clean levels (0.04-0.05 vs 0.23 "
    "undefended). The direct penalty arm (conf_reg) does nothing under "
    "attack; the inner adversary, not the penalty, does the work.",
)

CAVEATS = (
    f"n = {SUBSAMPLE_ROWS} test-prefix rows: about +/-1.5 points of binomial "
    "standard error on any rate. One seed per arm: gaps of a few hundredths "
    "are not adjudicable. These numbers rank; they do not report.",
    "Non-adaptive: each defended arm is evaluated against the attack family "
    "it trained on. The adaptive re-tuned attacker is open work.",
    "Raw-split protocol: 'width == height -> fake' scores 0.985 on this "
    "split, so the clean column is not a forensic-quality claim. "
    "Within-model comparisons (clean vs attacked, arm vs arm) are "
    "unaffected.",
    "Missing cells: evaluations that predate the fp32 eval fix are being "
    "re-run; blank AutoAttack/Square cells fill in as the queue drains.",
)


def _fmt(value, metrics) -> str:
    if value is None:
        return "<td class='na'>&ndash;</td>"
    if isinstance(value, float) and math.isnan(value):
        # NaN here has one observed meaning: the uncertainty score went
        # constant under attack and the AUROC is undefined.
        n_op = metrics.get("overconf_n_operating_points")
        tag = " (n_op=1)" if n_op == 1 else ""
        return f"<td class='bad'>deleted{tag}</td>"
    if isinstance(value, float):
        cls = ""
        return f"<td{cls}>{value:.3f}</td>"
    return f"<td>{html.escape(str(value))}</td>"


def build_page(strategy: str) -> str:
    rows = []
    for cfg in build_grid(strategy, 12):
        metrics = _latest_test_metrics(cfg["name"])
        if not metrics or "nat_accuracy_top1" not in metrics:
            rows.append(
                f"<tr><td class='arm'>{cfg['name']}</td>"
                + f"<td class='na' colspan='{len(COLUMNS)}'>pending</td></tr>"
            )
            continue
        cells = []
        for _, key in COLUMNS:
            if key is None:
                # None when either attack's AUROC is missing or undefined --
                # renders as a dash; the deletion shows in its own column.
                value = confidence_resilience(metrics)
            else:
                value = metrics.get(key)
                if key == "nat_n_operating_points" and value is not None:
                    value = int(value)
            cells.append(_fmt(value, metrics))
        rows.append(
            f"<tr><td class='arm'>{cfg['name']}</td>" + "".join(cells) + "</tr>"
        )

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_joined = "".join(rows)
    head = "".join(f"<th>{name}</th>" for name, _ in COLUMNS)
    findings = "".join(f"<li>{html.escape(f)}</li>" for f in FINDINGS)
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="120">
<title>TrustFake — sweep results</title>
<style>
 body {{ font: 15px/1.6 system-ui, sans-serif; margin: 2rem auto; max-width: 74rem;
        padding: 0 1rem; background: #14151a; color: #d8dae0; }}
 h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1.05rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; width: 100%;
        font-variant-numeric: tabular-nums; }}
 th, td {{ padding: .35rem .6rem; text-align: right;
        border-bottom: 1px solid #2a2c33; }}
 th {{ color: #9aa0ae; font-weight: 500; }}
 td.arm {{ text-align: left; font-family: ui-monospace, monospace; color: #e8eaf0; }}
 td.na {{ color: #565a66; }} td.bad {{ color: #e07a6a; }}
 li {{ margin: .45rem 0; }} .muted {{ color: #9aa0ae; }}
 code {{ background: #1e2027; padding: .1rem .3rem; border-radius: 4px; }}
</style></head><body>
<h1>TrustFake — 8/255 sweep, live results</h1>
<p class="muted">Generated {stamp} on each request from the metrics CSVs ·
auto-refreshes every 2 min · n = {SUBSAMPLE_ROWS} test-prefix rows · one seed ·
accuracy columns under prediction attacks; fd-AUROC columns are
failure-detection AUROC (&lt; 0.5 = abstention selects errors); rr = residual
risk at thresholds frozen on clean calib.</p>
<table><tr><th style="text-align:left">arm</th>{head}</tr>{rows_joined}</table>
<h2>Findings so far</h2><ul>{findings}</ul>
<h2>Read with</h2><ul>{caveats}</ul>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    strategy = "ACE"

    def do_GET(self):  # noqa: N802
        try:
            body = build_page(self.strategy).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        except Exception as error:  # surface, never 500-blank
            body = f"<pre>{html.escape(repr(error))}</pre>".encode()
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8793)
    ap.add_argument("--strategy", default="ACE")
    args = ap.parse_args()
    if "OUTPUT_PATH" not in os.environ:
        raise SystemExit("OUTPUT_PATH not set; source the repo's .env first")
    Handler.strategy = args.strategy
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving on 0.0.0.0:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
