"""Live results over HTTP, for reading a sweep from another machine.

Stdlib only. Rebuilds the page from the metrics CSVs on every GET, so it is
always current with no cron, no cache and no state -- the CSV logs on disk
are the single source of truth. Serve with:

    set -a; . ./.env; set +a
    python jobs/serve_results.py --port 8793 --strategy ACE

Presentation: one profile card per arm (bar charts, ranked by resilience)
because the two findings that matter are SHAPES, not digits -- gradient
masking is a tall PGD bar next to an empty Square bar, and confidence
inversion is an fd-AUROC bar that fails to reach the 0.5 line. The full
numeric table follows for detail.

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

AXIS = {
    "standard": "control",
    "pgd_at": "label axis",
    "trades": "label axis",
    "at_kl": "label axis (hybrid)",
    "mart": "label axis (hybrid)",
    "at_conf": "confidence axis",
    "conf_reg": "confidence axis",
    "evidential_adversarial": "evidential",
}

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
    "suspect; read AutoAttack / Square. The card shape: a tall PGD bar over "
    "an empty Square bar.",
    "The evidence-targeted adversary does not remove the masking "
    "(ev_at_b0: PGD 0.48, AutoAttack 0.014).",
    "The over-confidence attack can DELETE the evidential selective signal, "
    "not just invert it: on ev_at_b0 the uncertainty score collapses to one "
    "distinct value (n_op = 1), leaving a single operating point -- accept "
    "everything. Shown as a hatched bar.",
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
    "Missing bars/cells: evaluations that predate the fp32 eval fix are "
    "being re-run and fill in as the queue drains.",
)


def _cell_class(key: str | None, value: float) -> str:
    if key is None or (key and key.endswith("fd_auroc")):
        if value < 0.5:
            return " class='bad'"
        if value >= 0.7:
            return " class='good'"
    if key and "residual_risk" in key:
        if value > 0.15:
            return " class='bad'"
        if value <= 0.06:
            return " class='good'"
    return ""


def _fmt(value, metrics, key=None) -> str:
    if value is None:
        return "<td class='na'>&ndash;</td>"
    if isinstance(value, float) and math.isnan(value):
        n_op = metrics.get("overconf_n_operating_points")
        tag = " (n_op=1)" if n_op == 1 else ""
        return f"<td class='bad'>deleted{tag}</td>"
    if isinstance(value, float):
        return f"<td{_cell_class(key, value)}>{value:.3f}</td>"
    return f"<td>{html.escape(str(value))}</td>"


def _bar(label: str, value, kind: str) -> str:
    """One horizontal bar. kind: 'acc' (neutral blue), 'fd' (0.5 reference
    line, colour by threshold), 'rr' (colour by SLA distance, bar spans
    0..0.4 so the interesting range is visible)."""
    if value is None:
        return (
            f"<div class='bar'><span class='blab'>{label}</span>"
            "<div class='track'><span class='pend'>pending</span></div>"
            "<span class='bval'></span></div>"
        )
    if isinstance(value, float) and math.isnan(value):
        return (
            f"<div class='bar'><span class='blab'>{label}</span>"
            "<div class='track'><div class='fill del' style='width:100%'>"
            "</div></div><span class='bval bad-t'>del.</span></div>"
        )
    pct = max(0.0, min(1.0, float(value))) * 100
    cls = "acc"
    if kind == "fd":
        cls = "bad" if value < 0.5 else ("good" if value >= 0.7 else "mid")
    elif kind == "rr":
        cls = "bad" if value > 0.15 else ("good" if value <= 0.06 else "mid")
        pct = max(0.0, min(1.0, float(value) / 0.4)) * 100
    track = "track fdline" if kind == "fd" else "track"
    return (
        f"<div class='bar'><span class='blab'>{label}</span>"
        f"<div class='{track}'><div class='fill {cls}' "
        f"style='width:{pct:.1f}%'></div></div>"
        f"<span class='bval'>{value:.3f}</span></div>"
    )


def _card(name: str, pipe: str, metrics, resilience) -> str:
    axis = AXIS.get(pipe, "evidential")
    if not metrics:
        return (
            f"<div class='card pending'><div class='chead'>{name}"
            f"<span class='tag'>{axis}</span></div>"
            "<p class='muted'>pending</p></div>"
        )
    g = metrics.get
    res = "&ndash;" if resilience is None else f"{resilience:.3f}"
    return f"""<div class='card'>
<div class='chead'>{name}<span class='tag'>{axis}</span>
<span class='res'>resilience {res}</span></div>
<div class='grp'><div class='glab'>accuracy — clean and prediction
attacks</div>
{_bar("clean", g("nat_accuracy_top1"), "acc")}
{_bar("PGD", g("pgd_accuracy_top1"), "acc")}
{_bar("AutoAtk", g("autoattack_accuracy_top1"), "acc")}
{_bar("Square", g("square_accuracy_top1"), "acc")}</div>
<div class='grp'><div class='glab'>failure-detection AUROC — line marks 0.5;
below it abstention selects errors</div>
{_bar("clean", g("nat_fd_auroc"), "fd")}
{_bar("@ACE", g("ace_uint8_fd_auroc"), "fd")}
{_bar("@ovconf", g("overconf_fd_auroc"), "fd")}</div>
<div class='grp'><div class='glab'>residual risk at the frozen gate
(SLA 0.05; bar spans 0&ndash;0.4)</div>
{_bar("clean", g("nat_moderation_residual_risk"), "rr")}
{_bar("@ACE", g("ace_uint8_moderation_residual_risk"), "rr")}</div>
</div>"""


def build_page(strategy: str) -> str:
    grid = build_grid(strategy, 12)
    entries = []
    for cfg in grid:
        metrics = _latest_test_metrics(cfg["name"])
        good = bool(metrics) and "nat_accuracy_top1" in metrics
        res = confidence_resilience(metrics) if good else None
        entries.append((cfg, metrics if good else {}, res))
    # Cards ranked by resilience; unfinished and unrankable arms sink.
    entries.sort(key=lambda e: (e[2] is None, -(e[2] or 0.0)))

    cards = "".join(_card(c["name"], c["pipe"], m, r) for c, m, r in entries)

    rows = []
    for cfg, metrics, _ in entries:
        if not metrics:
            rows.append(
                f"<tr><td class='arm'>{cfg['name']}</td>"
                f"<td class='na' colspan='{len(COLUMNS)}'>pending</td></tr>"
            )
            continue
        cells = []
        for _, key in COLUMNS:
            if key is None:
                value = confidence_resilience(metrics)
            else:
                value = metrics.get(key)
                if key == "nat_n_operating_points" and value is not None:
                    value = int(value)
            cells.append(_fmt(value, metrics, key))
        rows.append(
            f"<tr><td class='arm'>{cfg['name']}</td>" + "".join(cells) + "</tr>"
        )

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_joined = "".join(rows)
    head = "".join(f"<th>{name}</th>" for name, _ in COLUMNS)
    findings = "".join(f"<li>{html.escape(f)}</li>" for f in FINDINGS)
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="120">
<title>TrustFake — sweep results</title>
<style>
 :root {{ --bg:#fdfdfa; --fg:#1c1e24; --mut:#5c6270; --line:#d9dbe0;
   --mono:#111; --badbg:#fbe9e7; --bad:#9c2a1d; --goodbg:#e6f3e8;
   --good:#1d6b34; --midbg:#fdf2dc; --mid:#8a5b12; --card:#f5f5f1;
   --accent:#3b6ea5; --track:#e6e6e1; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --bg:#14151a; --fg:#e6e8ee; --mut:#9aa0ae; --line:#2c2e36;
     --mono:#f0f2f8; --badbg:#3a1f1b; --bad:#ff9c8a; --goodbg:#1b3323;
     --good:#7ed99a; --midbg:#39301a; --mid:#e8b95c; --card:#1d1f26;
     --accent:#7aa7d8; --track:#2a2c33; }} }}
 * {{ box-sizing: border-box; }}
 body {{ font: 17px/1.6 system-ui, sans-serif; margin: 0 auto; padding: 1rem;
   max-width: 78rem; background: var(--bg); color: var(--fg); }}
 h1 {{ font-size: 1.35rem; margin: .4rem 0; }}
 h2 {{ font-size: 1.1rem; margin-top: 1.8rem; }}
 .muted {{ color: var(--mut); font-size: .92em; }}
 .cards {{ display: grid; gap: 1rem;
   grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); }}
 .card {{ background: var(--card); border: 1px solid var(--line);
   border-radius: 12px; padding: .9rem 1rem; }}
 .card.pending {{ opacity: .55; }}
 .chead {{ font-family: ui-monospace, Menlo, monospace; font-weight: 600;
   color: var(--mono); margin-bottom: .5rem; }}
 .tag {{ font: 12px/1 system-ui; color: var(--mut); border: 1px solid
   var(--line); border-radius: 999px; padding: .15rem .55rem;
   margin-left: .5rem; vertical-align: 2px; }}
 .res {{ float: right; font: 600 13px/1.6 system-ui; color: var(--mut); }}
 .grp {{ margin: .7rem 0 0; }}
 .glab {{ font-size: 12.5px; color: var(--mut); margin-bottom: .25rem; }}
 .bar {{ display: flex; align-items: center; gap: .5rem; margin: .22rem 0; }}
 .blab {{ flex: 0 0 4.2rem; font-size: 13px; color: var(--mut);
   text-align: right; }}
 .track {{ flex: 1; height: 14px; background: var(--track);
   border-radius: 7px; overflow: hidden; position: relative; }}
 .track.fdline::after {{ content: ''; position: absolute; left: 50%; top: 0;
   bottom: 0; width: 2px; background: var(--mut); opacity: .8; }}
 .fill {{ height: 100%; border-radius: 7px; }}
 .fill.acc {{ background: var(--accent); }}
 .fill.good {{ background: var(--good); }}
 .fill.mid {{ background: var(--mid); }}
 .fill.bad {{ background: var(--bad); }}
 .fill.del {{ background: repeating-linear-gradient(45deg, var(--badbg),
   var(--badbg) 6px, var(--bad) 6px, var(--bad) 8px); opacity: .75; }}
 .bval {{ flex: 0 0 3.2rem; font-size: 13px; font-variant-numeric:
   tabular-nums; }}
 .bad-t {{ color: var(--bad); font-weight: 600; }}
 .pend {{ font-size: 12px; color: var(--mut); padding-left: .5rem; }}
 .wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch;
   border: 1px solid var(--line); border-radius: 10px; margin-top: .6rem; }}
 table {{ border-collapse: collapse; min-width: 900px; width: 100%;
   font-variant-numeric: tabular-nums; font-size: 16px; }}
 th, td {{ padding: .5rem .7rem; text-align: right; white-space: nowrap;
   border-bottom: 1px solid var(--line); }}
 tr:last-child td {{ border-bottom: none; }}
 tr:nth-child(even) td {{ background: var(--card); }}
 th {{ color: var(--mut); font-weight: 600; }}
 td.arm, th.arm {{ text-align: left; font-family: ui-monospace, Menlo,
   monospace; color: var(--mono); position: sticky; left: 0;
   background: var(--bg); font-weight: 600; }}
 tr:nth-child(even) td.arm {{ background: var(--card); }}
 td.na {{ color: var(--mut); opacity: .6; }}
 td.bad {{ background: var(--badbg); color: var(--bad); font-weight: 600; }}
 td.good {{ background: var(--goodbg); color: var(--good); }}
 li {{ margin: .55rem 0; }}
</style></head><body>
<h1>TrustFake — 8/255 sweep, live results</h1>
<p class="muted">Generated {stamp}, rebuilt from the metrics CSVs on every
request · auto-refreshes every 2 min · n = {SUBSAMPLE_ROWS} test-prefix rows ·
one seed. Cards are ranked by confidence resilience. Bar colour: green =
holding, amber = degraded, red = failing; hatched = the attack deleted the
uncertainty score outright. The masking signature is a tall PGD bar over an
empty Square bar.</p>
<div class="cards">{cards}</div>
<h2>Full table</h2>
<div class="wrap"><table>
<tr><th class="arm">arm</th>{head}</tr>{rows_joined}</table></div>
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
