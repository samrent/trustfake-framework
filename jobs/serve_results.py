"""Live results over HTTP, for reading a sweep from another machine.

Stdlib only. Rebuilds the page from the metrics CSVs on every GET, so it is
always current with no cron, no cache and no state -- the CSV logs on disk
are the single source of truth. Serve with:

    set -a; . ./.env; set +a
    python jobs/serve_results.py --port 8793 --strategy ACE

Presentation: three charts, because the findings are RELATIONSHIPS, not
magnitudes -- (1) the clean-accuracy / confidence-resilience frontier,
(2) the per-arm collapse of failure-detection AUROC under ACE, and (3) the
gradient-masking check, PGD against AutoAttack, where distance below the
diagonal is exactly the overstatement PGD reports. The numeric table follows
for detail.

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

#: Colour encodes the arm's DESIGN category, constant across all charts.
AXIS = {
    "standard": ("control", "#8a8f9c"),
    "pgd_at": ("label axis", "#3b6ea5"),
    "trades": ("label axis", "#3b6ea5"),
    "at_kl": ("label axis", "#3b6ea5"),
    "mart": ("label axis", "#3b6ea5"),
    "at_conf": ("confidence axis", "#c2571f"),
    "conf_reg": ("confidence axis", "#c2571f"),
    "evidential_adversarial": ("evidential", "#7b5ed1"),
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
    "abstention actively selects errors -- chart 2's grey line.",
    "Gradient masking in the evidential head: PGD reports 0.54 robust "
    "accuracy on ev_only while gradient-free Square reports 0.004. In chart "
    "3, distance below the diagonal IS the overstatement.",
    "The evidence-targeted adversary does not remove the masking "
    "(ev_at_b0: PGD 0.48, AutoAttack 0.014).",
    "The over-confidence attack can DELETE the evidential selective signal "
    "outright: on ev_at_b0 the uncertainty score collapses to one distinct "
    "value (n_op = 1) -- a single operating point, accept everything. Such "
    "arms drop off chart 2 and are footnoted.",
    "Classical label-axis training substantially repairs the confidence "
    "axis, and full EV-AT (beta = 1) joins it -- against ev_at_b0's deleted "
    "score, the evidence-alignment term REA is what does the work. Read "
    "with the masking caveat: the confidence attacks are gradient-based "
    "too, so evidential resilience needs a gradient-free confirmation.",
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
    "Missing points/cells: evaluations that predate the fp32 eval fix are "
    "being re-run and fill in as the queue drains.",
)


def _cell_class(key, value):
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


def _fmt(value, metrics, key=None):
    if value is None:
        return "<td class='na'>&ndash;</td>"
    if isinstance(value, float) and math.isnan(value):
        n_op = metrics.get("overconf_n_operating_points")
        tag = " (n_op=1)" if n_op == 1 else ""
        return f"<td class='bad'>deleted{tag}</td>"
    if isinstance(value, float):
        return f"<td{_cell_class(key, value)}>{value:.3f}</td>"
    return f"<td>{html.escape(str(value))}</td>"


def _short(name):
    return name.removeprefix("e8_")


def _spread(desired, gap=15.0, lo=None, hi=None):
    """1-D label layout: keep the given order, enforce a minimum gap."""
    out = []
    for y in desired:
        if out and y < out[-1] + gap:
            y = out[-1] + gap
        out.append(y)
    if hi is not None and out and out[-1] > hi:
        shift = out[-1] - hi
        out = [y - shift for y in out]
        for i in range(len(out) - 2, -1, -1):
            if out[i] > out[i + 1] - gap:
                out[i] = out[i + 1] - gap
    if lo is not None:
        out = [max(y, lo) for y in out]
    return out


W, H, ML, MR, MT, MB = 680, 340, 56, 160, 18, 40


def _frame(x0, x1, y0, y1, xticks, yticks, xlab, ylab):
    def sx(v):
        return ML + (v - x0) / (x1 - x0) * (W - ML - MR)

    def sy(v):
        return MT + (1 - (v - y0) / (y1 - y0)) * (H - MT - MB)

    parts = []
    for t in xticks:
        parts.append(
            f"<line x1='{sx(t):.1f}' y1='{MT}' x2='{sx(t):.1f}' "
            f"y2='{H - MB}' class='grid'/>"
            f"<text x='{sx(t):.1f}' y='{H - MB + 16}' class='tick' "
            f"text-anchor='middle'>{t:g}</text>"
        )
    for t in yticks:
        parts.append(
            f"<line x1='{ML}' y1='{sy(t):.1f}' x2='{W - MR}' "
            f"y2='{sy(t):.1f}' class='grid'/>"
            f"<text x='{ML - 8}' y='{sy(t):.1f}' class='tick' "
            f"text-anchor='end' dominant-baseline='central'>{t:g}</text>"
        )
    parts.append(
        f"<text x='{(ML + W - MR) / 2:.0f}' y='{H - 6}' class='axis' "
        f"text-anchor='middle'>{xlab}</text>"
        f"<text x='{ML}' y='{MT - 6}' class='axis'>{ylab}</text>"
    )
    return sx, sy, "".join(parts)


def _scatter(points, x0, x1, y0, y1, xticks, yticks, xlab, ylab, refline):
    """points: (x, y, name, colour). refline: None | ('h', v) | ('diag',)."""
    sx, sy, body = _frame(x0, x1, y0, y1, xticks, yticks, xlab, ylab)
    if refline:
        if refline[0] == "h":
            y = sy(refline[1])
            body += (
                f"<line x1='{ML}' y1='{y:.1f}' x2='{W - MR}' y2='{y:.1f}' class='ref'/>"
            )
        else:
            lo, hi = max(x0, y0), min(x1, y1)
            body += (
                f"<line x1='{sx(lo):.1f}' y1='{sy(lo):.1f}' "
                f"x2='{sx(hi):.1f}' y2='{sy(hi):.1f}' class='ref'/>"
                f"<text x='{sx(hi) - 4:.1f}' y='{sy(hi) + 14:.1f}' "
                "class='tick' text-anchor='end'>honest = on the line</text>"
            )
    pts = sorted(points, key=lambda p: sy(p[1]))
    label_y = _spread([sy(p[1]) for p in pts], 15, MT + 8, H - MB - 4)
    for (x, y, name, colour), ly in zip(pts, label_y, strict=True):
        px, py = sx(x), sy(y)
        body += f"<circle cx='{px:.1f}' cy='{py:.1f}' r='5' fill='{colour}'/>"
        lx = px + 9
        if abs(ly - py) > 8:
            body += (
                f"<line x1='{px + 5:.1f}' y1='{py:.1f}' x2='{lx:.1f}' "
                f"y2='{ly:.1f}' class='lead'/>"
            )
        body += (
            f"<text x='{lx + 2:.1f}' y='{ly:.1f}' class='plab' "
            f"fill='{colour}' dominant-baseline='central'>{name}</text>"
        )
    return body


def _svg(body):
    return (
        f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' "
        "preserveAspectRatio='xMidYMid meet'>" + body + "</svg>"
    )


def _chart_frontier(rows):
    pts = [
        (m["nat_accuracy_top1"], r, _short(n), c)
        for n, c, m, r in rows
        if r is not None and "nat_accuracy_top1" in m
    ]
    body = _scatter(
        pts,
        0.6,
        0.9,
        0.2,
        0.9,
        (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
        (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "clean accuracy",
        "confidence resilience",
        ("h", 0.5),
    )
    return _svg(body)


def _chart_collapse(rows):
    xc, xa = ML + 60, W - MR - 40
    sy = lambda v: MT + (1 - (v - 0.0) / 0.9) * (H - MT - MB)  # noqa: E731
    body = ""
    for t in (0.0, 0.25, 0.5, 0.75):
        cls = "ref" if t == 0.5 else "grid"
        body += (
            f"<line x1='{ML}' y1='{sy(t):.1f}' x2='{W - MR}' "
            f"y2='{sy(t):.1f}' class='{cls}'/>"
            f"<text x='{ML - 8}' y='{sy(t):.1f}' class='tick' "
            f"text-anchor='end' dominant-baseline='central'>{t:g}</text>"
        )
    body += (
        f"<text x='{xc}' y='{H - MB + 18}' class='axis' "
        "text-anchor='middle'>clean</text>"
        f"<text x='{xa}' y='{H - MB + 18}' class='axis' "
        "text-anchor='middle'>under ACE</text>"
        f"<text x='{ML}' y='{MT - 6}' class='axis'>failure-detection AUROC "
        "(0.5 line: below it, abstention selects errors)</text>"
    )
    slopes, deleted = [], []
    for n, c, m, _ in rows:
        a, b = m.get("nat_fd_auroc"), m.get("ace_uint8_fd_auroc")
        if a is None or b is None:
            continue
        if isinstance(b, float) and math.isnan(b):
            deleted.append(_short(n))
            continue
        slopes.append((a, b, _short(n), c))
    slopes.sort(key=lambda s: sy(s[1]))
    label_y = _spread([sy(s[1]) for s in slopes], 15, MT + 8, H - MB - 4)
    for (a, b, name, colour), ly in zip(slopes, label_y, strict=True):
        y1, y2 = sy(a), sy(b)
        body += (
            f"<line x1='{xc}' y1='{y1:.1f}' x2='{xa}' y2='{y2:.1f}' "
            f"stroke='{colour}' stroke-width='2.2' stroke-linecap='round' "
            f"opacity='.85'/>"
            f"<circle cx='{xc}' cy='{y1:.1f}' r='4' fill='{colour}'/>"
            f"<circle cx='{xa}' cy='{y2:.1f}' r='4' fill='{colour}'/>"
            f"<text x='{xa + 10}' y='{ly:.1f}' class='plab' fill='{colour}' "
            f"dominant-baseline='central'>{name} {b:.2f}</text>"
        )
    note = (
        f"score deleted by overconf (n_op=1): {', '.join(deleted)}" if deleted else ""
    )
    return _svg(body), note


def _chart_masking(rows):
    pts, missing = [], []
    for n, c, m, _ in rows:
        p, a = m.get("pgd_accuracy_top1"), m.get("autoattack_accuracy_top1")
        if p is None or a is None:
            if "nat_accuracy_top1" in m:
                missing.append(_short(n))
            continue
        pts.append((p, a, _short(n), c))
    body = _scatter(
        pts,
        0.0,
        0.8,
        0.0,
        0.8,
        (0.0, 0.2, 0.4, 0.6, 0.8),
        (0.0, 0.2, 0.4, 0.6, 0.8),
        "accuracy under PGD (gradient-based)",
        "accuracy under AutoAttack (ensemble)",
        ("diag",),
    )
    note = f"awaiting AutoAttack: {', '.join(missing)}" if missing else ""
    return _svg(body), note


def build_page(strategy):
    grid = build_grid(strategy, 12)
    rows = []
    for cfg in grid:
        metrics = _latest_test_metrics(cfg["name"])
        good = bool(metrics) and "nat_accuracy_top1" in metrics
        res = confidence_resilience(metrics) if good else None
        colour = AXIS.get(cfg["pipe"], ("evidential", "#7b5ed1"))[1]
        rows.append((cfg["name"], colour, metrics if good else {}, res))

    chart1 = _chart_frontier(rows)
    chart2, note2 = _chart_collapse(rows)
    chart3, note3 = _chart_masking(rows)

    table_rows = []
    ordered = sorted(rows, key=lambda e: (e[3] is None, -(e[3] or 0.0)))
    for name, _, metrics, _res in ordered:
        if not metrics:
            table_rows.append(
                f"<tr><td class='arm'>{name}</td>"
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
        table_rows.append(f"<tr><td class='arm'>{name}</td>" + "".join(cells) + "</tr>")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_joined = "".join(table_rows)
    head = "".join(f"<th>{name}</th>" for name, _ in COLUMNS)
    findings = "".join(f"<li>{html.escape(f)}</li>" for f in FINDINGS)
    caveats = "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
    legend = " ".join(
        f"<span style='color:{col}'>&#9679; {cat}</span>"
        for cat, col in (
            ("control", "#8a8f9c"),
            ("label axis", "#3b6ea5"),
            ("confidence axis", "#c2571f"),
            ("evidential", "#7b5ed1"),
        )
    )
    n2 = f"<p class='muted'>{html.escape(note2)}</p>" if note2 else ""
    n3 = f"<p class='muted'>{html.escape(note3)}</p>" if note3 else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="120">
<title>TrustFake — sweep results</title>
<style>
 :root {{ --bg:#fdfdfa; --fg:#1c1e24; --mut:#5c6270; --line:#d9dbe0;
   --mono:#111; --badbg:#fbe9e7; --bad:#9c2a1d; --goodbg:#e6f3e8;
   --good:#1d6b34; --card:#f5f5f1; --grid:#e7e7e2; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --bg:#14151a; --fg:#e6e8ee; --mut:#9aa0ae; --line:#2c2e36;
     --mono:#f0f2f8; --badbg:#3a1f1b; --bad:#ff9c8a; --goodbg:#1b3323;
     --good:#7ed99a; --card:#1d1f26; --grid:#23252c; }} }}
 * {{ box-sizing: border-box; }}
 body {{ font: 17px/1.6 system-ui, sans-serif; margin: 0 auto; padding: 1rem;
   max-width: 78rem; background: var(--bg); color: var(--fg); }}
 h1 {{ font-size: 1.35rem; margin: .4rem 0; }}
 h2 {{ font-size: 1.1rem; margin-top: 1.9rem; }}
 .muted {{ color: var(--mut); font-size: .92em; }}
 .chart {{ background: var(--card); border: 1px solid var(--line);
   border-radius: 12px; padding: .6rem .8rem; margin-top: .6rem; }}
 svg {{ display: block; }}
 .grid {{ stroke: var(--grid); stroke-width: 1; }}
 .ref {{ stroke: var(--mut); stroke-width: 1.4; stroke-dasharray: 5 4; }}
 .lead {{ stroke: var(--mut); stroke-width: .8; opacity: .6; }}
 .tick {{ font: 12px system-ui; fill: var(--mut); }}
 .axis {{ font: 13px system-ui; fill: var(--mut); }}
 .plab {{ font: 600 13px ui-monospace, monospace; }}
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
one seed · {legend}</p>
<h2>1 · The frontier — what confidence robustness costs in clean accuracy</h2>
<p class="muted">Up and right is better. Below the dashed line, abstention
selects errors.</p>
<div class="chart">{chart1}</div>
<h2>2 · The collapse — failure-detection AUROC, clean &rarr; under ACE</h2>
<p class="muted">Flat is robust; a steep drop is the confidence attack
working; crossing the dashed line is inversion.</p>
<div class="chart">{chart2}</div>{n2}
<h2>3 · The masking check — PGD vs AutoAttack</h2>
<p class="muted">On the diagonal, PGD tells the truth. Distance below it is
robustness PGD reports that a stronger attack refutes -- the obfuscated
gradients signature.</p>
<div class="chart">{chart3}</div>{n3}
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


def main():
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
