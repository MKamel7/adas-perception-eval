"""Render the evaluation as one self-contained HTML file.

No server, no build step, no external stylesheet, no CDN. The artifact is
committed and opens from disk, which matters because a report that requires
infrastructure to read is a report nobody reads. Everything is inlined.

WHAT THE LAYOUT IS ARGUING. The aggregate is shown once, small, at the top, and
then the page is given over to the spread. That is deliberate: the claim this
project exists to make is that the single number hides the failures that matter,
and a report that leads with a big confident headline would be making the
opposite case in its typography while making this one in its text.

Cells computed from too few objects are marked rather than hidden. Dropping them
would be quietly choosing which evidence to show; showing them unmarked would
invite a comparison between a number from four hundred objects and one from
three.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

from ape.cache import Header
from ape.classes import EVALUATED, HEADLINE
from ape.evaluate import Evaluation
from ape.slices import DIMENSIONS

STYLE = """
:root { --ink:#16181d; --dim:#666e7a; --line:#e2e5ea; --bg:#fff;
        --warn:#8a5a00; --bad:#a4243b; --good:#1c6b4a; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8eaee; --dim:#98a1ad; --line:#2a2f38; --bg:#14161a;
          --warn:#e0a63c; --bad:#ef7a8c; --good:#5fd0a0; } }
* { box-sizing:border-box; }
body { margin:0; padding:2.5rem 1.25rem 5rem; background:var(--bg); color:var(--ink);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:70rem; margin:0 auto; }
h1 { font-size:1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1.15rem; margin:2.75rem 0 .4rem; letter-spacing:-.01em; }
p.sub { color:var(--dim); margin:0 0 2rem; }
p.q { color:var(--dim); margin:.1rem 0 .8rem; font-style:italic; }
.meta { color:var(--dim); font-size:.85rem; border-top:1px solid var(--line);
        padding-top:1rem; margin-top:3rem; }
.headline { display:flex; gap:2.5rem; flex-wrap:wrap; align-items:baseline;
            border:1px solid var(--line); border-radius:.5rem; padding:1rem 1.25rem; }
.headline b { font-size:1.5rem; font-weight:600; }
.headline span { color:var(--dim); font-size:.85rem; display:block; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; min-width:34rem; }
th,td { text-align:right; padding:.4rem .6rem; border-bottom:1px solid var(--line);
        font-variant-numeric:tabular-nums; }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--dim); font-weight:500; font-size:.8rem; }
td.thin { color:var(--dim); }
td.thin::after { content:" ~"; }
.bar { display:inline-block; height:.5rem; border-radius:.25rem;
       background:var(--good); vertical-align:middle; margin-right:.4rem; }
.note { border-left:3px solid var(--line); padding:.1rem 0 .1rem 1rem;
        color:var(--dim); margin:1rem 0; }
.bad { color:var(--bad); } .warn { color:var(--warn); }
"""


def _cell(ap: float, positives: int, trustworthy: bool) -> str:
    if not positives:
        return '<td class="thin">-</td>'
    if ap != ap:
        return '<td class="thin">n/a</td>'
    width = max(1, round(ap * 60))
    bar = f'<span class="bar" style="width:{width}px"></span>'
    klass = ' class="thin"' if not trustworthy else ""
    return f"<td{klass}>{bar}{ap:.3f}</td>"


def render(result: Evaluation, header: Header) -> str:
    e = html.escape
    rows: list[str] = []

    rows.append("<h1>ADAS perception evaluation</h1>")
    rows.append(
        f'<p class="sub">{e(header.model)} on {result.frames} KITTI frames, '
        f'{result.objects} annotated objects, IoU 0.5. '
        f'Pretrained COCO weights; nothing was trained.</p>')

    spreads = []
    for label in HEADLINE:
        best, worst, _, where_worst = result.spread(label)
        if best == best and worst > 0:
            spreads.append((label, best / worst, where_worst, worst))

    rows.append('<div class="headline">')
    rows.append(f'<div><b>{result.headline:.3f}</b>'
                f'<span>mAP over {", ".join(HEADLINE)}</span></div>')
    for label, ratio, where, worst in spreads:
        rows.append(f'<div><b>{ratio:.1f}x</b><span>{e(label)} spread, '
                    f'worst {worst:.3f} at {e(where)}</span></div>')
    rows.append('</div>')

    rows.append('<div class="note">The aggregate above is the least '
                'informative number on this page. Everything below is the same '
                'detections cut along attributes the benchmark annotated before '
                'anyone saw a result.</div>')

    # Overall and difficulty
    rows.append("<h2>Overall, and by the benchmark's own difficulty</h2>")
    rows.append('<div class="scroll"><table><thead><tr><th>subset</th>'
                + "".join(f"<th>{e(c)}</th>" for c in EVALUATED)
                + "<th>objects</th></tr></thead><tbody>")
    total_positives = sum(result.overall[c].positives for c in EVALUATED)
    rows.append("<tr><td>all</td>" + "".join(
        _cell(result.overall[c].average_precision, result.overall[c].positives, True)
        for c in EVALUATED) + f"<td>{total_positives}</td></tr>")
    for tier, classes in result.by_difficulty.items():
        objects = sum(classes[c].positives for c in EVALUATED)
        rows.append(f"<tr><td>{e(tier)}</td>" + "".join(
            _cell(classes[c].average_precision, classes[c].positives, True)
            for c in EVALUATED) + f"<td>{objects}</td></tr>")
    rows.append("</tbody></table></div>")

    # Slices
    for dimension in DIMENSIONS:
        rows.append(f"<h2>By {e(dimension.name)}</h2>")
        rows.append(f'<p class="q">{e(dimension.question)}</p>')
        rows.append('<div class="scroll"><table><thead><tr><th>'
                    + e(dimension.name) + "</th>"
                    + "".join(f"<th>{e(c)}</th>" for c in EVALUATED)
                    + "<th>objects</th></tr></thead><tbody>")
        for item in (s for s in result.slices if s.dimension == dimension.name):
            objects = sum(c.curve.positives for c in item.cells)
            rows.append(f"<tr><td>{e(item.bin)}</td>" + "".join(
                _cell(c.curve.average_precision, c.curve.positives, c.trustworthy)
                for c in item.cells) + f"<td>{objects}</td></tr>")
        rows.append("</tbody></table></div>")

    rows.append('<div class="note">A tilde marks a cell computed from fewer '
                'than ten objects. It is shown rather than hidden, and should '
                'not be compared with a cell computed from hundreds.</div>')

    rows.append('<h2>What this does not claim</h2><div class="note">'
                'Cyclist is reported but excluded from the headline: KITTI '
                'annotates a rider and bicycle as one box and a COCO detector '
                'emits two, so that number measures box convention as much as '
                'detection. The IoU threshold is 0.5 for every class, where '
                "KITTI's own benchmark uses 0.7 for Car. Average precision "
                'follows COCO 101-point interpolation, not KITTI 40-point, so '
                'these figures are not directly comparable to the KITTI '
                'leaderboard. SOTIF vocabulary is borrowed; its process is not '
                'performed and no compliance is claimed.</div>')

    rows.append(f'<p class="meta">Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC '
                f'from {e((header.frames and str(header.frames)) or "?")} frames, '
                f'{e(header.first_frame)} to {e(header.last_frame)}, '
                f'detections kept above score {header.score_threshold}. '
                f'The mAP implementation is checked against pycocotools to '
                f'within 0.001 by the test suite.</p>')

    body = "\n".join(rows)
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>ADAS perception evaluation</title><style>{STYLE}</style>"
            f"</head><body><main>{body}</main></body></html>")
