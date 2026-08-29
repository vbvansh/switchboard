"""A single page showing where the money went.

Server-rendered HTML with inline styles and inline SVG. No JavaScript, no build
step, no external fonts or scripts. Three reasons, in order of how much they
matter:

* It has to work on a machine with no internet access, which is exactly the
  deployment `SWITCHBOARD_LOCAL_ONLY` exists to support.
* A dashboard that needs `npm install` is a dashboard that stops working in
  eight months.
* Anything loaded from a CDN is a third party who can see when your engineers
  look at their AI spend.

The numbers come from the same ledger the CLI reads, so the page and
`switchboard usage` can never disagree.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from switchboard.shadow import ShadowSummary

STYLE = """
:root {
  --bg: #ffffff; --fg: #16181d; --muted: #6b7280; --line: #e5e7eb;
  --card: #f9fafb; --good: #047857; --warn: #b45309; --accent: #1d4ed8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e7e9ee; --muted: #9aa3b2; --line: #262b36;
    --card: #171a21; --good: #34d399; --warn: #fbbf24; --accent: #60a5fa;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.5rem;
  background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .75rem; }
.sub { color: var(--muted); margin: 0 0 1.5rem; }
.tiles {
  display: grid; gap: .75rem;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
}
.tile {
  background: var(--card); border: 1px solid var(--line);
  border-radius: .5rem; padding: .9rem 1rem;
}
.tile .label {
  color: var(--muted); font-size: .78rem;
  text-transform: uppercase; letter-spacing: .04em;
}
.tile .value {
  font-size: 1.5rem; font-weight: 650; margin-top: .2rem;
  font-variant-numeric: tabular-nums;
}
.tile .note { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
table {
  width: 100%; border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); }
th {
  color: var(--muted); font-weight: 600; font-size: .8rem;
  text-transform: uppercase; letter-spacing: .03em;
}
td.num, th.num { text-align: right; }
.good { color: var(--good); }
.warn { color: var(--warn); }
.banner {
  border: 1px solid var(--line); border-left: 3px solid var(--warn);
  background: var(--card); border-radius: .4rem; padding: .7rem .9rem;
  color: var(--muted); font-size: .86rem; margin-bottom: 1.25rem;
}
.bar {
  height: .5rem; background: var(--line);
  border-radius: 99px; overflow: hidden;
}
.bar > span { display: block; height: 100%; background: var(--accent); }
footer {
  color: var(--muted); font-size: .8rem; margin-top: 2.5rem;
  border-top: 1px solid var(--line); padding-top: 1rem;
}
code { background: var(--card); padding: .1rem .3rem; border-radius: .25rem; }
"""


@dataclass
class DashboardData:
    """Everything the page needs, gathered before any rendering happens."""

    usage_rows: list
    model_rows: list
    shadow: ShadowSummary
    cache: dict
    routing: dict
    simulated: bool
    shadow_mode: bool
    #: (label, action, requests, cost) rows from the usage policy.
    policy_rows: list = field(default_factory=list)
    #: Which rules are doing the flagging: (rule name, times).
    policy_rules: list = field(default_factory=list)
    policy: dict = field(default_factory=dict)


def _e(value) -> str:
    return html.escape(str(value))


def _tile(label: str, value: str, note: str = "", css: str = "") -> str:
    note_html = f'<div class="note">{_e(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="label">{_e(label)}</div>'
        f'<div class="value {css}">{value}</div>{note_html}</div>'
    )


def render(data: DashboardData) -> str:
    total_spent = sum(r.spent_usd for r in data.usage_rows)
    total_baseline = sum(r.baseline_usd for r in data.usage_rows)
    total_requests = sum(r.requests for r in data.usage_rows)
    saved = total_baseline - total_spent
    saved_pct = (100.0 * saved / total_baseline) if total_baseline else 0.0

    banner = ""
    if data.simulated:
        banner = (
            '<div class="banner"><strong>Simulated pricing.</strong> These '
            "models run locally and cost nothing. Each is priced as the "
            "commercial model it stands in for, so budgets and savings are "
            "meaningful and comparable. No real money is involved.</div>"
        )

    tiles = [
        _tile("Requests", f"{total_requests:,}", "this month"),
        _tile("Spent", f"${total_spent:,.4f}", "simulated" if data.simulated else ""),
        _tile("Would have cost", f"${total_baseline:,.4f}", "always top tier"),
        _tile("Saved", f"{saved_pct:.0f}%", f"${saved:,.4f}", "good"),
        _tile(
            "Cache hit rate",
            f"{100 * data.cache.get('hit_rate', 0):.0f}%",
            f"{data.cache.get('hits', 0):,} hits",
        ),
    ]

    # --- Per user -----------------------------------------------------------
    user_rows = "".join(
        f"<tr><td>{_e(r.name)}</td>"
        f'<td class="num">{r.requests:,}</td>'
        f'<td class="num">${r.spent_usd:,.4f}</td>'
        f'<td class="num">${r.baseline_usd:,.4f}</td>'
        f'<td class="num good">{r.saved_pct:.0f}%</td>'
        f'<td class="num">${r.remaining_usd:,.2f}</td></tr>'
        for r in data.usage_rows
    ) or '<tr><td colspan="6">No usage recorded yet.</td></tr>'

    # --- Per model ----------------------------------------------------------
    busiest = max((count for _, count, _ in data.model_rows), default=0) or 1
    model_rows = "".join(
        f"<tr><td>{_e(model)}</td>"
        f'<td class="num">{count:,}</td>'
        f'<td class="num">${cost:,.4f}</td>'
        f'<td><div class="bar"><span style="width:{100 * count / busiest:.1f}%"></span>'
        "</div></td></tr>"
        for model, count, cost in data.model_rows
    ) or '<tr><td colspan="4">No requests recorded yet.</td></tr>'

    # --- Shadow mode --------------------------------------------------------
    shadow = data.shadow
    if shadow.requests:
        direction = "good" if shadow.projected_saving_usd >= 0 else "warn"
        shadow_html = f"""
        <div class="tiles">
          {_tile("Shadowed requests", f"{shadow.requests:,}")}
          {_tile("Served cost", f"${shadow.actual_cost_usd:,.4f}")}
          {_tile(
              "Routing would cost",
              f"${shadow.shadow_cost_usd:,.4f}",
              "estimated",
          )}
          {_tile(
              "Projected saving",
              f"{shadow.projected_saving_pct:.0f}%",
              "",
              direction,
          )}
          {_tile("Different choice", f"{shadow.changed_pct:.0f}%",
                 f"{shadow.downgraded:,} cheaper, {shadow.upgraded:,} dearer")}
        </div>
        <div class="banner">
          <strong>These are projections, not measurements.</strong> The shadow
          model was never called, so its cost is estimated from the tokens the
          real model produced &mdash; a chattier model would really have cost
          more. And nothing here says whether the answer would have been as
          good: no answer was produced to check.
        </div>"""
    elif data.shadow_mode:
        shadow_html = (
            '<p class="sub">Shadow mode is on, but no shadowed requests have '
            "been recorded yet. Send some traffic through.</p>"
        )
    else:
        shadow_html = (
            '<p class="sub">Shadow mode is off. Turn it on with '
            "<code>SWITCHBOARD_SHADOW_MODE=true</code> to measure what routing "
            "would do to your traffic, without letting it change anything.</p>"
        )


    # --- Usage policy -------------------------------------------------------
    mode = data.policy.get("mode", "off")
    if mode == "off":
        policy_html = (
            '<p class="sub">The usage policy is off. Turn it on with '
            "<code>SWITCHBOARD_GUARDRAILS_MODE=flag</code> to label requests "
            "that look personal rather than work &mdash; without refusing "
            "anything.</p>"
        )
    else:
        examined = sum(count for _, _, count, _ in data.policy_rows)
        flagged = sum(
            count
            for label, _, count, _ in data.policy_rows
            if label and label != "clean"
        )
        flagged_cost = sum(
            cost
            for label, _, _, cost in data.policy_rows
            if label and label != "clean"
        )
        share = (100.0 * flagged / examined) if examined else 0.0
        rule_rows = "".join(
            f"<tr><td>{_e(name)}</td><td class=\"num\">{count:,}</td></tr>"
            for name, count in data.policy_rules[:10]
        ) or '<tr><td colspan="2">Nothing flagged.</td></tr>'
        policy_html = f"""
        <div class="tiles">
          {_tile("Requests examined", f"{examined:,}")}
          {_tile("Flagged as personal", f"{flagged:,}", f"{share:.1f}% of traffic")}
          {_tile("Spend on flagged", f"${flagged_cost:,.4f}")}
          {_tile("Mode", _e(mode),
                 "blocking" if mode == "block" else "labels only, nothing refused")}
        </div>
        <table>
          <thead><tr><th>Rule that matched</th>
          <th class="num">Times</th></tr></thead>
          <tbody>{rule_rows}</tbody>
        </table>
        <div class="banner">
          <strong>This is a keyword match, and it is wrong sometimes.</strong>
          Treat these counts as a prompt to go and look, never as a finding
          about a person. If a rule above keeps catching your team&rsquo;s real
          work, delete it: point
          <code>SWITCHBOARD_GUARDRAILS_FILE</code> at your own rule file, which
          replaces the built-in set rather than adding to it.
        </div>"""

    routing_note = (
        f"routing over {', '.join(_e(m) for m in data.routing.get('models', []))}"
        if data.routing.get("enabled")
        else _e(data.routing.get("reason", "routing disabled"))
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Switchboard</title><style>{STYLE}</style></head>
<body><main>
  <h1>Switchboard</h1>
  <p class="sub">{routing_note}</p>
  {banner}

  <div class="tiles">{"".join(tiles)}</div>

  <h2>Per developer, this month</h2>
  <table>
    <thead><tr><th>User</th><th class="num">Requests</th>
    <th class="num">Spent</th><th class="num">Would have cost</th>
    <th class="num">Saved</th><th class="num">Budget left</th></tr></thead>
    <tbody>{user_rows}</tbody>
  </table>

  <h2>Where requests went</h2>
  <table>
    <thead><tr><th>Model</th><th class="num">Requests</th>
    <th class="num">Cost</th><th>Share</th></tr></thead>
    <tbody>{model_rows}</tbody>
  </table>

  <h2>Shadow mode</h2>
  {shadow_html}

  <h2>Usage policy</h2>
  {policy_html}

  <footer>
    Served from the same ledger as <code>switchboard usage</code>, so this page
    and the command line can never disagree. Raw counters at
    <code>/metrics</code>.
  </footer>
</main></body></html>"""
