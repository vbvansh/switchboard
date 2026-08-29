"""The public landing page, served at `/`.

WHY IT LIVES INSIDE THE APP. The alternative is a separate static site on
separate hosting with a separate deploy. That means two things to keep in sync,
two places for the numbers to drift apart, and a marketing page that can claim
things the software no longer does. Serving it from the same process means one
deploy, one URL, and figures that come from the same source as the docs.

SAME RULES AS THE DASHBOARD. Server-rendered HTML, inline CSS, one small inline
script for the calculator. No JavaScript framework, no build step, no fonts or
scripts from a CDN. It renders on a machine with no internet, and no third party
learns who visited.

ONE CONTENT RULE, and it is the reason the page reads differently from most
product sites: **every number here appears in docs/RESULTS.md with its method
next to it, and the limitations section is not optional.** A product whose whole
argument is "we report honestly" cannot have a landing page that oversells. The
honest section is a feature, not a disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The measured saving, from docs/RESULTS.md section 3. Defined once here so
#: the calculator and the prose can never disagree.
MEASURED_SAVING_PCT = 57.0
MEASURED_ACCURACY = 88.3
BASELINE_ACCURACY = 86.8

STYLE = """
:root {
  --bg: #ffffff; --fg: #14161a; --muted: #5b6472; --line: #e4e7ec;
  --card: #f8f9fb; --accent: #1d4ed8; --good: #047857; --warn: #b45309;
  --code: #0f1115; --code-fg: #e7e9ee;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d0f13; --fg: #e8eaf0; --muted: #9aa3b2; --line: #232833;
    --card: #14171d; --accent: #7aa2ff; --good: #34d399; --warn: #fbbf24;
    --code: #05070a; --code-fg: #e7e9ee;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 62rem; margin: 0 auto; padding: 0 1.25rem; }
a { color: var(--accent); }

nav {
  border-bottom: 1px solid var(--line); position: sticky; top: 0;
  background: var(--bg); z-index: 5;
}
nav .wrap { display: flex; align-items: center; gap: 1.5rem; height: 3.5rem; }
nav .brand { font-weight: 700; letter-spacing: -.01em; margin-right: auto; }
nav a { color: var(--muted); text-decoration: none; font-size: .92rem; }
nav a:hover { color: var(--fg); }

header { padding: 4.5rem 0 3rem; }
h1 {
  font-size: clamp(2.1rem, 5vw, 3.2rem); line-height: 1.1;
  letter-spacing: -.03em; margin: 0 0 1rem; max-width: 20ch;
}
.lede { font-size: 1.18rem; color: var(--muted); max-width: 62ch; margin: 0 0 2rem; }

.headline {
  display: grid; gap: 1rem; margin: 2.5rem 0;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
}
.stat {
  border: 1px solid var(--line); border-radius: .6rem;
  background: var(--card); padding: 1.1rem 1.2rem;
}
.stat .n {
  font-size: 1.9rem; font-weight: 680; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums;
}
.stat .k {
  font-size: .76rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin-bottom: .3rem;
}
.stat .s { font-size: .84rem; color: var(--muted); margin-top: .2rem; }

h2 {
  font-size: 1.6rem; letter-spacing: -.02em; margin: 3.5rem 0 .5rem;
  scroll-margin-top: 4.5rem;
}
h3 { font-size: 1.05rem; margin: 2rem 0 .4rem; }
p { max-width: 68ch; }
section { padding-bottom: .5rem; }
.muted { color: var(--muted); }

pre {
  background: var(--code); color: var(--code-fg); padding: 1rem 1.1rem;
  border-radius: .5rem; overflow-x: auto; font-size: .86rem; line-height: 1.5;
}
code { font-family: ui-monospace, "Cascadia Code", Menlo, monospace; }
p code, li code, td code {
  background: var(--card); border: 1px solid var(--line);
  padding: .06rem .3rem; border-radius: .25rem; font-size: .88em;
}

table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .94rem; }
th, td {
  text-align: left; padding: .55rem .6rem;
  border-bottom: 1px solid var(--line);
}
th {
  font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted);
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.win td { font-weight: 650; }
.tablewrap { overflow-x: auto; }

.grid {
  display: grid; gap: 1rem; margin: 1.5rem 0;
  grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
}
.card {
  border: 1px solid var(--line); border-radius: .6rem;
  background: var(--card); padding: 1.1rem 1.2rem;
}
.card h4 { margin: 0 0 .35rem; font-size: .98rem; }
.card p { margin: 0; font-size: .9rem; color: var(--muted); }

ol.flow { counter-reset: s; list-style: none; padding: 0; margin: 1.5rem 0; }
ol.flow li {
  counter-increment: s; position: relative; padding: .55rem 0 .55rem 2.6rem;
  border-bottom: 1px solid var(--line);
}
ol.flow li::before {
  content: counter(s); position: absolute; left: 0; top: .55rem;
  width: 1.7rem; height: 1.7rem; border-radius: 50%;
  background: var(--card); border: 1px solid var(--line);
  display: grid; place-items: center; font-size: .8rem; color: var(--muted);
}
ol.flow b { font-weight: 620; }

.calc {
  border: 1px solid var(--line); border-radius: .6rem;
  padding: 1.25rem; background: var(--card);
}
.calc .row {
  display: grid; gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
}
.calc label {
  display: block; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); margin-bottom: .3rem;
}
.calc input {
  width: 100%; padding: .5rem .6rem; font: inherit; font-size: .95rem;
  border: 1px solid var(--line); border-radius: .4rem;
  background: var(--bg); color: var(--fg);
}
.calc .out {
  margin-top: 1.25rem; padding-top: 1rem;
  border-top: 1px solid var(--line);
}
.calc .big { font-size: 1.8rem; font-weight: 680; font-variant-numeric: tabular-nums; }

.note {
  border: 1px solid var(--line); border-left: 3px solid var(--warn);
  background: var(--card); border-radius: .4rem; padding: .85rem 1rem;
  font-size: .9rem; color: var(--muted); margin: 1.25rem 0;
}
.note strong { color: var(--fg); }

footer {
  border-top: 1px solid var(--line); margin-top: 4rem; padding: 2rem 0 3rem;
  color: var(--muted); font-size: .88rem;
}
footer a { margin-right: 1.25rem; }
"""

CALCULATOR_JS = """
(function () {
  var saving = SAVING_PCT / 100;
  function n(id) { return parseFloat(document.getElementById(id).value) || 0; }
  function money(v) {
    return '$' + v.toLocaleString('en-US', {
      minimumFractionDigits: 2, maximumFractionDigits: 2
    });
  }
  function update() {
    var reqs = n('reqs'), inTok = n('intok'), outTok = n('outtok');
    var pin = n('pin'), pout = n('pout');
    var perReq = (inTok / 1e6) * pin + (outTok / 1e6) * pout;
    var now = perReq * reqs;
    var after = now * (1 - saving);
    document.getElementById('now').textContent = money(now);
    document.getElementById('after').textContent = money(after);
    document.getElementById('saved').textContent = money(now - after);
  }
  var ids = ['reqs', 'intok', 'outtok', 'pin', 'pout'];
  for (var i = 0; i < ids.length; i++) {
    document.getElementById(ids[i]).addEventListener('input', update);
  }
  update();
})();
"""


@dataclass(frozen=True)
class SiteContext:
    """What the page needs to know about the instance serving it."""

    version: str
    repo_url: str = "https://github.com/vbvansh/switchboard"
    #: True when this instance has no provider configured - which is the normal
    #: state for a public demo, since a hosting platform cannot run Ollama.
    demo_mode: bool = True


def _feature(title: str, body: str) -> str:
    return f'<div class="card"><h4>{title}</h4><p>{body}</p></div>'


def render(context: SiteContext) -> str:
    repo = context.repo_url
    demo_note = (
        '<div class="note"><strong>This instance is a demo.</strong> '
        "It is running the real software, but a hosting platform cannot run a "
        "local model, so no provider is connected and "
        "<code>/v1/chat/completions</code> will report that it has nowhere to "
        "send your request. Everything else &mdash; "
        '<a href="/health">/health</a>, <a href="/dashboard">/dashboard</a>, '
        '<a href="/metrics">/metrics</a> &mdash; is live. Run it yourself and '
        "connect your own models.</div>"
        if context.demo_mode
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Switchboard &mdash; a router for AI models</title>
<meta name="description" content="Self-hostable AI model router. Sends each
request to the cheapest model likely to get it right, enforces budgets, and
records what everything cost. Measured: 88.3% accuracy at 57% lower cost than
the best single model.">
<style>{STYLE}</style></head>
<body>

<nav><div class="wrap">
  <span class="brand">Switchboard</span>
  <a href="#how">How it works</a>
  <a href="#results">Results</a>
  <a href="#providers">Providers</a>
  <a href="#limits">Limitations</a>
  <a href="/dashboard">Dashboard</a>
  <a href="{repo}">GitHub</a>
</div></nav>

<div class="wrap">

<header>
  <h1>Stop sending every question to your most expensive model.</h1>
  <p class="lede">
    Switchboard sits between your application and your models. It works out
    which model is cheap enough <em>and</em> good enough for each request,
    enforces per-developer budgets, and writes down what everything cost
    &mdash; including what it would have cost the old way.
  </p>

  <pre><code>pip install -r requirements.txt
python -m switchboard serve

# then change one line in your app:
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-...")</code></pre>

  <div class="headline">
    <div class="stat">
      <div class="k">Measured accuracy</div>
      <div class="n">{MEASURED_ACCURACY}%</div>
      <div class="s">vs {BASELINE_ACCURACY}% for the best single model</div>
    </div>
    <div class="stat">
      <div class="k">Measured cost</div>
      <div class="n good">&minus;{MEASURED_SAVING_PCT:.0f}%</div>
      <div class="s">$6.42 vs $15.07 on the same 1,200 questions</div>
    </div>
    <div class="stat">
      <div class="k">Runs</div>
      <div class="n">Anywhere</div>
      <div class="s">your laptop, your server, fully offline if you want</div>
    </div>
  </div>

  <p class="muted" style="font-size:.9rem">
    More accurate <em>and</em> cheaper than the best model money can buy, on
    held-out questions. Every number on this page is reproducible with a
    command in the repository &mdash; see
    <a href="{repo}/blob/main/docs/RESULTS.md">the results document</a>, which
    also lists what these numbers do <em>not</em> prove.
  </p>
</header>

{demo_note}

<section id="why">
  <h2>The problem</h2>
  <p>
    Give a team unrestricted access to a top-tier model and three things are
    true at once, and only the third one is hard.
  </p>
  <div class="grid">
    {_feature("Most requests do not need it",
              "Formatting a string and proving a theorem go to the same "
              "$15-per-million-tokens model, because that is the one "
              "everybody configured.")}
    {_feature("Price does not predict quality",
              "On one benchmark, two flagship models score identically while "
              "one costs twice as much. On another, the cheapest model in the "
              "pool beats one costing 24 times more.")}
    {_feature("Nobody can tell which is which",
              "That is the actual hard part, and it is the part Switchboard "
              "was built to measure rather than assert.")}
  </div>
</section>

<section id="how">
  <h2>What happens to one request</h2>
  <p>
    Every step that can refuse a request happens <strong>before</strong> a model
    is called, so a refused request costs nothing.
  </p>
  <ol class="flow">
    <li><b>Identify the caller.</b> Per-developer API keys, stored only as a
        one-way hash. Unknown key, and nothing else runs.</li>
    <li><b>Rate limit.</b> A monthly budget does not stop a runaway script
        spending it in ninety seconds. A per-minute limit does.</li>
    <li><b>Choose a model.</b> A trained classifier estimates each model's
        chance of answering correctly, and the cheapest one clearing your
        confidence threshold wins. The reason is recorded in words.</li>
    <li><b>Check the usage policy.</b> Flags requests that look personal rather
        than work. By default it labels them and serves them anyway.</li>
    <li><b>Check the budget.</b> Over the monthly allowance, and the request
        stops here &mdash; recorded, and charged nothing.</li>
    <li><b>Check the cache.</b> An identical, deterministic request already
        answered is returned from memory, and recorded as costing zero.</li>
    <li><b>Call the model.</b> Transient failures are retried; a dead provider
        is failed over and then skipped by a circuit breaker.</li>
    <li><b>Record everything.</b> Who, which model, tokens, cost, latency, the
        routing reason &mdash; and what the same request would have cost on
        your top-tier model.</li>
  </ol>
  <p class="muted">
    Your request body is passed through untouched; only the model name is ever
    rewritten. That is why features this code has never heard of keep working.
  </p>
</section>

<section id="results">
  <h2>What was measured</h2>
  <p>
    Scored offline against public datasets containing hundreds of thousands of
    answers real models already gave, with their real prices. No API calls, no
    spend, and nothing self-reported.
  </p>
  <p class="muted" style="font-size:.92rem">
    MMLU-Pro, 6 flagship models, 1,200 <strong>held-out</strong> questions.
  </p>
  <div class="tablewrap"><table>
    <thead><tr>
      <th>Strategy</th><th class="num">Accuracy</th><th class="num">Cost</th>
      <th>On the trade-off curve?</th>
    </tr></thead>
    <tbody>
      <tr><td>Always the cheapest model</td><td class="num">84.0%</td>
          <td class="num">$0.71</td><td>yes</td></tr>
      <tr class="win"><td>Switchboard, threshold 0.60</td>
          <td class="num">{MEASURED_ACCURACY}%</td><td class="num">$6.42</td>
          <td>yes</td></tr>
      <tr><td>Hand-written keyword rules</td><td class="num">80.8%</td>
          <td class="num">$2.80</td><td>no &mdash; beaten outright</td></tr>
      <tr><td>Pick at random</td><td class="num">83.8%</td>
          <td class="num">$12.99</td><td>no &mdash; beaten outright</td></tr>
      <tr><td>Always the best model</td><td class="num">{BASELINE_ACCURACY}%</td>
          <td class="num">$15.07</td><td><strong>no &mdash; beaten
          outright</strong></td></tr>
    </tbody>
  </table></div>
  <p>
    "Beaten outright" means something else is better on <em>both</em> accuracy
    and cost, so no preference or budget makes it the right choice. Always
    using the best model lands in that category, which is a stronger claim than
    simply saving money.
  </p>
</section>

<section id="calculator">
  <h2>What it might save you</h2>
  <div class="calc">
    <div class="row">
      <div><label for="reqs">Requests per month</label>
           <input id="reqs" type="number" value="100000" min="0"></div>
      <div><label for="intok">Input tokens each</label>
           <input id="intok" type="number" value="1200" min="0"></div>
      <div><label for="outtok">Output tokens each</label>
           <input id="outtok" type="number" value="400" min="0"></div>
      <div><label for="pin">Your model, $ per M in</label>
           <input id="pin" type="number" value="3.00" step="0.01" min="0"></div>
      <div><label for="pout">Your model, $ per M out</label>
           <input id="pout" type="number" value="15.00" step="0.01" min="0"></div>
    </div>
    <div class="out">
      <div class="row">
        <div><div class="k muted" style="font-size:.78rem;text-transform:uppercase">
             Today</div><div class="big" id="now">&mdash;</div></div>
        <div><div class="k muted" style="font-size:.78rem;text-transform:uppercase">
             With routing</div><div class="big" id="after">&mdash;</div></div>
        <div><div class="k muted" style="font-size:.78rem;text-transform:uppercase">
             Saved per month</div>
             <div class="big good" id="saved">&mdash;</div></div>
      </div>
    </div>
  </div>
  <div class="note">
    <strong>This applies our measured {MEASURED_SAVING_PCT:.0f}% figure to your
    volumes. It is an illustration, not a quote.</strong>
    That figure comes from one benchmark with six specific models. Your traffic
    is a different shape, your model pool is different, and the honest way to
    find your own number is to run Switchboard in shadow mode for a week: it
    records what routing <em>would</em> have chosen on your real traffic while
    changing nothing.
  </div>
</section>

<section id="providers">
  <h2>Which models it works with</h2>
  <p>
    Anything that speaks OpenAI's request format, which is most of the industry
    &mdash; adding one is editing a text file, not writing code. Anthropic and
    Google do not, so Switchboard translates for them natively.
  </p>
  <div class="grid">
    {_feature("Hosted, OpenAI format",
              "OpenAI, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, "
              "Perplexity, Cerebras, DeepInfra, OpenRouter and anything else "
              "that copied the format.")}
    {_feature("Anthropic and Google, natively",
              "Claude and Gemini through their own APIs, translated in both "
              "directions, with no reseller in between taking a markup or an "
              "outage.")}
    {_feature("Your own hardware",
              "Ollama, vLLM, LM Studio, llama.cpp, TGI. Switch on local-only "
              "mode and Switchboard refuses to start if any configured "
              "provider is off-machine.")}
  </div>
  <h3>Model discovery</h3>
  <p>
    Hand-typing three hundred model names and prices is not a plan. Ask a
    provider what it has instead:
  </p>
  <pre><code>python -m switchboard discover openrouter</code></pre>
  <p class="muted">
    OpenRouter publishes real prices through its API, so those come back ready
    to use. Providers that publish no prices come back marked
    <code>REPLACE ME</code> and will not load until you fill them in &mdash;
    deliberately. A price Switchboard guessed would flow straight into your
    budgets and savings reports and be wrong in a way nobody could see.
  </p>
</section>

<section id="features">
  <h2>What else is in it</h2>
  <div class="grid">
    {_feature("An auditable ledger",
              "One row per request. Every row stores what it would have cost "
              "on your top-tier model, so the savings figure is a query, not a "
              "reconstruction.")}
    {_feature("Shadow mode",
              "Run the router on real traffic, record what it would have "
              "chosen, then ignore the decision and serve the request "
              "normally. Trial routing having risked nothing.")}
    {_feature("Budgets and rate limits",
              "Per-developer monthly budgets and per-minute limits. Refusals "
              "are recorded and cost nothing, so retrying cannot dig a deeper "
              "hole.")}
    {_feature("Caching, retries, failover",
              "Identical requests answered free from memory. Transient "
              "failures retried. Dead providers failed over and then skipped "
              "by a circuit breaker.")}
    {_feature("A usage policy that admits it is fallible",
              "Flags personal-looking requests without blocking them, and "
              "reports its own false-positive rate, including the prompts it "
              "got wrong.")}
    {_feature("Operable",
              "A dependency-free dashboard, Prometheus metrics, liveness and "
              "readiness probes, schema migrations that never destroy data, "
              "and a Docker image that runs as a non-root user.")}
  </div>
</section>

<section id="limits">
  <h2>What it does not do</h2>
  <p>
    A product whose argument is "we measure honestly" cannot have a landing page
    that only lists wins. These are real, current, and documented in the
    repository rather than discovered later.
  </p>
  <div class="grid">
    {_feature("The router does not transfer to short chat prompts",
              "It is trained on 700-character benchmark questions. Shown a "
              "34-character one, it returns roughly the same confidence for "
              "every model. Shadow mode exists to collect the traffic that "
              "fixes this.")}
    {_feature("One decisive win, one partial",
              "On a second dataset the router beats every naive baseline and "
              "sits on the trade-off curve throughout, but does not beat the "
              "best single model on accuracy.")}
    {_feature("Tool calls are not translated for Claude or Gemini",
              "The native adapters cover chat and streaming. For tool calling, "
              "reach those models through OpenRouter, which implements it. An "
              "honest gap beats a translation that half works.")}
    {_feature("The adapters are untested against live APIs",
              "They are written to the published specifications and covered by "
              "tests using recorded response shapes. Nobody has yet run them "
              "with a paid key.")}
    {_feature("Local models are priced, not free",
              "Prices for local models are simulated, so budgets and savings "
              "mean something. Every screen that shows money says so.")}
    {_feature("Shadow mode cannot judge quality",
              "The shadow model was never called, so there is no answer to "
              "grade. Its savings are projections, and are labelled as "
              "projections everywhere they appear.")}
  </div>
</section>

<section id="start">
  <h2>Run it</h2>
  <pre><code>git clone {repo}
cd switchboard
docker compose up --build

# http://localhost:8000/dashboard</code></pre>
  <p>
    Apache 2.0 licensed, including a patent grant. Self-hosted, so your prompts
    go to the providers you configure and nowhere else &mdash; there is no
    Switchboard service in the middle, because there is no Switchboard service.
  </p>
</section>

<footer><div>
  <a href="{repo}">Source</a>
  <a href="{repo}/blob/main/docs/ARCHITECTURE.md">Architecture</a>
  <a href="{repo}/blob/main/docs/RESULTS.md">Results</a>
  <a href="/dashboard">Dashboard</a>
  <a href="/health">Health</a>
  <p style="margin-top:1rem">
    Switchboard {context.version} &mdash; Apache 2.0. Benchmark datasets belong
    to their original authors and are not redistributed here.
  </p>
</div></footer>

</div>
<script>{CALCULATOR_JS.replace("SAVING_PCT", str(MEASURED_SAVING_PCT))}</script>
</body></html>"""
