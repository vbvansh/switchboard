# Switchboard

A self-hostable AI model router. It sits between an application and a pool of
models, decides which model is cheap enough and good enough for each request,
enforces per-user budgets, and records what every request cost — and what it
would have cost otherwise.

Bring your own providers. Switchboard speaks the OpenAI API, so any application
that already talks to OpenAI works by changing one URL, and any provider that
speaks that format — Ollama, OpenAI, Groq, OpenRouter, Together, vLLM, LM Studio
— is a few lines of YAML away.

**Runs entirely offline if you want it to.** Point it at a local Ollama and turn
on `SWITCHBOARD_LOCAL_ONLY=true`, and it will refuse to start if any configured
provider is off-machine.

## Status

`model: "auto"` routes to the cheapest model predicted to answer correctly,
subject to per-request latency, cost and quality limits. What works today:

- `POST /v1/chat/completions`, streaming and non-streaming
- Drop-in compatibility with any OpenAI client via `base_url`
- Multi-provider: add any OpenAI-compatible provider in `providers.yaml`
- Per-developer API keys (`401`) and monthly budgets (`402`)
- Every request logged: who, which model, tokens, cost, latency
- Docker image, PostgreSQL support, liveness/readiness probes
- Schema migrations — upgrades never destroy your data
- Offline evaluation against 696k recorded answers from real models
- Learned routing in the live API, with an auditable reason per request
- Response caching, retries, provider failover with a circuit breaker
- Per-user rate limiting and a Prometheus `/metrics` endpoint
- Shadow mode: measure routing on your own traffic before trusting it
- A dependency-free `/dashboard` page

| Phase | | |
|---|---|---|
| A | Foundations: licence, privacy, migrations, providers, Docker | done |
| B | Real benchmark data | done |
| C | The routing brain | done — routing is live |
| D | Caching, retries, failover, rate limits, metrics | done |
| E | Shadow mode + dashboard | done |
| F | Guardrails, docs, write-up | |

## Why this exists

Giving every developer unrestricted access to a top-tier model is expensive,
and most requests don't need one. Switchboard measures how much of that spend
can be recovered by routing on task complexity, and what it costs in quality.

The results are what matter, so the plan is to measure honestly:

- **Baselines first** — always-cheap, always-expensive, random, and a naive
  keyword heuristic. A routing result without these means nothing.
- **Objective ground truth** — evaluation uses GSM8K / HumanEval-style tasks
  where correctness is checkable in code, so no LLM judge is needed.
- **Two cost views, never blurred** — a *simulated* dollar cost (local models
  mapped onto published per-token prices, so budget logic is meaningful) and
  *measured* real cost (latency, tokens/sec, model swap count).

## Benchmarks: real models, real costs

Switchboard is evaluated against public routing benchmarks — hundreds of
thousands of recorded answers from real models, with real costs. Every routing
strategy can be scored offline, exactly, **without a single API call**.

```powershell
python scripts/fetch_llmrouterbench.py --extract   # ~1.3 GB download
python -m switchboard bench build all              # normalise into a fast cache
python -m switchboard bench list
python -m switchboard bench headroom llmrouterbench --suite gpqa
```

Two sources, because they cover each other's gaps:

| | [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) | [xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench) |
|---|---|---|
| Models | 40, incl. GPT-5, Claude, Gemini | 18 open-weight |
| Rows | 548,059 | 147,906 |
| Suites | 27 (GPQA, HLE, SWE-bench, …) | 13 (MMLU, GSM8K, MBPP, …) |
| Per-query latency | no | **yes** |

**The datasets are not redistributed here.** Neither declares a licence, and
both derive from many upstream benchmarks with mixed terms plus outputs from
commercial models. This repository ships a download script and the analysis
code; the data is fetched from its original source. Please cite the papers if
you use them.

### What the data says

Perfect routing beats the single best model, on both sources independently:

```
GPQA, 8 flagship models, 198 questions
  cheapest  (qwen3-235b)      58.6%   $0.07
  best      (gemini-2.5-pro)  84.8%  $16.59
  ORACLE                      96.0%   $1.07     +11.1 pts, 94% cheaper

xRouteBench, 18 models, 8,217 questions
  cheapest  (gpt-oss-20b)     68.6%   $0.12
  best      (cogito-671b)     77.3%   $3.74
  ORACLE                      91.7%   $0.21     +14.4 pts, 94% cheaper
```

Price does not predict quality. On GPQA, Gemini 2.5 Pro and GPT-5 score
identically while Gemini costs twice as much; on xRouteBench the cheapest model
in the pool beats one costing 24× more. Those inefficiencies are what a router
exists to exploit.


### Where the simple baselines land

Replaying the naive strategies against recorded answers, with no API calls:

```powershell
python -m switchboard bench replay llmrouterbench --suite gpqa
python -m switchboard bench replay xroutebench
```

GPQA, 8 flagship models, 198 questions:

| Strategy | Accuracy | Cost | Saving vs best | Trade-off curve |
|---|---|---|---|---|
| always-cheapest | 58.6% | $0.07 | 99.6% | on |
| keyword heuristic | 77.8% | $0.51 | 96.9% | on |
| *oracle (impossible)* | *96.0%* | *$1.07* | *93.6%* | *ceiling* |
| random | 70.7% | $3.96 | 76.1% | **dominated** |
| always-best (gemini-2.5-pro) | 84.8% | $16.59 | — | on |

Two things worth noting. `random` is **dominated** — the keyword heuristic is
both cheaper and more accurate, so there is no reason to ever pick randomly.
And on xRouteBench the result flips: the keyword heuristic is dominated there,
scoring 57.9% where simply always using the cheapest model scores 68.6%.

A hand-written heuristic that helps on one workload and actively hurts on
another is worse than a consistent baseline, because you cannot tell in advance
which case you are in. That is the problem the learned router has to solve.

### The learned router

`switchboard bench train` learns, per model, the probability it answers a given
question correctly, then routes to the cheapest model clearing a confidence
threshold. Sweeping that threshold traces a whole cost/quality curve from one
trained model. Everything is scored on **held-out questions**.

```powershell
python -m switchboard bench train llmrouterbench --suite mmlupro --models "gpt-5,claude-sonnet-4,gemini-2.5-pro,gemini-2.5-flash,kimi-k2-0905,qwen3-235b-a22b-2507"
```

MMLU-Pro, 6 flagship models, 1,200 held-out questions:

| Strategy | Accuracy | Cost | Saving | Curve |
|---|---|---|---|---|
| always-cheapest | 84.0% | $0.71 | 95% | on |
| learned @0.40 | 86.2% | $2.23 | 85% | on |
| *oracle (impossible)* | *93.6%* | *$2.43* | *84%* | *ceiling* |
| learned @0.50 | 87.4% | $3.68 | 76% | on |
| **learned @0.60** | **88.3%** | **$6.42** | **57%** | **on** |
| keyword | 80.8% | $2.80 | 81% | dominated |
| random | 83.8% | $12.99 | 14% | dominated |
| always-best | 86.8% | $15.07 | — | **dominated** |

**The learned router dominates always-best**: 88.3% versus 86.8% accuracy, at
57% lower cost. More accurate *and* cheaper than the best single model money
can buy — which is the outcome the whole project was testing for.

It also dominates both naive baselines. On xRouteBench it dominates `random`
and `keyword` and sits on the trade-off curve throughout, but does **not** beat
always-best on accuracy there. One decisive win, one partial — reported as
measured.

Prediction quality is reported as AUC per model (0.5 = guessing, 1.0 = perfect):
mean **0.745** on MMLU-Pro and **0.800** on xRouteBench. If those sat near 0.5
the features would carry no signal and no routing rule could help.

#### Cascades: call cheap, look, then decide

The learned router guesses from the question alone, before calling anything. A
**cascade** decides after: it pays for a cheap call, inspects the answer, and
escalates only if unconvinced. Two ways to judge that answer, neither allowed to
peek at whether it was actually correct:

- **agreement** — ask two cheap models; matching answers are evidence, a
  disagreement escalates. No training needed.
- **learned verifier** — a classifier predicting "was the cheap model right?"
  from the question plus what the cheap model did: answer length, and whether a
  second model agreed.

A cascade that escalates has paid for **both** calls, and the scoring charges
for every call made. Charging only the final model is the easiest way to make a
cascade look better than it is.

The two datasets disagree, which is the interesting part:

| | MMLU-Pro (6 flagships) | xRouteBench (18 open models) |
|---|---|---|
| Best learned router | **88.3%** @ $6.42 | 74.8% @ $0.33 |
| Best cascade | 87.5% @ $7.96 | **77.4%** @ $0.89 |
| Cascades on the curve | **0 of 6** | **5 of 6** |

On MMLU-Pro every cascade is **dominated** — escalating to a $15 model even a
third of the time costs more than the learned router's spread across six. On
xRouteBench, where models are closer in price, the double-call penalty is small
and the better information wins: cascades extend the frontier past where the
learned router tops out.

The lesson is about price spread, not about cascades being good or bad. When
the escalation target is far more expensive than the alternatives, paying twice
rarely pays off.

#### Latency SLAs: what a speed promise costs

An **SLA** (Service Level Agreement) is a promise about how a service behaves.
A latency SLA promises speed — "95% of requests answered within 4 seconds".

A router that optimises only cost and accuracy will happily pick a model that
takes 10 seconds. `switchboard bench sla` measures what promising otherwise
actually costs:

```powershell
python -m switchboard bench sla xroutebench --budgets "2.0,4.0,6.0,10.0"
```

xRouteBench, 2,466 held-out questions, 18 models:

| SLA | Accuracy | Cost | p95 latency | Violations | Models eligible |
|---|---|---|---|---|---|
| no SLA | 71.8% | $0.120 | 6.68s | — | 18 |
| ≤ 2s | — | — | — | — | **impossible** |
| ≤ 4s | 55.6% | $0.071 | 3.61s | 3.9% | 2 |
| ≤ 6s | 62.5% | $0.253 | 5.06s | 1.8% | 5 |
| ≤ 10s | 71.9% | $0.129 | 6.55s | 1.9% | 14 |

The tightest achievable promise gives up **16.1 percentage points of accuracy**
and keeps violations under the usual 5% target. No model in this pool can
promise 2 seconds at all — which is itself the answer: add a faster model or
loosen the budget.

Two rules keep this honest:

**Eligibility uses the tail, not the median.** The first version selected models
by median latency and produced a "fast" set whose p95 was *worse* than routing
with no SLA at all. `deepseek-v3.1` answers in 0.95s typically and 16.4s at p95.
To promise p95 ≤ B you must require p95 ≤ B.

**Violations are counted against what actually happened** — the recorded
per-request latency, not the averages the router used to decide. Picking a fast
model is not the same as being fast.

#### A note on features

The default text representation is **TF-IDF**, not neural embeddings.
Embeddings are supported (`--features embedding`) and are richer in principle,
but were measured at roughly **0.5 texts/second** on the development machine —
a 17-minute run that never finished. TF-IDF fits thousands of questions in
under a second and the whole experiment takes 12 seconds. On a machine with a
GPU, use embeddings.

### Routing in the live API

`model: "auto"` now routes. Train an artifact, and the server loads it at
startup:

```powershell
python -m switchboard router train xroutebench
python -m switchboard router info
python -m switchboard serve
```

Per-request limits arrive as headers, so the body stays a valid OpenAI request:

| Header | Effect |
|---|---|
| `X-Switchboard-Max-Latency` | seconds; models slower than this are excluded |
| `X-Switchboard-Min-Quality` | minimum predicted chance of success |
| `X-Switchboard-Max-Cost` | per-request budget cap |

`/health` reports whether routing is on, which models it can drive, and what it
was trained on. Every request records its `routing_reason` in the ledger, so a
decision that looks wrong can be explained after the fact.

**Name mapping.** A router trained on public benchmarks knows
`qwen2.5-7b-instruct`; your catalog has `qwen2.5:7b`. Each model in
`providers.yaml` declares `benchmark_alias` to say which benchmark model it
stands in for. `switchboard router info` shows what mapped and what did not.
With fewer than two models mapped, routing stays off and says why — a stale
artifact must never take the service down.

#### An honest limitation

A benchmark-trained router **does not transfer to short chat prompts.**

```
trained-shaped prompts (709 chars avg):  p spans 0.02–0.88, 38% below threshold
short chat prompts      (34 chars avg):  p clusters 0.67–0.87, no discrimination
```

Measured on its own distribution the classifier works (held-out AUC 0.800).
Shown a 34-character question it has never seen anything like, it returns
roughly the same probability for every model and everything goes to the
cheapest one.

This is distribution shift, not a broken model, and it is only visible once the
router is wired to real traffic — which is the argument for doing C.4 rather
than trusting the offline table. The fix is to train on the traffic you
actually serve; the ledger already records what it needs for that.

## Hardware note

Developed on a 4 GB GPU with 7.3 GB system RAM. That constraint shapes the
design: only about two small models stay resident, so routing to a larger tier
evicts the warm set and the *next* few cheap requests pay a reload penalty too.
The router accounts for warm/cold state rather than pretending model choice is
free. This mirrors GPU-pool locality behaviour in real serving stacks.

## Run with Docker

```bash
docker compose up --build
```

That builds the image, applies database migrations, and starts the server on
`http://localhost:8000`. It talks to an Ollama running on your host machine.

```bash
curl http://localhost:8000/health
```

To use PostgreSQL instead of the default SQLite file:

```bash
docker compose --profile postgres up --build
```

then set `SWITCHBOARD_DATABASE_URL` on the switchboard service to
`postgresql+psycopg://switchboard:switchboard@postgres:5432/switchboard`.

The ledger lives in a named volume. Without it, replacing the container would
destroy every user, budget and spending record.

## Run from source

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
pip freeze > requirements.lock.txt

copy .env.example .env
```

`requirements.txt` is what the server needs. `requirements-dev.txt` adds tests
and the evaluation harness, which pull in a large scientific stack the running
server never uses — which is why the container image ships without them.

Create the database:

```powershell
python -m switchboard db upgrade
```

Run this after every upgrade of Switchboard too. It applies any schema changes
without touching your data, and it is safe to run when nothing has changed.
Switchboard refuses to start against a database it does not recognise rather
than failing confusingly later.

See what providers and models are configured, and check they are reachable:

```powershell
python -m switchboard providers
python -m switchboard check
```

## Adding a provider

Edit [`providers.yaml`](providers.yaml) — no code changes. Set `enabled: true`,
list the models you want with their prices, and name the environment variable
holding your key:

```yaml
- id: groq
  type: openai-compatible
  base_url: https://api.groq.com/openai/v1
  api_key_env: GROQ_API_KEY
  enabled: true
  models:
    - id: llama-3.3-70b-versatile
      tier: T2
      input_per_mtok: 0.59
      output_per_mtok: 0.79
      context_window: 131072
```

**Keys never go in this file.** `api_key_env` names an environment variable;
the key lives in `.env` or your secret store. `providers.yaml` stays safe to
commit.

Claude and other Anthropic models are reachable through OpenRouter, which
exposes them in the OpenAI format — no separate adapter needed.

Create a developer. The API key is shown **once** — only its hash is stored.

```powershell
python -m switchboard users add alice --budget 50
```

## Run

```powershell
python -m switchboard serve
```

Point any OpenAI client at it — the only changes are `base_url` and the key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-swbd-...",             # the key printed by `users add`
)

response = client.chat.completions.create(
    model="auto",                      # hands model choice to Switchboard
    messages=[{"role": "user", "content": "Explain a hash map in two sentences."}],
)
print(response.choices[0].message.content)
```

`model="auto"` is the routing seam. Today it resolves to `default_model`; from
milestone 4 it triggers a real routing decision. An explicit model name is
always honoured, which is what makes per-tier benchmarking possible.

### From PowerShell

Use `Invoke-RestMethod` rather than `curl`. PowerShell rewrites arguments before
they reach `curl.exe`, so JSON bodies with escaped quotes arrive corrupted —
a confusing failure that looks like a server bug and is not one.

```powershell
$key  = "sk-swbd-..."
$body = @{
  model    = "auto"
  messages = @(@{ role = "user"; content = "What is 17 + 28?" })
} | ConvertTo-Json -Depth 5

$r = Invoke-RestMethod -Uri "http://localhost:8000/v1/chat/completions" `
     -Method Post -Headers @{ Authorization = "Bearer $key" } `
     -ContentType "application/json" -Body $body -TimeoutSec 300

$r.choices[0].message.content
```

## Admin

```powershell
python -m switchboard usage                      # spend and savings this month
python -m switchboard users list
python -m switchboard users budget alice 100     # change a limit
python -m switchboard users deactivate alice     # block without deleting history
```

## Money is simulated

Local inference costs nothing, so budgets would be meaningless. Each local model
is assigned the price tag of a commercial model of comparable capability, in
[`switchboard/prices.json`](switchboard/prices.json) — a file rather than a
database table, so price changes appear in a git diff and historical results
stay explainable.

Every request also records what it **would** have cost on the top tier
(`qwen2.5:7b`). That difference is the savings figure, stored per row rather
than reconstructed later.

Real measured cost — latency, token counts, model switches — is recorded
separately and never mixed with the simulated dollars.

## Privacy

**Switchboard does not store prompt text by default.** Only token counts, costs,
timings, and which model served each request are recorded.

Storing prompt text is available and useful — the routing classifier learns from
real examples — but it means the database holds whatever your users type, which
in most organisations includes customer data and credentials. So it is an
explicit opt-in:

```
SWITCHBOARD_STORE_PROMPTS=true
```

If you turn it on: tell your users, protect the database file, and check what
your local data-protection rules require. The database is excluded from git.

## Caching and retries

### Response cache

Identical requests are answered from memory for nothing. This is the largest
saving a gateway can offer and the simplest to reason about.

```
X-Switchboard-Cache: hit    served from memory, cost $0
X-Switchboard-Cache: miss   went to the provider, and was stored
X-Switchboard-Cache: skip   not eligible - see below
```

Four rules, each of them a way this could go wrong:

**Only byte-identical requests hit.** The key covers the model, the messages
and every sampling option that changes the output. "Similar" questions are not
matched. Semantic matching would return an answer to a question nobody asked,
and a cache that is occasionally confidently wrong is worse than no cache.

**Nothing random is cached.** A request at `temperature: 0.8` is asking for
variety. Returning a stored answer would silently defeat that.

**Hits cost nothing, and the ledger says so.** A hit recorded at full price
would inflate every savings figure. It is stored with `status: cached`, zero
cost, and the baseline it would have cost — so cache savings show up in
`switchboard usage` alongside routing savings.

**Entries expire and the cache is size-bounded** (`SWITCHBOARD_CACHE_TTL_S`,
`SWITCHBOARD_CACHE_MAX_ENTRIES`; set entries to 0 to disable). Models get
replaced; an answer from three weeks ago may not be what that model says today.

### Retries

Transient failures are retried with exponential backoff and jitter.

| Retried | Not retried |
|---|---|
| 408, 409, 429, 500, 502, 503, 504 | 400, 401, 403, 404, 422 |
| connection resets, timeouts | programming errors |

A malformed request will be malformed the second time too — retrying it wastes
time and, on a paid provider, money. A `Retry-After` header from the provider
wins over our own backoff, clamped so a very long one cannot hold a request
open indefinitely. Jitter matters: without it every client that failed during
an outage retries in the same instant and knocks the provider over again.

## Shadow mode

Nobody sensible points a new routing system at production traffic and hopes.

Shadow mode runs the router on every request, records the model it would have
chosen and what that would have cost, and then **ignores the decision** and
serves the request exactly as it would have been served anyway.

```powershell
$env:SWITCHBOARD_SHADOW_MODE = "true"
python -m switchboard serve
# ... a week of real traffic ...
python -m switchboard shadow
```

After a week you have a report on your own workload:

```
Requests shadowed              1,284
Cost as served                 $41.20
Cost if routed (estimated)     $14.07
Projected saving               $27.13 (65.8%)
Different model chosen         912 (71%)
  ... to something cheaper     889
  ... to something dearer       23
```

That is a decision an engineering manager can act on, having risked nothing.

It also fixes the limitation found in Phase C.4. A router trained on public
benchmarks does not understand short chat prompts; the fix is training on the
traffic you actually serve, and shadow mode is what collects it with the
router's opinion attached.

### Two honest limits

**The shadow cost is an estimate.** The shadow model was never called, so its
token count does not exist. The estimate reuses the tokens the real model
produced and prices them at the shadow model's rates. A chattier model would
truly have cost more. This is labelled as a projection everywhere it appears.

**It cannot tell you whether quality would have held up.** No answer was
produced to grade. What it can report is the router's own predicted probability
of success — a forecast, not evidence.

## Dashboard

`GET /dashboard` renders spend, savings, per-model traffic, cache hit rate and
the shadow projection as a single page.

Server-rendered HTML with inline styles. No JavaScript, no build step, no
external fonts or scripts — it works on a machine with no internet access,
which is exactly what `SWITCHBOARD_LOCAL_ONLY` exists to support, and nothing
loaded from a CDN can observe when your engineers check their AI spend.

The numbers come from the same ledger the CLI reads, so the page and
`switchboard usage` can never disagree.

## Reliability: failover, rate limits, metrics

### Failover

Several providers may serve the same model. Declare it more than once in
`providers.yaml` and the extras become backups, tried in the order they appear:

```yaml
- id: groq
  models: [{id: llama-3.3-70b, ...}]      # preferred
- id: together
  models: [{id: llama-3.3-70b, ...}]      # backup
```

A provider that raises, or returns a 5xx, is recorded as failed and the next
one is tried. A **4xx is not a failover** — the request itself is wrong, and
every other provider would reject it identically. Retrying it elsewhere would
multiply the cost of one bad request by the number of providers configured.

### Circuit breaker

Retries handle a blip; a breaker handles an outage. Without one, every request
to a dead provider waits for its full timeout before failing over — with a
60-second timeout and an hour-long outage, that is an hour of 60-second waits
for answers that were never coming.

After five **consecutive** failures a provider is skipped for 30 seconds, then
one trial request is allowed through. Success restores normal service; failure
starts the cooldown again. Only consecutive failures count, so a provider that
fails once an hour is never tripped.

A tripped provider moves to the **back** of the list rather than out of it: if
everything is failing, trying a dead provider still beats having nowhere to
send the request. `/health` reports the state of every circuit.

### Rate limiting

A monthly budget does not stop someone spending it in ninety seconds. A runaway
retry loop can burn a month's allowance before anyone notices, and hammer the
provider hard enough to get the whole organisation throttled.

Requests are counted in a **sliding** sixty-second window, per user. A fixed
window would let someone send a full allowance at 11:59:59 and another at
12:00:00 — twice the intended rate, in one second, entirely within the rules.

Refused requests get `429` with `Retry-After`, and are refused **before** any
provider is called. Per-user overrides live in the database; the default is
`SWITCHBOARD_RATE_LIMIT_PER_MINUTE`.

The counter is per process, so two instances behind a load balancer together
allow twice the limit. Making it exact needs Redis — a whole extra service for
a guard rail whose job is catching runaway loops, not metering billing.

### Metrics

`GET /metrics` serves the Prometheus text format. No credentials: metrics carry
no prompt text, no keys and no user identities, and a scrape endpoint that
needs a key is one nobody configures.

```
switchboard_requests_total{status="ok"}
switchboard_cache_events_total{event="hit"}
switchboard_provider_attempts_total{provider="groq",outcome="5xx"}
switchboard_failovers_total{to="together"}
switchboard_request_duration_seconds_bucket{model="...",le="0.5"}
```

Labels are drawn only from small fixed sets — a status, a provider, a model.
Never a user id or a request id: every distinct label combination becomes a
time series stored forever, and labelling by user is a well-known way to take
a monitoring system down with your own observability code.

## Health endpoints

| Endpoint | Purpose |
|---|---|
| `/health/live` | Is the process alive? Checks nothing else. |
| `/health/ready` | Can it serve traffic? 503 if the database or every provider is down. |
| `/health` | Detailed status for a human. |

The split matters under an orchestrator. A failing **liveness** probe causes a
restart; a failing **readiness** probe only stops traffic being routed to the
instance. If liveness checked providers, a provider outage would restart
Switchboard in a loop — fixing nothing and destroying its own logs.

## Database upgrades

Schema changes ship as migrations under [`migrations/`](migrations/). Upgrading
Switchboard never requires deleting your data:

```powershell
python -m switchboard db status     # what revision am I on?
python -m switchboard db upgrade    # apply anything missing
```

If you have a database created before migrations existed, its tables are already
correct — record that fact without re-creating them:

```powershell
python -m switchboard db stamp-baseline
```

## Licence

Apache 2.0 — see [LICENSE](LICENSE). You may use, modify, and redistribute this
commercially. The licence includes a patent grant, which is why companies tend
to prefer it over MIT.

## Tests

No Ollama instance required — the provider is stubbed.

```powershell
pytest
```

## Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 1 | OpenAI-compatible proxy | done |
| 2 | Ledger: users, budgets, simulated cost accounting | done |
| 3 | Baseline strategies + live eval harness (first real numbers) | next |
| 4 | Embedding classifier and cascade routing | |
| 5 | Guardrails, with honest false-positive-rate reporting | |
| 6 | Offline replay, Pareto plots, dashboard, writeup | |

## Model tiers

| Tier | Model | Intended for |
|------|-------|--------------|
| T0 | `qwen2.5:1.5b` | Classification, formatting, short factual |
| T1 | `qwen2.5:3b` | Routine generation |
| T2 | `qwen3:4b` | Reasoning-capable (emits `<think>` blocks) |
| T3 | `qwen2.5:7b` | Hard tasks; evicts other models from VRAM |
