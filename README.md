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
- A usage policy that flags personal requests and reports its own error rate
- Native Anthropic and Gemini adapters, plus model discovery from any provider
- A public landing page served by the app itself, deployable in one step
- Train the router on YOUR traffic, from ratings your application sends back
- Routing that works on the first request, with no training at all

| Phase | | |
|---|---|---|
| A | Foundations: licence, privacy, migrations, providers, Docker | done |
| B | Real benchmark data | done |
| C | The routing brain | done — routing is live |
| D | Caching, retries, failover, rate limits, metrics | done |
| E | Shadow mode + dashboard | done |
| F | Guardrails, docs, write-up | done |
| G | Universal providers: native Anthropic + Gemini, model discovery | done |
| H | Public landing page and one-step deployment | done |
| I | Feedback endpoint and training on your own traffic | done |
| J.1 | Installable as a package, with files in the right places | done |
| J.2 | Cold start: ladder routing + answer verification, no training | done |

**Documentation:** [Architecture](docs/ARCHITECTURE.md) - how it is put
together and what happens to one request. [Results](docs/RESULTS.md) -
every measured number, and what each one does not prove.
[Deploy](DEPLOY.md) - putting it on a public URL.

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

## Where Switchboard keeps its files

```powershell
python -m switchboard where
```

Two layouts, detected rather than configured:

| | Config and data go |
|---|---|
| **A git checkout, or the Docker image** | beside the package, exactly as before |
| **A pip install** | your operating system's config and data directories |

The rule is simple: if a `providers.yaml` sits next to the package, that is a
deliberately laid-out installation and nothing moves. Otherwise it is an
installed copy, and writing into `site-packages` would be both unwritable and
wiped on the next upgrade.

Override with `SWITCHBOARD_HOME` for everything, or `SWITCHBOARD_PROVIDERS_FILE`
and `SWITCHBOARD_DATABASE_URL` individually.

**One bug this fixed.** The database default used to be
`sqlite:///data/switchboard.db` — a *relative* path, resolved against whatever
directory you were standing in. Start the server from a different folder and
every user, budget and spending record appeared to be gone, with no error,
because SQLite cheerfully creates a fresh empty file. It is now absolute.

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

## Which models it works with

Three kinds of provider, and adding any of them is editing a text file.

**1. Anything speaking OpenAI's format** — most of the industry, because most of
the industry copied it. One adapter covers all of them:

> OpenAI, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, Perplexity,
> Cerebras, SambaNova, DeepInfra, Nvidia NIM, Azure OpenAI, OpenRouter, and every
> self-hosted server: Ollama, vLLM, LM Studio, llama.cpp, TGI.

**2. Anthropic and Google, natively.** They did not copy the format, so
Switchboard translates for them:

```yaml
  - id: anthropic
    type: anthropic          # <- selects the translating adapter
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    enabled: true
```

Your application still sends and receives OpenAI-shaped requests. The
translation handles the four differences that matter — the system prompt is a
separate field, `max_tokens` is mandatory, content comes back as a list, and
token counts have different names. That last one is the dangerous one: get it
wrong and every Claude request records as costing nothing, with no error
anywhere.

**Tool calls are not translated.** The native adapters cover chat and streaming.
For tool calling, reach those models through OpenRouter, which implements it. An
honest gap beats a translation that fails deep inside somebody's agent.

**The adapters are written to the published API specifications and covered by
tests using recorded response shapes — but nobody has run them against a paid
key yet.** Stated here rather than left for you to discover.

**3. Your own hardware.** Ollama, vLLM, LM Studio, llama.cpp, TGI. With
`SWITCHBOARD_LOCAL_ONLY=true`, startup fails if any enabled provider is
off-machine.

### Model discovery

Hand-typing three hundred model names and prices is not a plan:

```powershell
python -m switchboard discover openrouter
python -m switchboard discover openrouter --contains claude --limit 5
python -m switchboard discover gemini --out gemini-models.yaml
```

It asks the provider what it has and prints YAML to paste under that provider's
`models:` key.

**It will not invent a price.** OpenRouter publishes real per-token prices
through its API, so those come back ready to use. OpenAI, Anthropic, Google and
every local server publish no prices at all — those models come back with their
price lines marked `REPLACE ME`, and the catalog will not load until a human
fills them in. That is deliberate: a price this project guessed would flow
straight into budget enforcement and savings reports and be wrong in a way
nobody could see from the outside.

It also does not edit `providers.yaml` for you. That file is full of comments
explaining why each model is priced as it is, and no automatic rewriter
preserves them.

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

## Usage policy: is this actually work?

Every organisation that puts an LLM gateway in front of its engineers eventually
asks how much of the bill is real work. Switchboard can answer that **without
becoming the thing everybody hates.**

```powershell
python -m switchboard guardrails check "Plan my holiday to Goa next month"
python -m switchboard guardrails calibrate
python -m switchboard guardrails report
```

Three modes, set with `SWITCHBOARD_GUARDRAILS_MODE`:

| Mode | Behaviour |
|---|---|
| `off` | do nothing |
| **`flag`** (default) | score every request, label it in the ledger, **serve it normally** |
| `block` | refuse flagged requests with `403`, with an override header |

**Why the default flags instead of blocking.** The mistakes are not symmetric.
Missing a personal request costs a fraction of a cent. Wrongly blocking a real
one stops an engineer working, at the moment they are trying to work, with an
error message accusing them of slacking. The second is far worse, and far more
likely — the detector is regular expressions, and human language is not.

So the rules are biased towards letting things through. Technical content in a
prompt (code fences, stack traces, SQL, file paths) **subtracts** from the
score, and rules have weights: a phrase that could only be personal trips on its
own, one that turns up in real tickets needs a second signal.

### It reports its own error rate

```
Usage policy on 70 labelled prompts
  False-positive rate (the one that matters)    0.0%   (0 of 45 work prompts)
  Personal prompts caught (recall)             84.0%
  Precision                                   100.0%

Personal prompts this missed:
  - Help me pick a birthday gift for my mum
  - Which phone should I buy under 30000 rupees?
  - Tell me a joke about cats
  - My son has a school project on volcanoes, help him write it
```

**Read the caveat before quoting those numbers.** The 70 labelled prompts were
written by hand by this project's author, so they flatter the detector. They are
a smoke test, not evidence about your team. Run `flag` mode for a week, export
the flagged requests and label them yourself — that is the calibration set that
means anything.

### Overriding a block

If `block` mode gets it wrong, the refusal says so and says what to do:

```
This request was held by your organisation's usage policy (category: personal;
matched: holiday_planning). This check is a keyword match and it does get
things wrong. If this is work, resend it with the header
'x-switchboard-policy-override: <short reason>' and it will go through.
```

The override is a speed bump, not a security control. It is recorded with its
reason, so overrides are visible in the report.

### Your own rules

Start from [guardrails.example.yaml](guardrails.example.yaml):

```yaml
# SWITCHBOARD_GUARDRAILS_FILE=guardrails.yaml
rules:
  - name: holiday_planning
    label: personal
    weight: 1.0          # 1.0 trips on its own; 0.5 needs a second signal
    # SINGLE quotes. In double-quoted YAML `\b` means backspace, not a regex
    # word boundary, and the pattern would silently never match.
    pattern: '\b(plan|book) (my|our) (holiday|vacation)\b'
```

The file **replaces** the built-in rules rather than adding to them, so a
shipped rule that keeps catching your team's real work can actually be removed.
`switchboard guardrails report` shows which rules are doing the flagging, which
is how you find the one to delete.

### What it stores

The category and the names of the rules that matched. **Never the prompt text.**
A feature built to police what people type must not become the reason a company
starts recording what people type — that stays behind
`SWITCHBOARD_STORE_PROMPTS`, off by default, exactly as before.

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

## The website, and putting it online

Switchboard serves its own landing page at `/`, so one deploy gives you the
site, the dashboard, the API and the metrics endpoint at a single address.
There is no separate frontend to host and nothing to build.

```
https://your-app.onrender.com/            the landing page
https://your-app.onrender.com/dashboard   spend and savings
https://your-app.onrender.com/health      status
https://your-app.onrender.com/v1/...      the OpenAI-compatible API
```

The page is written under the same rules as the dashboard — server-rendered
HTML, inline CSS, one small inline script, nothing from a CDN. It renders on a
machine with no internet, and no third party learns who visited.

One content rule keeps it honest: **every number on the page also appears in
[docs/RESULTS.md](docs/RESULTS.md) with its method beside it, and a test fails
if they drift apart.** The limitations section is not a disclaimer at the
bottom; it is a section with the same weight as the results.

See **[DEPLOY.md](DEPLOY.md)** for the walkthrough — Render, Railway, Fly.io and
a plain server — plus the three settings that silently break a deploy if you
miss them:

- bind to `0.0.0.0`, not `127.0.0.1`, or the platform reports a startup timeout
- point the health check at `/health/live`, not `/health/ready` — a deployed
  instance normally has no provider, so readiness is `503` by design and the
  service would restart forever
- a free plan has no persistent disk, so SQLite loses every user and every
  spending record on redeploy; attach PostgreSQL for anything real

**There is no sign-up on the website.** It is informational; API keys are
created by an administrator with `switchboard users add`. Web accounts mean
passwords, sessions, email verification and account recovery — a real security
surface that deserves its own phase rather than being tacked onto a landing
page.

## Training the router on your own traffic

The honest limitation above says a benchmark-trained router goes blank on short
chat prompts. This is the fix, and it is the loop shadow mode was built to feed.

```powershell
python -m switchboard router data         # can I train yet, and if not why not
python -m switchboard router train-live   # train from my own ledger
```

### The missing half was a label

Training needs pairs of *(question, did this model get it right)*. A benchmark
ships with an answer key, so the second half is free. **Real traffic has no
answer key** — nobody wrote down the correct response to "why is this test
flaky". Without that, the ledger has thousands of questions and no outcomes,
and no amount of collecting improves anything.

So every response now carries a handle, and your application sends back a
verdict:

```
POST /v1/chat/completions
  -> 200, X-Switchboard-Request-Id: kJ8fQ2vXn4TbLm9wRzA1cQ

POST /v1/feedback
  {"request_id": "kJ8fQ2vXn4TbLm9wRzA1cQ", "rating": "bad", "note": "wrong API"}
```

In practice that is what a thumbs up/down in your UI calls. Ratings are scoped
to the caller's own requests: they become training data, so being able to rate
someone else's traffic is being able to steer their router.

**No label is ever invented.** It would be easy to guess — "the user asked
again thirty seconds later, so the first answer was probably bad." It is
cheap, it is clever, and it is wrong often enough to matter. A wrong label does
not raise an error; it quietly teaches the router something false and every
decision afterwards is built on it. Same rule as refusing to guess a model's
price: a made-up label is worse than no label.

### It refuses to train on too little

```
Not enough rated traffic to train a router.
  - Only 1 model(s) have enough rated traffic; 2 are needed.
    qwen2.5:7b needs 12 more rated requests
```

| Gate | Why |
|---|---|
| prompt storage must be on | otherwise there is no question to learn from |
| 30 rated requests per model | below this a classifier fits noise, then routes real traffic on it |
| 5 of **each** verdict per model | 40 ratings that all say "good" produce a classifier that answers yes to everything and wins every decision |
| 2 models must clear both | a router with one choice is not a router |

`--force` overrides all of it. It exists for inspecting a result, not for
serving traffic with one, and it says so when you use it.

### Why live data is shaped differently

Benchmark data is **dense** — every question answered by every model. Live
traffic is **sparse** — each request was answered by exactly one model, and
what the others would have said is unknown.

That suits this design, because there was never one model comparing all the
options: there is one classifier per model, each learning "will I get this kind
of question right?" from the requests it personally handled. The one thing that
must not happen is flattening sparse data into the dense shape, which would
write a zero wherever a model was not asked and teach every classifier that
every question somebody else handled was one it got wrong.

Held-out requests are split **by distinct prompt**, not by row, so the same
question asked twice cannot land in both halves and turn the AUC into a measure
of memory.

### A bug this found

While building it: a trained router pickles a reference to the module its
classes came from. Those classes lived in `eval/`, and the Docker image
deliberately does not copy `eval/` — it is 500 MB of research tooling a server
never runs.

So **inside a container, no trained router could load at all.** The failure was
caught safely, routing switched itself off, and `/health` reported "no router
artifact loaded" with nothing pointing at the cause. Anyone who deployed with a
trained router was running without routing and had no way to find out.

The classes now live in `switchboard/routing/`, the artifact version went from
1 to 2 so an old file says "retrain it" rather than failing mysteriously, and a
test asserts that everything a trained artifact refers to ships with the server.

## Routing on day one, with no training

A fresh install has no traffic, no ratings and no trained router. Something
still has to choose a model for the very first request.

```powershell
# nothing to train, nothing to configure
python -m switchboard serve
```

Two mechanisms, neither of which predicts anything:

### 1. The ladder — cheapest model that fits

Not a heuristic. It applies only facts:

- the cheapest model on your `ladder`, always
- unless the prompt does not fit its context window — a hard limit, not an opinion
- unless the caller sent a cost or latency cap, which is their decision, not a guess

**It deliberately does not guess from wording.** This project measured that: a
hand-written keyword heuristic scored 77.8% on one benchmark and **57.9% on
another — worse than simply always using the cheapest model.** A rule that helps
on one workload and hurts on another is worse than no rule.

Two further experiments confirmed nothing better is available before the answer
exists. Predicting a question's difficulty from its text does not transfer
between domains: within a suite the correlation was **0.077**, which is zero.
See [RESULTS.md section 8](docs/RESULTS.md).

### 2. Verification — look at what came back

Guessing needs knowledge of the world. Checking needs only the answer, which is
sitting right there:

| Check | Escalates? | Why |
|---|---|---|
| empty response | **yes** | another model may well produce content |
| invalid JSON when JSON was requested | **yes** | stronger models follow formats better |
| truncated at `max_tokens` | no | a stronger model hits the same limit — raise `max_tokens` |
| the model refused | **never** | see below |
| hedging, "I'm not sure" | no | that may be the correct answer |

```
SWITCHBOARD_VERIFY_MODE=off        do nothing
SWITCHBOARD_VERIFY_MODE=flag       check and record, change nothing   (DEFAULT)
SWITCHBOARD_VERIFY_MODE=escalate   also retry on the next model up
```

**Why `flag` is the default.** Escalation makes a second provider call, which
doubles the cost of the requests it touches. Nobody should acquire a larger bill
by installing software and leaving the defaults alone. Run in `flag` mode first,
look at how often checks fire on your own traffic, then decide — the same
argument that keeps blocking off in the usage policy.

### A refusal is never escalated

If a model declines a request, sending it up the ladder until one complies is
**shopping for a yes.** Switchboard records the refusal and passes it through.
There is a test named after this.

### Escalated requests are charged for both calls

A request that escalated made two provider calls and paid for both.
`simulated_cost_usd` holds the sum and `attempts` says how many calls it took.

Charging only for the model that produced the final answer would make this
feature look free, and it is exactly the error the cascade scoring was built to
avoid.

### What it costs

Every request starts at the cheapest model. On the benchmarks, always-cheapest
scored **84.0% against 86.8%** for always using the best model, at **1/20th of
the price** — before any escalation. Escalation recovers some of that gap on the
requests where failure is visible. It cannot recover the rest, and nothing here
pretends otherwise.

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
| 3 | Baseline strategies + live eval harness (first real numbers) | done |
| 4 | Embedding classifier and cascade routing | done |
| 5 | Guardrails, with honest false-positive-rate reporting | done |
| 6 | Offline replay, Pareto plots, dashboard, writeup | done |

## Model tiers

| Tier | Model | Intended for |
|------|-------|--------------|
| T0 | `qwen2.5:1.5b` | Classification, formatting, short factual |
| T1 | `qwen2.5:3b` | Routine generation |
| T2 | `qwen3:4b` | Reasoning-capable (emits `<think>` blocks) |
| T3 | `qwen2.5:7b` | Hard tasks; evicts other models from VRAM |
