# Switchboard

**A router for AI models.** It sits between your application and your models,
sends each request to the cheapest model likely to handle it, enforces
per-developer budgets, and records what everything cost — including what it
would have cost the old way.

Self-hosted. Your prompts go to the providers you configure and nowhere else.

```powershell
pip install switchboard-router
switchboard init
switchboard serve
```

Then change one line in your application:

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-switchboard-...")
```

It routes on the **first request**. Nothing to train, nothing to wait for.

---

## The problem

Models differ in price by a factor of two hundred. Most questions don't need the
expensive one — but nobody can tell which, so in practice everything goes to the
expensive one.

Three things are true, and only the third is hard:

- **Most requests don't need the best model.** "Format this JSON" and "prove
  this theorem" go to the same $15-per-million-tokens model, because that's the
  one everyone configured.
- **Price doesn't predict quality.** Measured: two flagship models score
  identically on one benchmark while one costs twice as much. On another, the
  cheapest model in the pool beats one costing 24× more.
- **Nobody can tell which is which in advance.** That's the actual problem.

## The result

Measured offline against public datasets of answers real models already gave.
MMLU-Pro, 6 flagship models, 1,200 questions the router had never seen:

| Strategy | Accuracy | Cost |
|---|---|---|
| Always the cheapest model | 84.0% | $0.71 |
| **Switchboard** | **88.3%** | **$6.42** |
| Always the best model | 86.8% | $15.07 |

**More accurate *and* 57% cheaper than the best model money can buy.**

Every number in this README is reproducible with a command in this repository.
[docs/RESULTS.md](docs/RESULTS.md) has the methods — and, for each figure, what
it does **not** prove.

---

## Quick start

### 1. Install

```powershell
pip install switchboard-router
```

### 2. Set it up

```powershell
switchboard init
```

Asks which providers you have, **checks each key works before writing
anything**, finds the models behind it, writes the config, sets up the database,
and prints your first API key.

Have a local [Ollama](https://ollama.com)? It finds that too, and then
everything runs on your own machine for free.

### 3. Run it

```powershell
switchboard serve
```

### 4. Point your application at it

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-switchboard-...",        # printed by `switchboard init`
)

response = client.chat.completions.create(
    model="auto",                        # let Switchboard choose
    messages=[{"role": "user", "content": "Why is this test flaky?"}],
)
```

Send `model: "auto"` and Switchboard chooses. Name a model instead and it always
honours that.

### 5. See where the money went

```powershell
switchboard usage
```

Or open `http://localhost:8000/dashboard` in a browser.

---

## How it decides

Two mechanisms, and neither of them guesses.

### Before the call — the cheapest model that fits

```
your own trained router      knows your prompts. Needs traffic first.
        ↓ (if you don't have one)
the shipped router           trained on 40 public benchmark suites
        ↓ (if it has no opinion)
the ladder                   cheapest model that fits. Always available.
```

The ladder applies only facts: cheapest model, unless the prompt doesn't fit its
context window, unless you asked for a cost or speed cap.

**It deliberately sends `"hi"` and `"prove this theorem"` to the same model.**
There's a test asserting that — guessing from wording was measured and found to
be *worse than useless*: a hand-written keyword rule scored 77.8% on one
benchmark and 57.9% on another, which is worse than always using the cheapest
model.

### After the call — check the answer, escalate if it obviously failed

```
empty response                → try the next model up
asked for JSON, got prose     → try the next model up
cut off at the token limit    → record it (a bigger model hits the same limit)
the model refused             → record it and pass it through
"I'm not sure"                → record it (that may be the right answer)
```

Guessing beforehand needs knowledge of the world. Checking afterwards needs only
the answer, which is right there.

**A refusal is never escalated.** Retrying up the ladder until a model complies
is shopping for a yes.

### And it says when it doesn't know

When the router's confidence for every model comes out about the same, it hasn't
distinguished them — so it says so rather than pretending:

```
predictions span only 0.021 across 4 models - no usable discrimination
on this prompt, so no routing decision was made; ladder chose qwen2.5:1.5b
```

---

## What it works with

**Anything speaking OpenAI's format** — most of the industry, because most of
the industry copied it:

> OpenAI, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, Perplexity,
> Cerebras, DeepInfra, OpenRouter — plus every self-hosted server: Ollama, vLLM,
> LM Studio, llama.cpp, TGI.

**Anthropic and Google natively.** They use different formats, so Switchboard
translates in both directions. Your application only ever speaks one format.

**Adding one is editing a text file**, not writing code:

```yaml
  - id: groq
    type: openai-compatible
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY      # the KEY lives in .env, not here
    enabled: true
    models:
      - id: "llama-3.1-8b-instant"
        input_per_mtok: 0.05
        output_per_mtok: 0.08
```

Don't want to type model names by hand?

```powershell
switchboard discover openrouter
```

Asks the provider what it has and prints config to paste. **It will not invent a
price** — OpenRouter publishes real ones; providers that publish none come back
marked `REPLACE ME`. A guessed price would flow straight into your budgets and
be wrong in a way nobody can see.

---

## What else is in it

| | |
|---|---|
| **An auditable ledger** | one row per request. Every row stores what it *would* have cost on your priciest model, so "we saved 57%" is a query, not a claim |
| **Budgets and rate limits** | per developer. Refusals cost nothing, so retrying can't dig a deeper hole |
| **A response cache** | identical requests answered free from memory, recorded as costing zero |
| **Retries and failover** | temporary failures retried; dead providers skipped by a circuit breaker |
| **Shadow mode** | run routing on real traffic, record what it *would* have chosen, then ignore it. Trial it having risked nothing |
| **A usage policy** | flags personal-looking requests without blocking them, and reports its own false-positive rate |
| **A dashboard** | zero JavaScript, zero external requests. Works offline |
| **Prometheus metrics** | at `/metrics`, for monitoring |
| **Schema migrations** | upgrades never destroy your data |
| **Docker** | multi-stage build, runs as a non-root user |

---

## Configuration

Settings go in `.env`. Every one has a safe default; these are the ones people
actually change:

```bash
# Which model answers when no router is available
SWITCHBOARD_DEFAULT_MODEL=qwen2.5:3b

# Requests per minute, per developer
SWITCHBOARD_RATE_LIMIT_PER_MINUTE=60

# Look at answers and act on obvious failures
#   off | flag (default: record only) | escalate (also retry)
SWITCHBOARD_VERIFY_MODE=flag

# Label personal-looking requests
#   off | flag (default) | block
SWITCHBOARD_GUARDRAILS_MODE=flag

# Refuse to start if any provider is off this machine
SWITCHBOARD_LOCAL_ONLY=false

# Password for /dashboard. SET THIS BEFORE DEPLOYING PUBLICLY.
SWITCHBOARD_DASHBOARD_PASSWORD=
```

**Provider API keys also go in `.env`:**

```bash
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
```

`providers.yaml` names the *variable*; the key itself lives in `.env`, which is
never committed. That's what makes the config file safe to share.

```powershell
switchboard where        # which files it's using
switchboard check        # what it can actually serve
switchboard providers    # which providers can see a key
```

---

## Commands

```powershell
switchboard init                   # set up providers and get an API key
switchboard serve                  # run the proxy
switchboard where                  # where your config and data live
switchboard check                  # is everything reachable?
switchboard providers              # models, prices, key status
switchboard discover <provider>    # ask a provider what it offers

switchboard users add alice --budget 25
switchboard usage                  # spend and savings this month

switchboard router info            # what the router can and can't judge
switchboard router data            # can I train on my own traffic yet?
switchboard router train-live      # train from my own ledger

switchboard guardrails calibrate   # how often is the usage policy wrong?
switchboard shadow                 # what would routing have done?
switchboard db upgrade             # apply schema migrations
```

---

## Running it in production

```powershell
docker compose up --build
```

Or deploy to Render, Railway, Fly.io or your own server — see
**[DEPLOY.md](DEPLOY.md)**, which also covers the three settings that silently
break a deployment if you miss them.

**Before any public URL**, set `SWITCHBOARD_DASHBOARD_PASSWORD`. The dashboard
shows spend per developer by name. It shows no prompt text and no keys — but
that's still not for whoever finds the link.

---

## Getting better over time

The shipped router is trained on public benchmarks. A router trained on **your**
traffic will always beat it, and Switchboard collects what's needed.

**1.** Turn on prompt storage — read the warning first, since it means recording
what your users type:

```bash
SWITCHBOARD_STORE_PROMPTS=true
```

**2.** Send a verdict back when someone rates an answer. Every response carries
an `X-Switchboard-Request-Id` header:

```
POST /v1/feedback   {"request_id": "kJ8fQ2...", "rating": "good"}
```

That's what a thumbs up/down button calls.

**3.** Check progress, then train:

```powershell
switchboard router data          # tells you exactly what's still missing
switchboard router train-live
```

It refuses to train on too little and says why. **No label is ever guessed** —
inferring one from behaviour is cheap, wrong often enough to matter, and a wrong
label doesn't error; it quietly teaches the router something false.

---

## What it does *not* do

A README that only lists wins isn't much use. These are real, current, and
measured.

- **The router tells topics apart better than it tells hard from easy.** Across
  40 benchmark suites it scores 0.756 at picking a model by *kind* of question
  and 0.600 at spotting hard ones within a kind. Useful — and a smaller claim
  than the headline.
- **It works on reasoning, code and maths; it has no signal on commonsense,
  emotion, creative writing or factual lookup.** `switchboard router info`
  prints the whole table. Where it has no signal it abstains and the ladder
  decides.
- **Difficulty can't be predicted from prompt text across domains.** Measured
  twice and published as a negative result — see
  [RESULTS.md section 8](docs/RESULTS.md).
- **The shipped router only knows models it was trained on.** Good coverage of
  open-weight models, thin for commercial ones — there's no `gpt-4o-mini`.
  Unknown models fall through to the ladder.
- **Tool calls aren't translated for Claude or Gemini.** Chat and streaming
  work; for tool calling, reach them through OpenRouter.
- **Only one adapter has been verified against a live API** (Groq). Anthropic
  and Gemini are written to published specs and tested against recorded
  responses.
- **No real API spend was measured.** Costs come from recorded benchmark prices
  or are simulated. Local models are priced as the commercial models they stand
  in for, and every screen that shows money says so.

---

## Documentation

| | |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | how it works, with diagrams |
| [Results](docs/RESULTS.md) | every number, its method, and what it doesn't prove |
| [Deploy](DEPLOY.md) | putting it on a public URL |
| [Releasing](RELEASING.md) | publishing to PyPI |

---

## Development

```powershell
git clone https://github.com/vbvansh/switchboard
cd switchboard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

pytest              # 795 tests, no models or network needed
ruff check .
```

The research harness under `eval/` needs the optional extras and the benchmark
datasets:

```powershell
pip install -e ".[research]"
python scripts/fetch_llmrouterbench.py --extract
python -m switchboard bench build all
```

**The benchmark datasets are not redistributed here.** Neither declares a
licence and both derive from many upstream benchmarks with mixed terms, plus
outputs from commercial models. This repository ships the download script and
the analysis code. Please cite the original papers if you use them.

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE). Free to use, modify and redistribute,
commercially included. It carries a patent grant, which is why companies tend to
prefer it over MIT.
