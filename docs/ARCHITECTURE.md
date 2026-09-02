# How Switchboard is put together

This document explains the whole system in plain English: what each piece does,
why it exists, and what happens to one request from the moment it arrives.

If you only read one section, read [The request lifecycle](#the-request-lifecycle).

---

## The one-sentence version

Switchboard is a proxy that speaks the OpenAI API. Your application points at
it instead of at OpenAI; Switchboard decides which model should actually answer,
sends the request there, and writes down what happened.

---

## The shape of the system

```mermaid
flowchart TB
    subgraph client [Your application]
      APP["any OpenAI client<br/>base_url = switchboard"]
    end

    subgraph sb [Switchboard]
      SITE["site.py<br/>landing page at /"]
      API["api.py<br/>OpenAI-compatible HTTP"]
      GUARD["guardrails.py<br/>usage policy"]
      ROUTE["routing/<br/>which model?"]
      CACHE["cache.py<br/>seen this before?"]
      POOL["providers/<br/>pool, retries, failover"]
      LEDGER["ledger/<br/>who, what, how much"]
      DASH["dashboard.py + metrics.py<br/>reporting"]
    end

    subgraph up [Model providers]
      OLL["Ollama / vLLM<br/>local"]
      OAI["OpenAI format<br/>OpenAI, Groq, OpenRouter, ..."]
      NAT["Anthropic, Gemini<br/>translated natively"]
    end

    DB[("SQLite or<br/>PostgreSQL")]

    APP -->|POST /v1/chat/completions| API
    API --> GUARD
    API --> ROUTE
    API --> CACHE
    API --> POOL
    POOL --> OLL
    POOL --> OAI
    POOL --> NAT
    API --> LEDGER
    LEDGER --> DB
    DASH --> DB
```

Nothing above is a microservice. It is one Python process. The boxes are
modules, and they are separate modules because each one is a decision that can
be got wrong independently.

---

## The request lifecycle

This is the sequence every chat completion goes through. The order is chosen so
that **every possible refusal happens before any money is spent.**

```mermaid
sequenceDiagram
    participant C as Client
    participant A as api.py
    participant L as Ledger
    participant G as Guardrails
    participant R as Router
    participant P as Provider

    C->>A: POST /v1/chat/completions
    A->>L: who is this API key?
    L-->>A: user, or 401
    A->>A: rate limit for this user? (429)
    A->>R: model "auto" - which one?
    R-->>A: model + a reason
    A->>G: does this prompt look like work?
    G-->>A: verdict (flag, or 403 in block mode)
    A->>L: is this user within budget? (402)
    A->>A: cache hit? -> return, cost 0
    A->>P: forward the request
    P-->>A: answer (retry, then fail over on error)
    A->>L: record user, model, tokens, cost, latency, reason
    A-->>C: the answer, unchanged
```

Every one of the numbered failures is a normal OpenAI-shaped error body, so an
existing client surfaces it properly instead of crashing on an unexpected shape.

---

## The pieces, and why each exists

### `api.py` — the HTTP surface

Implements `POST /v1/chat/completions`, `GET /v1/models`, the health endpoints,
`/metrics` and `/dashboard`.

The important design decision: **the request body is not schema-validated.** It
is passed through to the provider as it arrived, and only the `model` field is
ever rewritten. Validating it would mean modelling every OpenAI feature — tools,
response formats, log probabilities, whatever ships next month — and any feature
not modelled would break. Passing it through means Switchboard stays compatible
with things it has never heard of.

Streaming is a genuine pass-through. The bytes go straight to the client while a
sniffer reads the token usage out of the stream as it passes, so accounting
happens without buffering the answer or delaying the first token.

### `catalog.py` — what models exist

Reads `providers.yaml`. Every model has a provider, a price per million tokens
in and out, a tier, and optionally a `benchmark_alias`.

Two decisions here matter:

**API keys are never in the config.** `providers.yaml` names an environment
variable (`api_key_env: OPENAI_API_KEY`); the key itself lives in `.env`, which
is gitignored. The config file is safe to commit, which is the only way a config
file ever stays correct.

**Two providers may declare the same model.** That is not a conflict — it is
failover. If Groq is down, the same model is served from OpenRouter.

A provider's `type` selects its adapter. Three exist: `openai-compatible`,
`anthropic`, and `gemini`.

### `routing/` — which model should answer

Three layers:

- `base.py` — the interface. A strategy takes a `RoutingContext` and returns a
  `RoutingDecision` with a model *and a reason*. The reason is mandatory because
  a routing decision you cannot explain afterwards is a routing decision you
  cannot debug.
- `baselines.py` — always-cheapest, always-best, random, keyword heuristic.
  These exist to be beaten. A clever router with no baselines beside it is an
  unsupported claim.
- `live.py` — loads a trained artifact and drives it in the live API, applying
  per-request limits sent as headers.
- `features.py`, `predictor.py` — turning a question into numbers, and one
  classifier per model. **These live here, not in `eval/`, and that is
  load-bearing.** A joblib pickle records the module each class came from,
  and the Docker image does not copy `eval/`. While they lived there, no
  trained router could load inside a container: the failure was caught
  safely, routing switched itself off, and `/health` said "no router
  artifact loaded" with nothing pointing at the cause. Anything a trained
  artifact refers to has to ship with the server.

A missing, corrupt, or stale router artifact **never stops the server.** Routing
switches off, requests fall back to `default_model`, and `/health` says exactly
why. A routing feature that can take the gateway down is worse than no routing.

### `guardrails.py` — the usage policy

Notices when the company gateway is being used to plan somebody's holiday.

The design is shaped by one asymmetry: missing a personal request costs a
fraction of a cent, while wrongly blocking a real one stops an engineer working
at the moment they are trying to work. So:

- The default mode is **`flag`**, which refuses nothing. It writes a label to
  the ledger and serves the request normally.
- Technical content in the prompt (code fences, stack traces, SQL, file paths)
  **subtracts** from the score, so anything that looks like engineering needs a
  much stronger personal signal to trip.
- Rules have weights. A phrase that could only ever be personal counts 1.0 and
  trips on its own; a phrase that shows up in real tickets counts 0.5 and needs
  a second signal.
- `block` mode exists, and it ships with an override header, and the refusal
  message tells you about it. A false positive should cost five seconds, not a
  support ticket.
- Only the verdict and the names of the matched rules are stored — never the
  prompt text. A feature built to police what people type must not become the
  reason a company starts recording what people type.
- `switchboard guardrails calibrate` reports its own false-positive rate,
  including the exact prompts it got wrong.

### `cache.py` — never pay twice for the same question

An in-memory LRU with a TTL. A cache hit costs nothing, and the ledger records
it as costing nothing — a hit billed at full price would inflate every savings
figure in the product.

Only *byte-identical, deterministic* requests hit. Semantic matching is
deliberately not implemented: a cache that occasionally returns a confident
answer to a question nobody asked is worse than no cache.

### `providers/` — talking to models

- `openai_compatible.py` — covers most of the industry, because most of the
  industry copied OpenAI's format.
- `anthropic.py`, `gemini.py` — the two that did not. See below.
- `sse.py` — reassembles streamed events from network chunks that do not line
  up with them.
- `pool.py` — holds the clients, knows which providers can serve a model, and
  enforces `LOCAL_ONLY`.
- `retry.py` — exponential backoff with jitter, for *transient* failures only
  (timeouts, 429, 5xx). A malformed request is never retried: it would fail
  identically and cost twice.
- `breaker.py` — a circuit breaker. After N consecutive failures a provider is
  skipped for a cooldown, then probed. Without this, every request to a dead
  provider waits out its full timeout before failing over.

Failover rule: a provider that raises or returns 5xx is a failure, and the next
provider is tried. **A 4xx is not a failover** — the request itself is wrong and
every other provider would reject it identically, so retrying elsewhere just
multiplies the cost of one bad request by the number of providers configured.

### Adapters — making every provider look the same

Switchboard speaks OpenAI to your application, always, in both directions. Most
providers copied that format, so `openai_compatible.py` covers them by
forwarding requests untouched.

Anthropic and Google did not, so `anthropic.py` and `gemini.py` translate. Four
differences do the damage if missed:

| | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| System prompt | a message in the list | a separate `system` field | `systemInstruction` |
| The assistant | `role: "assistant"` | same | `role: "model"` |
| `max_tokens` | optional | **required** | `maxOutputTokens` |
| Token counts | `prompt_tokens` | `input_tokens` | `promptTokenCount` |

The last row is the dangerous one. Getting it wrong does not crash anything — it
records every request to that provider as costing zero, and the savings report
looks wonderful. There is a test named after exactly that failure.

Streaming is translated too, and the rebuilt stream ends with a usage chunk in
the shape `streaming.py` already reads, so the ledger needs no special case per
vendor.

**Tool calls are deliberately not translated.** Their formats differ in more
than naming, and a half-working translation fails deep inside somebody's agent
with a confusing error. A request carrying `tools` is refused with a message
pointing at OpenRouter, which implements them.

### `discovery.py` — asking a provider what it has

Format translation makes a provider *callable*. Discovery is what makes it
*bearable*: hand-typing three hundred model names and prices is not a plan.

`switchboard discover <provider>` asks, parses the answer, and prints YAML.

It refuses to invent a price. Only OpenRouter publishes prices through its API;
everyone else publishes a bare list. Those models come back marked `REPLACE ME`
and will not load until a human fills them in. A guessed price would flow
straight into budget enforcement and savings reports and be wrong invisibly.

It also does not rewrite `providers.yaml`. That file's comments explain why each
model is priced as it is, and no automatic rewriter preserves them.

### `site.py` — the public landing page

Served at `/` by the same process as the API, so there is one deploy, one URL,
and no separate marketing site to drift out of date. Same construction rules as
the dashboard: server-rendered, inline CSS, one inline script, nothing external.

The content rule is enforced by a test: every number on the page must also
appear in `docs/RESULTS.md`. A landing page that quietly advertises a figure
nobody can reproduce is the exact failure this project is built against.

### `ledger/` — who spent what

Two tables. `users` holds names, hashed API keys, budgets and rate limits.
`requests` holds one row per request: user, requested model, served model,
tokens, simulated cost, baseline cost, latency, status, routing reason, shadow
decision, and policy verdict.

The **baseline cost** column is what makes the savings claim auditable. Every
row stores what that same request would have cost on the top-tier model, so
"we saved 57%" is a `SELECT`, not a reconstruction from assumptions.

Only the hash of an API key is stored. The raw key exists once, at creation, and
is never recoverable.

Timestamps are naive UTC everywhere, because SQLite does not preserve timezone
information and carrying tz-aware values through it invites comparison bugs that
surface months later.

### `training.py` — learning from your own traffic

The loop shadow mode exists to feed, and the step that was missing between them.

A benchmark ships with an answer key, so training on one is free. Real traffic
has none — nobody wrote down the correct answer to "why is this test flaky" — so
the ledger held thousands of questions and no outcomes. `POST /v1/feedback`
supplies the missing half: every response carries an `X-Switchboard-Request-Id`,
and the application sends back `good` or `bad`.

No label is ever inferred. Guessing from behaviour ("they asked again, so it was
wrong") is cheap and wrong often enough to matter, and a wrong label does not
error — it quietly teaches the router something false.

Training **refuses** below its thresholds: prompt storage on, 30 rated requests
per model, 5 of each verdict, 2 models clearing both. Each exists to stop a
specific failure, and the commonest is one-sided data — forty ratings that all
say "good" would fit a classifier that answers yes to everything and then wins
every routing decision.

Live data is *sparse* (one model per request) where benchmark data is *dense*
(every model per question), so each classifier trains only on the requests its
own model handled. Flattening that into the dense shape would write a zero
wherever a model was not asked.

### `shadow.py` — trying routing without risking anything

Runs the router on every request, records what it *would* have chosen and what
that would have cost — then ignores the decision and serves the request exactly
as it would have been served with no router at all.

After a week of real traffic an operator has a report about their own workload,
having risked nothing. Two limits are stated everywhere the numbers appear: the
shadow cost is an estimate (that model was never called, so its token count does
not exist), and shadow mode cannot say whether quality would have held up (no
answer was produced to grade).

### `dashboard.py` and `metrics.py` — reporting

`/dashboard` is server-rendered HTML with inline CSS. No JavaScript, no build
step, no CDN. It has to work on a machine with no internet — the deployment
`LOCAL_ONLY` exists to support — and nothing loaded from a CDN should get to see
when your engineers look at their AI spend.

`/metrics` is the Prometheus text format, written by hand rather than pulling in
a client library. The rule that matters there is **cardinality**: labels are only
ever drawn from small fixed sets (a status, a provider, a model). Labelling by
user or by prompt would create a new time series per user or per request and
eventually take the monitoring system down.

### `migrations/` — changing the schema without losing data

Alembic, five revisions. The server **refuses to start** against a database
whose schema does not match the code. Starting anyway does not fail cleanly; it
fails later, mid-request, with an error pointing somewhere unhelpful.

---

## The research half

Everything under `eval/` is offline analysis. It is **not** in the Docker image
and is not needed to run the gateway.

```mermaid
flowchart LR
    DL["public benchmark<br/>datasets"] --> BUILD["bench build<br/>normalise"]
    BUILD --> GRID["Grid<br/>question x model:<br/>correct? cost? latency?"]
    GRID --> REPLAY["bench replay<br/>score baselines"]
    GRID --> TRAIN["bench train<br/>per-model classifiers"]
    GRID --> SLA["bench sla<br/>latency budgets"]
    TRAIN --> ART["router.joblib"]
    ART --> LIVE["routing/live.py<br/>in the server"]
```

The whole trick is that these datasets contain **recorded answers from real
commercial models**. Once normalised into a grid of question × model → (correct,
cost, latency), any routing strategy can be scored exactly, offline, with zero
API calls and zero spend. That is what made it possible to evaluate against
GPT-5, Claude and Gemini without a budget.

Held-out splits are by question, never by row, so the same question cannot
appear in training and test under two different models.

---

## Design rules that show up everywhere

**Safe by default, useful on request.** Prompt storage is off. Blocking is off.
`LOCAL_ONLY` exists. Nobody should acquire a legal liability by installing
software and leaving the defaults alone.

**Never flatter the numbers.** Cache hits cost zero. Cascades are charged for
every call they make. Budget-blocked requests cost zero. Shadow projections are
labelled as projections. Each of these is a place where the self-serving version
of the accounting would have looked better.

**A degraded feature must not take the service down.** No router: fall back and
say so. Provider down: fail over. Stale artifact: routing off, `/health`
explains. The one exception is a broken *policy* file, which is fatal at startup
— a policy an operator believes is running must never be silently off.

**Every decision records its reason.** `routing_reason`, `guardrail_rules`,
`X-Switchboard-Cache`, the circuit-breaker snapshot in `/health`. A gateway you
cannot interrogate after the fact is a gateway nobody will trust with production
traffic.

---

## Where the numbers are

See [RESULTS.md](RESULTS.md) for every measured figure, how it was produced, and
what it does not prove.
