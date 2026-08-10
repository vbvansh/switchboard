# Switchboard

A local-only AI model router. It sits between an application and a pool of
models, decides which model is cheap enough and good enough for each request,
enforces per-user budgets, and blocks requests that shouldn't be served.

Runs entirely on local Ollama models. **No API keys. No paid providers. No
inference request leaves the machine.** That is enforced by a startup check
(`switchboard/config.py`) and covered by tests, not left to convention.

## Status

Milestone 2 of 6 — a metered OpenAI-compatible proxy. **Routing is not
implemented yet**; every request still goes to one fixed model. What works
today:

- `POST /v1/chat/completions`, streaming and non-streaming
- Drop-in compatibility with any OpenAI client via `base_url`
- Per-developer API keys (`401` on a bad key)
- Monthly budgets, enforced (`402` when exhausted)
- Every request logged: who, which model, tokens, simulated cost, latency
- Admin CLI for users, budgets, and usage reports
- Local-provider enforcement

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

## Hardware note

Developed on a 4 GB GPU with 7.3 GB system RAM. That constraint shapes the
design: only about two small models stay resident, so routing to a larger tier
evicts the warm set and the *next* few cheap requests pay a reload penalty too.
The router accounts for warm/cold state rather than pretending model choice is
free. This mirrors GPU-pool locality behaviour in real serving stacks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock.txt

copy .env.example .env
```

Ollama must be running. Confirm it and see the available tiers with prices:

```powershell
python -m switchboard check
```

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

`store_prompts` defaults to **on**: the full `messages` array of every request
is written to the database, because milestone 4's router needs real examples to
learn from. That means the database holds whatever users typed. It is excluded
from git. Set `SWITCHBOARD_STORE_PROMPTS=false` to keep only token counts and
costs.

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
