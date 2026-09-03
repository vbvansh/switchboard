# What was measured

Every number in this file was produced by a command in this repository, against
public datasets of recorded model answers. Nothing here is an estimate, a
projection, or a vendor's claim.

Each section also says what the number **does not** prove. That part is the
point of the document.

---

## How the measurement works

The two benchmark datasets contain hundreds of thousands of **answers real
commercial models actually gave**, with the price of each call and, for one
dataset, its latency. Normalising them gives a grid:

```
                  gpt-5      claude-sonnet-4    gemini-2.5-pro   ...
question 1     correct        wrong              correct
               $0.0141        $0.0038            $0.0192
question 2     wrong          correct            correct
               ...
```

Once that grid exists, a routing strategy is just a rule for choosing a column
per row. Its accuracy and its cost can be computed exactly, offline, with no API
calls and no spend. That is what made it possible to evaluate routing across
GPT-5, Claude and Gemini on a zero budget.

| | LLMRouterBench | xRouteBench |
|---|---|---|
| Models | 40, including GPT-5, Claude, Gemini | 18 open-weight |
| Rows | 548,059 | 147,906 |
| Suites | 27 (GPQA, HLE, SWE-bench, …) | 13 (MMLU, GSM8K, MBPP, …) |
| Per-query latency | no | yes |

Held-out splits are **by question**, so the same question never appears in both
training and test under different models.

---

## 1. There is real headroom

The "oracle" is a router that always picks correctly. It is impossible — it
would need to know the answer already — and it exists to mark the ceiling.

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

**What it proves:** the best single model is not the best possible answer. Price
does not predict quality — on GPQA, Gemini 2.5 Pro and GPT-5 score identically
while Gemini costs twice as much; on xRouteBench the cheapest model beats one
costing 24× more.

**What it does not prove:** that any achievable router gets near this. The
oracle is a ceiling, not a target.

---

## 2. Hand-written heuristics are unreliable

| Strategy | GPQA | xRouteBench |
|---|---|---|
| always-cheapest | 58.6% | 68.6% |
| keyword heuristic | 77.8% | **57.9%** |

The same keyword heuristic that helps on one workload actively **hurts** on the
other — worse than simply always using the cheapest model.

**What it proves:** a heuristic that is better on one workload and worse on
another is worse than a consistent baseline, because you cannot tell in advance
which case you are in.

**Why this mattered:** it is the finding that justified building a learned
router instead of shipping a rule of thumb. It was also the strongest argument
for running baselines first — without the second dataset, the heuristic looked
like a success.

---

## 3. The learned router beats the best single model

Per model, a classifier predicts the probability that this model answers this
question correctly. Route to the cheapest model clearing a confidence threshold.
Sweeping the threshold traces a whole cost/quality curve from one trained model.

**MMLU-Pro, 6 flagship models, 1,200 held-out questions:**

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

**The headline: 88.3% accuracy at $6.42 versus 86.8% at $15.07.** More accurate
*and* 57% cheaper than the best single model money can buy.

Prediction quality, reported as held-out AUC per model (0.5 = guessing):
mean **0.745** on MMLU-Pro, **0.800** on xRouteBench.

**What it does not prove:** that this holds everywhere. On xRouteBench the
learned router dominates `random` and `keyword` and sits on the trade-off curve
throughout, but does **not** beat always-best on accuracy. One decisive win, one
partial. Reported as measured.

---

## 4. Cascades depend on price spread

A cascade calls a cheap model first, looks at the answer, and escalates only if
unconvinced. Scoring charges for **every call made** — charging only the final
model is the easiest way to make a cascade look better than it is.

| | MMLU-Pro (6 flagships) | xRouteBench (18 open models) |
|---|---|---|
| Best learned router | **88.3%** @ $6.42 | 74.8% @ $0.33 |
| Best cascade | 87.5% @ $7.96 | **77.4%** @ $0.89 |
| Cascades on the curve | **0 of 6** | **5 of 6** |

**What it proves:** when the escalation target is far more expensive than the
alternatives, paying twice rarely pays off. On MMLU-Pro, escalating to a $15
model even a third of the time costs more than spreading across six. On
xRouteBench, where prices are close, the better information wins and cascades
extend the frontier past where the learned router tops out.

The lesson is about price spread, not about cascades being good or bad.

---

## 5. A speed promise has a price

**xRouteBench, 2,466 held-out questions, 18 models:**

| SLA | Accuracy | Cost | p95 latency | Violations | Models eligible |
|---|---|---|---|---|---|
| no SLA | 71.8% | $0.120 | 6.68s | — | 18 |
| ≤ 2s | — | — | — | — | **impossible** |
| ≤ 4s | 55.6% | $0.071 | 3.61s | 3.9% | 2 |
| ≤ 6s | 62.5% | $0.253 | 5.06s | 1.8% | 5 |
| ≤ 10s | 71.9% | $0.129 | 6.55s | 1.9% | 14 |

**What it proves:** the tightest achievable promise costs **16.1 percentage
points of accuracy**. No model in this pool can promise 2 seconds at all — which
is itself an answer: add a faster model or loosen the budget.

Two rules keep this honest. Eligibility uses **p95, not the median** — the first
version selected on median latency and produced a "fast" model set whose p95 was
*worse* than routing with no SLA. And violations are counted against **recorded
per-request latency**, not against the averages the router used to decide.
Picking a fast model is not the same as being fast.

---

## 6. The router does not transfer to short chat prompts

```
trained-shaped prompts (709 chars avg):  p spans 0.02-0.88, 38% below threshold
short chat prompts      (34 chars avg):  p clusters 0.67-0.87, no discrimination
```

Measured on its own distribution the classifier works (held-out AUC 0.800).
Shown a 34-character question unlike anything it was trained on, it returns
roughly the same probability for every model, and everything goes to the
cheapest one.

This is **distribution shift**, not a broken model. It is reported here rather
than quietly omitted because it only became visible after wiring the router to
live traffic — which is the argument for doing that instead of trusting the
offline table.

**The fix**, and why shadow mode exists: train on the traffic you actually
serve. Shadow mode collects exactly that, with the router's opinion attached and
nothing at risk.

**That fix now exists** — `switchboard router train-live`, fed by ratings your
application sends to `POST /v1/feedback`. What is NOT here is a measurement of
it. Nobody has run Switchboard on real traffic for long enough to train one, so
there is no held-out AUC to report and no before/after comparison. The mechanism
is built and tested; its effect is unmeasured, and this document will say so
until somebody measures it.

---

## 7. The usage policy, and how often it is wrong

`switchboard guardrails calibrate`, on the 70 labelled prompts shipped in
`switchboard/guardrail_samples.jsonl`:

| Measure | Value |
|---|---|
| **False-positive rate** (work prompts wrongly flagged) | **0.0%** (0 of 45) |
| Personal prompts caught (recall) | 84.0% (21 of 25) |
| Precision | 100% |

The four misses, printed by the tool itself:

```
Help me pick a birthday gift for my mum
Which phone should I buy under 30000 rupees?
Tell me a joke about cats
My son has a school project on volcanoes, help him write it
```

Each is a single half-weight rule match that did not reach the threshold. That
is the design working as intended: misses cost a fraction of a cent, false
alarms cost somebody their afternoon.

**What this does not prove — and this matters more than the numbers.** Those 70
prompts were written by hand by this project's author. A detector measured on
prompts written by the person who wrote its rules will always look better than
it is. Treat this as a smoke test, not as evidence about your team. The real
calibration set is your own traffic: run in `flag` mode for a week, export the
flagged requests, and label them yourself.

The false-positive rate is pinned by a test, so a future rule cannot quietly
raise it.

---

## 8. A negative result: difficulty does not transfer across domains

The idea being tested: difficulty is a property of the QUESTION, not of any
model, so it could be measured once on public data and shipped inside the
package — giving every user a working router on install, with no traffic and no
training. That would remove the cold start entirely.

**It does not work, and the reason is specific.**

```powershell
python -m switchboard bench difficulty all --holdout 8
```

35,420 questions across 40 suites. Difficulty is the fraction of models that
got a question wrong. Eight whole suites were held back — not a random sample of
questions, because a model that has seen GPQA can pattern-match GPQA.

| Measure | Value |
|---|---|
| Correlation across all held-out suites mixed | 0.309 |
| **Correlation WITHIN each held-out suite (median)** | **0.023** |
| Absolute error vs always guessing the mean | −1.1% (worse) |

Per held-out suite, the within-suite correlation:

| Suite | Questions | Correlation |
|---|---|---|
| xroutebench/math | 950 | 0.337 |
| llmrouterbench/swe-bench | 500 | 0.200 |
| llmrouterbench/simpleqa | 4,826 | 0.053 |
| llmrouterbench/arcc | 1,172 | 0.042 |
| llmrouterbench/arenahard_coding | 253 | 0.005 |
| llmrouterbench/arenahard | 750 | 0.005 |
| xroutebench/aime_2020_2024 | 120 | 0.002 |
| llmrouterbench/hle | 2,658 | −0.001 |

**What went wrong.** The overall figure of 0.309 looks like a weak but real
signal. It is not. Mixing every held-out suite together rewards a model for
recognising *which suite* a question came from — AIME questions are hard, ARC
questions are easy — which is vocabulary matching, not difficulty estimation.
Inside any single suite the correlation is 0.023: no ability to tell one
question from another.

**Why that kills the idea.** A real user's traffic is one "suite" — their own
workload. Distinguishing their requests from somebody else's is worth nothing.
Distinguishing their easy requests from their hard ones is the entire job, and
that is exactly the part that scores zero.

**A reporting bug this exposed.** The first version of this report judged
success on absolute error against a constant baseline, and printed rows like
"correlation −0.048, beats guessing +25%". Both numbers were correct and the
conclusion drawn from them was not: a model can sit closer to a suite's average
while ranking every question inside it backwards. The report now measures
within-suite ranking and says so.

**What this saved.** Roughly six phases of work building a shipped prior on a
foundation that does not hold. The experiment took an afternoon.

---

## What was NOT measured

Stated plainly, because a results document that only lists wins is marketing.

- **No real API spend.** Every cost figure is either from recorded benchmark
  prices or simulated: local models are priced as the commercial models they
  stand in for. No live commercial model was ever called by this project.
- **No production traffic.** Every routing result is from public benchmarks.
  Benchmark questions are longer, harder and more uniform than real chat
  traffic — see section 6 for what that costs.
- **No quality measurement in shadow mode.** The shadow model was never called,
  so there is no answer to grade. Shadow savings are projections and are
  labelled that way in the CLI, the dashboard and the code.
- **No human evaluation.** Correctness comes from the benchmark answer keys.
  There is no LLM judge anywhere in this project, deliberately: grading with a
  model would make the results depend on a model.
- **Difficulty estimated from prompt text alone.** Section 8 measured it and
  it does not transfer between domains. The cold-start prior that would have
  been built on it is not being built.
- **No router has yet been trained on real traffic.** The pipeline is built,
  gated and tested end to end, but every routing number in this document still
  comes from public benchmarks. Whether training on live traffic actually fixes
  the short-prompt failure in section 6 is untested.
- **The Anthropic and Gemini adapters have never been run against a live API.**
  They are written to the published specifications and covered by tests using
  recorded response shapes, which proves the translation matches the documented
  format. It does not prove the format is still current. Only a call with a real
  key does that, and nobody has made one.

---

## Reproducing all of it

```powershell
python scripts/fetch_llmrouterbench.py --extract   # ~1.3 GB
python -m switchboard bench build all
python -m switchboard bench headroom llmrouterbench --suite gpqa
python -m switchboard bench replay xroutebench
python -m switchboard bench train llmrouterbench --suite mmlupro --models "gpt-5,claude-sonnet-4,gemini-2.5-pro,gemini-2.5-flash,kimi-k2-0905,qwen3-235b-a22b-2507"
python -m switchboard bench sla xroutebench --budgets "2.0,4.0,6.0,10.0"
python -m switchboard guardrails calibrate
```

The datasets are **not** redistributed here. Neither declares a licence and both
derive from many upstream benchmarks with mixed terms, plus outputs from
commercial models. This repository ships the download script and the analysis
code. Please cite the original papers if you use them.
