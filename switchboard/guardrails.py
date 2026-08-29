"""Usage policy: noticing when a company's AI gateway is being used for
somebody's holiday planning.

WHY THIS EXISTS. Every organisation that puts an LLM gateway in front of its
engineers eventually asks the same question: how much of this bill is actually
work? It is a fair question, and answering it is easy - the ledger already
knows what every request cost. What is hard is answering it *without* becoming
the thing everybody hates.

WHY IT DOES NOT BLOCK BY DEFAULT. The failure modes here are not symmetric.

* A missed personal request costs the company a fraction of a cent.
* A wrongly blocked request stops an engineer doing their job, at the moment
  they are trying to do it, with an error message accusing them of slacking.

The second is enormously worse, and it is also the more likely one: the
detector is a set of regular expressions, and human language is not. So the
default mode is `flag` - the request is served normally, and a label is written
to the ledger. An operator gets their report; nobody gets blocked by a regex.
`block` mode exists for organisations that genuinely need it, and it comes with
a documented override so a false positive is a five-second annoyance rather
than a support ticket.

WHY IT IS BIASED TOWARDS LETTING THINGS THROUGH. Work signals (code fences,
stack traces, SQL, file paths) SUBTRACT from the score. A prompt that looks
even slightly technical needs a much stronger personal signal before it is
flagged. This deliberately raises the miss rate to lower the false-alarm rate,
because of the asymmetry above.

WHAT IT STORES. The label and the names of the rules that matched - never the
prompt text. A feature built to police what people type must not become the
reason a company starts recording what people type. Prompt storage stays
behind SWITCHBOARD_STORE_PROMPTS, off by default, exactly as before.

HOW HONEST IT IS. Run `switchboard guardrails calibrate` and it reports its own
false-positive rate on a labelled sample, including the specific prompts it got
wrong. A detector that will not report its error rate should not be trusted
with anyone's work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Modes for SWITCHBOARD_GUARDRAILS_MODE.
MODE_OFF = "off"
MODE_FLAG = "flag"
MODE_BLOCK = "block"
MODES = (MODE_OFF, MODE_FLAG, MODE_BLOCK)

#: Actions recorded in the ledger.
ACTION_ALLOWED = "allowed"
ACTION_FLAGGED = "flagged"
ACTION_BLOCKED = "blocked"
ACTION_OVERRIDDEN = "overridden"

#: Header a caller sends to push a request through in `block` mode. It is a
#: speed bump, not a security control, and it is deliberately not secret: the
#: point is to make a false positive recoverable in seconds while leaving a
#: record that someone overrode the policy, and why.
OVERRIDE_HEADER = "x-switchboard-policy-override"

#: A prompt scoring at or above this is flagged. Weights are set so that one
#: unambiguous phrase reaches it on its own.
DEFAULT_THRESHOLD = 1.0

#: The most a prompt's technical content can discount its score. Without a cap,
#: pasting one stack trace would excuse everything that followed it.
MAX_WORK_DISCOUNT = 1.0

#: Only the first N characters are scanned. A 200 KB pasted log file is not
#: made more or less personal by its last page, and running a regex over all of
#: it on every request would put the detector in the hot path of a proxy.
SCAN_LIMIT = 4000


@dataclass(frozen=True)
class Rule:
    """One pattern, and how much it counts for.

    `weight` is the point of the design. 1.0 means "this phrase alone is
    enough", and is reserved for wordings that essentially never appear in
    engineering work. 0.5 means "suspicious, but I have seen it in a real
    ticket" - two of those are needed before anything happens.
    """

    name: str
    label: str
    pattern: str
    weight: float = 1.0

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


@dataclass
class Verdict:
    """What the policy thinks, and why it thinks it."""

    label: str | None = None
    score: float = 0.0
    matched: tuple[str, ...] = ()
    work_signals: tuple[str, ...] = ()
    action: str = ACTION_ALLOWED
    override_reason: str | None = None

    @property
    def flagged(self) -> bool:
        return self.label is not None

    @property
    def blocked(self) -> bool:
        return self.action == ACTION_BLOCKED

    def explain(self) -> str:
        if not self.flagged:
            parts = []
            if self.matched:
                # A near miss is worth saying out loud: it is how someone
                # decides whether the threshold is set where they want it.
                parts.append(
                    f"Matched {', '.join(self.matched)}, but scored only "
                    f"{self.score:.1f} - below the threshold."
                )
            if self.work_signals:
                parts.append(
                    "Looks like work. Technical content found: "
                    + ", ".join(self.work_signals)
                )
            return " ".join(parts) or "Nothing matched."
        why = ", ".join(self.matched) or "no rule"
        note = (
            f"; offset by technical content ({', '.join(self.work_signals)})"
            if self.work_signals
            else ""
        )
        return f"{self.label} (score {self.score:.1f}) - matched {why}{note}"


# --- The shipped rule set ---------------------------------------------------
#
# Phrase-level, not word-level, and that is the whole trick. "trip" appears in
# "round trip latency"; "plan my holiday" does not appear in anything else.
# Every weight-1.0 rule below was checked against one question: could a
# developer plausibly write this sentence about their job?

BUILTIN_RULES: tuple[Rule, ...] = (
    # --- Travel and leisure -------------------------------------------------
    Rule(
        "holiday_planning",
        "personal",
        r"\b(plan|planning|book|booking)\b[^.?!]{0,30}\b(my|our|a)\b"
        r"[^.?!]{0,20}\b(holiday|vacation|honeymoon|getaway|road ?trip|"
        r"city break)\b",
    ),
    Rule(
        "travel_booking",
        "personal",
        r"\b(book|find|cheapest)\b[^.?!]{0,25}\b"
        r"(flight|flights|hotel|airbnb|hostel|train ticket)\b",
    ),
    Rule(
        "itinerary",
        "personal",
        r"\b\d+[- ]day itinerary\b|\bitinerary for (my|our|a) (trip|visit)\b",
    ),
    # --- Relationships ------------------------------------------------------
    Rule(
        "relationships",
        "personal",
        r"\bmy (girlfriend|boyfriend|wife|husband|ex|crush)\b"
        r"|\b(marriage|divorce|breakup|break[- ]up) advice\b",
    ),
    Rule(
        "romantic_writing",
        "personal",
        r"\b(write|draft|compose) (me )?(a|an) (love letter|"
        r"poem for (my|our|her|his)|birthday (card|wish|message))"
        r"|\bwedding (speech|vows|toast)\b",
    ),
    # --- Health, money, life admin -----------------------------------------
    Rule(
        "personal_health",
        "personal",
        r"\b(diet|meal|workout|gym|training) plan for me\b"
        r"|\bi (have|think i have) (a )?(headache|fever|rash|cold|flu)\b"
        r"|\bshould i see a doctor\b",
    ),
    Rule(
        "personal_finance",
        "personal",
        r"\bshould i (buy|invest in|sell)\b[^.?!]{0,25}\b"
        r"(stock|stocks|shares|crypto|bitcoin|property|house|car)\b"
        r"|\bmy (mortgage|tax return|credit score|salary negotiation)\b",
    ),
    Rule(
        "horoscope",
        "personal",
        r"\b(horoscope|astrology|zodiac|tarot|numerology|birth chart)\b",
    ),
    # --- Homework -----------------------------------------------------------
    Rule(
        "academic_ghostwriting",
        "personal",
        r"\bwrite my (essay|assignment|homework|thesis|dissertation)\b"
        r"|\b(essay|assignment) (due|for class) (tomorrow|monday|tonight)\b"
        r"|\bmy (college|university) application essay\b",
    ),
    # --- Entertainment ------------------------------------------------------
    Rule(
        "entertainment",
        "personal",
        r"\b(what should i watch|movie recommendations?|netflix)\b"
        r"|\b(fantasy (football|cricket)|match score|world cup final)\b",
    ),
    Rule(
        "cooking",
        "personal",
        r"\brecipe for (dinner|lunch|a cake|chicken|pasta)\b"
        r"|\bwhat (should|can) i (cook|make) for (dinner|lunch)\b",
    ),
    # --- Half weight: real signals that also occur in real work -------------
    Rule(
        "family",
        "personal",
        r"\bmy (mum|mom|dad|parents|sister|brother|son|daughter|kids)\b",
        0.5,
    ),
    Rule(
        "occasion",
        "personal",
        r"\bfor (my|our) (birthday|anniversary|wedding|graduation)\b",
        0.5,
    ),
    Rule(
        "shopping",
        "personal",
        r"\b(which|what) (phone|laptop|car|tv) should i buy\b",
        0.5,
    ),
    Rule(
        "casual_chat",
        "personal",
        r"\b(tell me a joke|are you conscious|what do you think of me)\b",
        0.5,
    ),
)

#: Signals that this is work. Each subtracts 0.5, up to MAX_WORK_DISCOUNT.
#: Kept broad on purpose - the cost of over-applying these is a missed flag,
#: which is the cheap mistake.
WORK_SIGNALS: tuple[tuple[str, str], ...] = (
    ("code block", r"```"),
    ("code syntax", r"\b(def |class |function |import |const |public static)"),
    ("sql", r"\b(select .+ from|insert into|create table|join .+ on)\b"),
    ("stack trace", r"\b(traceback|stack trace|exception|segfault|nullpointer)\b"),
    ("file path", r"[\w/\\.-]+\.(py|js|ts|tsx|go|rs|java|rb|sql|ya?ml|json|tf)\b"),
    ("infrastructure", r"\b(kubernetes|docker|terraform|nginx|ci/cd|pipeline)\b"),
    (
        "engineering",
        r"\b(unit test|pull request|merge conflict|refactor|api endpoint|"
        r"regex|migration|latency|throughput)\b",
    ),
)


class Guardrails:
    """Scores prompts against a rule set. Holds no state between requests."""

    def __init__(
        self,
        mode: str = MODE_FLAG,
        rules: tuple[Rule, ...] = BUILTIN_RULES,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.threshold = threshold
        self.rules = rules
        self._compiled = [(rule, rule.compiled()) for rule in rules]
        self._work = [
            (name, re.compile(pattern, re.IGNORECASE))
            for name, pattern in WORK_SIGNALS
        ]

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    def score(self, text: str) -> Verdict:
        """Score one prompt. Decides nothing - see `evaluate`."""
        if not self.enabled or not text:
            return Verdict()

        text = text[:SCAN_LIMIT]

        matched: list[str] = []
        labels: dict[str, float] = {}
        raw = 0.0
        for rule, pattern in self._compiled:
            if pattern.search(text):
                matched.append(rule.name)
                raw += rule.weight
                labels[rule.label] = labels.get(rule.label, 0.0) + rule.weight

        signals = [
            name for name, pattern in self._work if pattern.search(text)
        ]
        discount = min(MAX_WORK_DISCOUNT, 0.5 * len(signals))
        score = max(0.0, raw - discount)

        if score < self.threshold:
            # Near misses still carry their evidence, so `guardrails check` can
            # explain one. That is what makes the threshold tunable.
            return Verdict(
                score=score, matched=tuple(matched), work_signals=tuple(signals)
            )

        label = max(labels, key=lambda key: labels[key])
        return Verdict(
            label=label,
            score=score,
            matched=tuple(matched),
            work_signals=tuple(signals),
            action=ACTION_FLAGGED,
        )

    def evaluate(self, text: str, override: str | None = None) -> Verdict:
        """Score, then decide what to do about it.

        `override` is the value of the override header. In `block` mode a
        non-empty value serves the request anyway and records that it was
        overridden, with the reason given. In `flag` mode there is nothing to
        override, so it is ignored.
        """
        verdict = self.score(text)
        if not verdict.flagged or self.mode != MODE_BLOCK:
            return verdict

        if override:
            verdict.action = ACTION_OVERRIDDEN
            verdict.override_reason = override[:200]
            return verdict

        verdict.action = ACTION_BLOCKED
        return verdict

    def refusal(self, verdict: Verdict) -> str:
        """The message a blocked caller sees.

        Says what matched, says it might be wrong, and says how to get past it.
        An error that does none of those turns a regex false positive into a
        support ticket and a grudge.
        """
        return (
            "This request was held by your organisation's usage policy "
            f"(category: {verdict.label}; matched: "
            f"{', '.join(verdict.matched) or 'no rule'}). This check is a "
            "keyword match and it does get things wrong. If this is work, "
            f"resend it with the header '{OVERRIDE_HEADER}: <short reason>' "
            "and it will go through."
        )

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rules": len(self.rules),
            "threshold": self.threshold,
            "blocking": self.mode == MODE_BLOCK,
        }


# --- Reading a prompt -------------------------------------------------------


def prompt_text(messages: Any) -> str:
    """Flatten a `messages` array to the text the person actually wrote.

    System messages are skipped. They are written by the application, not the
    person, so scoring them would flag every user of a product whose system
    prompt happens to mention holidays.
    """
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # OpenAI's multimodal shape: a list of {type, text} parts.
            parts.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return "\n".join(parts)


# --- Loading ----------------------------------------------------------------


def load_rules(path: str | Path) -> tuple[Rule, ...]:
    """Read a rule set from YAML, REPLACING the built-ins.

    Replacing rather than merging, because a merge makes it impossible to
    remove a shipped rule that keeps flagging your team's actual work.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("rules")
    if not entries:
        raise ValueError(f"{path}: no `rules:` list found.")

    rules = []
    for index, entry in enumerate(entries):
        try:
            rule = Rule(
                name=str(entry["name"]),
                label=str(entry.get("label", "personal")),
                pattern=str(entry["pattern"]),
                weight=float(entry.get("weight", 1.0)),
            )
            rule.compiled()  # fail here, not on the first request
        except (KeyError, TypeError, re.error) as exc:
            raise ValueError(f"{path}: rule {index} is invalid: {exc}") from exc
        rules.append(rule)
    return tuple(rules)


def build_guardrails(mode: str, rules_file: str | None = None) -> Guardrails:
    """Construct from settings. A bad rules file is fatal at STARTUP.

    Deliberately not tolerated the way a missing router artifact is. A router
    that fails to load costs you routing and says so; a policy file that failed
    to load quietly would leave a policy an operator believes is running
    switched off.
    """
    rules = load_rules(rules_file) if rules_file else BUILTIN_RULES
    return Guardrails(mode=mode, rules=rules)


# --- Calibration ------------------------------------------------------------


@dataclass
class Calibration:
    """How often the detector is right, on prompts with known answers."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    false_positive_examples: list[str] = field(default_factory=list)
    false_negative_examples: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def false_positive_rate(self) -> float:
        """Share of genuine WORK prompts that got flagged.

        The number that matters. It is the rate at which this thing gets in
        somebody's way while they are trying to do their job.
        """
        work = self.true_negative + self.false_positive
        return self.false_positive / work if work else 0.0

    @property
    def recall(self) -> float:
        """Share of genuine personal prompts that were caught."""
        personal = self.true_positive + self.false_negative
        return self.true_positive / personal if personal else 0.0

    @property
    def precision(self) -> float:
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else 0.0

    def describe(self) -> str:
        work = self.true_negative + self.false_positive
        return (
            f"{self.total} labelled prompts: caught {self.recall:.0%} of the "
            f"personal ones, and wrongly flagged "
            f"{self.false_positive_rate:.1%} of the work ones "
            f"({self.false_positive} of {work})."
        )


def calibrate(guardrails: Guardrails, samples) -> Calibration:
    """Score labelled samples. `samples` yields (text, is_personal) pairs."""
    result = Calibration()
    for text, is_personal in samples:
        flagged = guardrails.score(text).flagged
        if is_personal and flagged:
            result.true_positive += 1
        elif is_personal:
            result.false_negative += 1
            result.false_negative_examples.append(text)
        elif flagged:
            result.false_positive += 1
            result.false_positive_examples.append(text)
        else:
            result.true_negative += 1
    return result


def load_samples(path: str | Path) -> list[tuple[str, bool]]:
    """Read the labelled sample file: one JSON object per line."""
    import json

    samples: list[tuple[str, bool]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        samples.append((row["text"], row["label"] == "personal"))
    return samples
