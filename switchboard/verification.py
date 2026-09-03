"""Looking at an answer and deciding whether it obviously failed.

WHY THIS EXISTS. Every routing method in this project so far has tried to guess,
before calling anything, whether a cheap model would cope. Two experiments say
that guess is not available: difficulty is only visible in a prompt when the
difficulty comes from how much work is required, and most real traffic is not
like that.

So stop guessing. Call the cheap model, then LOOK at what came back.

The difference matters. Predicting needs knowledge we do not have at routing
time - you cannot tell that "what is the capital of Burkina Faso" is harder than
"what is the capital of France" without knowing geography. Checking needs only
the answer, which is sitting right there.

WHAT THIS CAN AND CANNOT SEE. It catches OBVIOUS failure, never wrongness. An
answer that is fluent, confident, well-formed and completely incorrect passes
every check here. That is not a flaw to be fixed later; it is the boundary of
what can be known without a human or a second opinion, and every report says so.

Obvious failure is still worth catching. It is free, it is objective, and it
needs no model of the world.

THE RULE THAT SHAPES EVERYTHING BELOW: a check may only trigger an escalation
if escalating would actually FIX the problem. Three of the checks here
deliberately do not.

    truncated at max_tokens   a stronger model hits the same limit. The fix is
                              to raise max_tokens, so this is reported, not
                              retried - retrying spends money to reproduce the
                              same cut-off answer.

    the model refused         ESCALATION IS NOT APPROPRIATE, and this is the
                              important one. If a model declines a request,
                              sending it to a more capable model until one
                              complies is shopping for a yes. Switchboard will
                              record a refusal and pass it through. It will not
                              go looking for a model that says yes instead.

    hedging language          "I'm not sure" may be the correct answer. Escalating
                              on uncertainty punishes a model for being honest,
                              and would rank a confident wrong answer above a
                              cautious right one. Flagged, off by default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

#: A response with less than this much text is treated as empty. Not zero: some
#: providers return a single space or a stray newline on a failed generation.
MIN_CONTENT_CHARS = 2

#: Only the opening of an answer is scanned for refusals and hedging. A model
#: that is going to decline does it immediately; a mention of "I cannot" four
#: paragraphs into a long correct answer is prose, not a refusal.
OPENING_CHARS = 400

REFUSAL_PATTERNS = re.compile(
    r"^\s*(i (cannot|can't|won't|am unable to|am not able to)\b"
    r"|i'm (sorry|afraid|unable)\b"
    r"|sorry,? (but )?i\b"
    r"|as an ai\b"
    r"|i (do not|don't) (feel comfortable|think i should)\b)",
    re.IGNORECASE,
)

HEDGING_PATTERNS = re.compile(
    r"\b(i (don't|do not) (know|have (enough )?information)"
    r"|i'm not (sure|certain)"
    r"|i (cannot|can't) determine"
    r"|without more (context|information)"
    r"|it'?s (hard|difficult) to say)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Check:
    """One thing that can be wrong with an answer.

    `escalates` is the load-bearing field. It says whether sending the request
    to a stronger model would actually help - not whether the problem is
    serious. A refusal is serious and must NOT escalate; a truncation is
    annoying and escalating reproduces it.
    """

    name: str
    escalates: bool
    explanation: str


CHECKS = {
    "empty_response": Check(
        "empty_response",
        escalates=True,
        explanation="the model returned nothing at all",
    ),
    "invalid_json": Check(
        "invalid_json",
        escalates=True,
        explanation="JSON was requested and the answer is not valid JSON",
    ),
    "truncated": Check(
        "truncated",
        escalates=False,
        explanation=(
            "the answer was cut off at max_tokens. A stronger model hits the "
            "same limit, so raise max_tokens rather than paying twice"
        ),
    ),
    "refused": Check(
        "refused",
        escalates=False,
        explanation=(
            "the model declined. Recorded and passed through - escalating "
            "until a model complies would be shopping for a yes"
        ),
    ),
    "hedged": Check(
        "hedged",
        escalates=False,
        explanation=(
            "the model expressed uncertainty, which may be the correct answer. "
            "Off by default: escalating here punishes honesty"
        ),
    ),
}


@dataclass
class Finding:
    """One check that fired."""

    name: str
    detail: str = ""

    @property
    def escalates(self) -> bool:
        check = CHECKS.get(self.name)
        return bool(check and check.escalates)


@dataclass
class Inspection:
    """Everything noticed about one answer."""

    findings: list[Finding]

    @property
    def failed(self) -> bool:
        return bool(self.findings)

    @property
    def should_escalate(self) -> bool:
        return any(finding.escalates for finding in self.findings)

    @property
    def names(self) -> str:
        return ",".join(finding.name for finding in self.findings)

    def describe(self) -> str:
        if not self.findings:
            return "no problems detected"
        return "; ".join(
            f"{finding.name}"
            + (f" ({finding.detail})" if finding.detail else "")
            for finding in self.findings
        )


# --- Reading a response -----------------------------------------------------


def answer_text(body: Any) -> str:
    """The assistant's text, from an OpenAI-shaped response body."""
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def finish_reason(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def wants_json(payload: Any) -> bool:
    """Did the caller ask for JSON back?

    Only checked when they did. Running a JSON parse over ordinary prose and
    calling the failure a defect would flag every normal answer ever given.
    """
    if not isinstance(payload, dict):
        return False
    fmt = payload.get("response_format")
    if isinstance(fmt, dict):
        return str(fmt.get("type", "")).startswith("json")
    return False


# --- The checks -------------------------------------------------------------


def inspect(body: Any, payload: Any = None) -> Inspection:
    """Look at one answer and report everything obviously wrong with it.

    Pure and cheap: string operations over a response already in memory. No
    model is called, nothing is scored, and it adds no measurable latency.
    """
    findings: list[Finding] = []
    text = answer_text(body)

    if len(text.strip()) < MIN_CONTENT_CHARS:
        findings.append(Finding("empty_response"))
        # Nothing further is worth checking about an empty answer, and
        # reporting "empty AND not valid JSON" is noise, not information.
        return Inspection(findings)

    if finish_reason(body) == "length":
        findings.append(Finding("truncated"))

    if wants_json(payload):
        stripped = _strip_code_fence(text)
        try:
            json.loads(stripped)
        except ValueError as exc:
            findings.append(Finding("invalid_json", str(exc)[:80]))

    opening = text[:OPENING_CHARS]
    if REFUSAL_PATTERNS.search(opening):
        findings.append(Finding("refused"))
    elif HEDGING_PATTERNS.search(opening):
        # `elif`: a refusal is already the stronger statement, and reporting
        # both would double-count one sentence.
        findings.append(Finding("hedged"))

    return Inspection(findings)


def _strip_code_fence(text: str) -> str:
    """Models often wrap JSON in ```json ... ``` even when asked not to.

    That is a formatting quirk, not a failure to produce JSON, so it is
    unwrapped before parsing rather than counted as invalid.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    without_open = stripped.split("\n", 1)[-1]
    return without_open.rsplit("```", 1)[0].strip()
