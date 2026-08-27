"""Numbers a monitoring system can scrape.

Exposed in the Prometheus text format, which almost every monitoring tool
reads. Written by hand rather than pulling in a client library: the format is
about forty lines of string building, and this project already carries more
runtime dependencies than it started with.

The one rule that matters here is CARDINALITY. Every distinct combination of
label values becomes a separate time series that the monitoring system stores
forever. Labelling by user id, or request id, or prompt would create a new
series per user or per request and eventually take the monitoring system down -
a genuinely common way to cause an outage with observability code. So labels
are only ever drawn from small, fixed sets: a status, a provider name, a model
name. Never anything a caller controls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

#: Buckets for latency, in seconds. Chosen to straddle what actually happens
#: here: sub-second for a cached answer or a small local model, tens of seconds
#: for a large one running partly on CPU.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

Labels = tuple[tuple[str, str], ...]


def _labels(**kwargs: str) -> Labels:
    """Normalise labels into something hashable and consistently ordered."""
    return tuple(sorted((k, str(v)) for k, v in kwargs.items() if v is not None))


def _render_labels(labels: Labels) -> str:
    if not labels:
        return ""
    def escape(value: str) -> str:
        # Backslashes first: escaping quotes first would then double the
        # backslashes this step introduces.
        return value.replace("\\", "\\\\").replace('"', '\\"')

    inner = ",".join(f'{name}="{escape(value)}"' for name, value in labels)
    return "{" + inner + "}"


@dataclass
class _Histogram:
    counts: list[int] = field(default_factory=lambda: [0] * (len(LATENCY_BUCKETS) + 1))
    total: float = 0.0
    observations: int = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.observations += 1
        for index, edge in enumerate(LATENCY_BUCKETS):
            if value <= edge:
                self.counts[index] += 1
                return
        self.counts[-1] += 1  # the +Inf bucket


class Metrics:
    """A small counter/gauge/histogram registry.

    Thread-safe: the server handles requests concurrently, and two threads
    incrementing the same counter without a lock silently lose increments.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, Labels], float] = {}
        self._gauges: dict[tuple[str, Labels], float] = {}
        self._histograms: dict[tuple[str, Labels], _Histogram] = {}
        self._help: dict[str, str] = {}

    def describe(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    def increment(self, name: str, amount: float = 1.0, **labels: str) -> None:
        key = (name, _labels(**labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[(name, _labels(**labels))] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, _labels(**labels))
        with self._lock:
            self._histograms.setdefault(key, _Histogram()).observe(value)

    def snapshot(self) -> dict:
        """A plain dictionary of everything recorded, for /health and tests."""
        with self._lock:
            return {
                "counters": {
                    f"{name}{_render_labels(labels)}": value
                    for (name, labels), value in self._counters.items()
                },
                "gauges": {
                    f"{name}{_render_labels(labels)}": value
                    for (name, labels), value in self._gauges.items()
                },
            }

    def render(self) -> str:
        """The Prometheus text exposition format."""
        lines: list[str] = []

        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = dict(self._histograms)

        for kind, series in (("counter", counters), ("gauge", gauges)):
            for name in sorted({n for n, _ in series}):
                if help_text := self._help.get(name):
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                for (series_name, labels), value in sorted(series.items()):
                    if series_name == name:
                        lines.append(f"{name}{_render_labels(labels)} {value}")

        for name in sorted({n for n, _ in histograms}):
            if help_text := self._help.get(name):
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} histogram")
            for (series_name, labels), histogram in sorted(histograms.items()):
                if series_name != name:
                    continue
                cumulative = 0
                for index, edge in enumerate(LATENCY_BUCKETS):
                    cumulative += histogram.counts[index]
                    bucket = _render_labels(labels + (("le", str(edge)),))
                    lines.append(f"{name}_bucket{bucket} {cumulative}")
                cumulative += histogram.counts[-1]
                infinity = _render_labels(labels + (("le", "+Inf"),))
                lines.append(f"{name}_bucket{infinity} {cumulative}")
                lines.append(f"{name}_sum{_render_labels(labels)} {histogram.total}")
                lines.append(
                    f"{name}_count{_render_labels(labels)} {histogram.observations}"
                )

        return "\n".join(lines) + "\n"


#: Metric names, kept in one place so a typo cannot silently create a second
#: series that nobody is watching.
REQUESTS = "switchboard_requests_total"
REQUEST_DURATION = "switchboard_request_duration_seconds"
CACHE_EVENTS = "switchboard_cache_events_total"
PROVIDER_ATTEMPTS = "switchboard_provider_attempts_total"
FAILOVERS = "switchboard_failovers_total"
RATE_LIMITED = "switchboard_rate_limited_total"
TOKENS = "switchboard_tokens_total"
COST = "switchboard_simulated_cost_usd_total"


def build_registry() -> Metrics:
    metrics = Metrics()
    metrics.describe(REQUESTS, "Chat completion requests, by outcome.")
    metrics.describe(REQUEST_DURATION, "Time to serve a request, in seconds.")
    metrics.describe(CACHE_EVENTS, "Response cache hits, misses and skips.")
    metrics.describe(PROVIDER_ATTEMPTS, "Calls to a provider, by outcome.")
    metrics.describe(FAILOVERS, "Times a request moved to a backup provider.")
    metrics.describe(RATE_LIMITED, "Requests refused for exceeding a rate limit.")
    metrics.describe(TOKENS, "Tokens processed, by direction.")
    metrics.describe(COST, "Simulated spend. NOT real money - see the README.")
    return metrics
