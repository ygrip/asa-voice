from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal


MetricKind = Literal["counter", "gauge", "histogram"]


@dataclass(frozen=True)
class MetricSpec:
    kind: MetricKind
    labels: tuple[str, ...] = ()


METRIC_SPECS: dict[str, MetricSpec] = {
    "asa_voice_stt_sessions_active": MetricSpec("gauge", ("provider", "mode")),
    "asa_voice_stt_audio_received_seconds_total": MetricSpec("counter", ("provider",)),
    "asa_voice_stt_partial_latency_ms": MetricSpec("histogram"),
    "asa_voice_stt_final_latency_ms": MetricSpec("histogram"),
    "asa_voice_stt_decode_duration_ms": MetricSpec("histogram", ("provider", "profile")),
    "asa_voice_stt_audio_dropped_ms_total": MetricSpec("counter", ("reason",)),
    "asa_voice_stt_backpressure_events_total": MetricSpec("counter", ("layer",)),
    "asa_voice_stt_degraded_finals_total": MetricSpec("counter", ("finality",)),
    "asa_voice_stt_provider_fallback_total": MetricSpec(
        "counter", ("from_provider", "to_provider", "reason")
    ),
    "asa_voice_stt_temp_files_active": MetricSpec("gauge"),
    "asa_voice_stt_temp_file_bytes": MetricSpec("gauge"),
}

LABEL_VALUES: dict[str, frozenset[str]] = {
    "provider": frozenset({"faster_whisper", "openai", "unknown"}),
    "mode": frozenset({"command", "hands_free", "dictation"}),
    "profile": frozenset({"partial", "final"}),
    "reason": frozenset(
        {"pressure", "queue_limit", "remainder", "provider_error", "timeout", "unknown"}
    ),
    "layer": frozenset({"sidecar", "provider"}),
    "finality": frozenset(
        {
            "provider_final",
            "local_recovered_final",
            "partial_timeout",
            "connection_lost_partial",
            "cancelled",
        }
    ),
    "from_provider": frozenset({"faster_whisper", "openai", "unknown"}),
    "to_provider": frozenset({"faster_whisper", "openai", "unknown"}),
}


@dataclass(frozen=True)
class HistogramValue:
    count: int
    total: float
    maximum: float


class VoiceMetrics:
    """Small content-free registry with fixed metric and label schemas.

    This adapter deliberately has no endpoint or backend. A production exporter can read snapshots
    later without changing instrumentation ownership or accepting free-form labels.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], HistogramValue] = {}

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        key, spec = self._key(name, labels)
        if spec.kind != "counter" or value < 0:
            raise ValueError(f"Metric {name} is not a non-negative counter")
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def add_gauge(self, name: str, delta: float, **labels: str) -> None:
        key, spec = self._key(name, labels)
        if spec.kind != "gauge":
            raise ValueError(f"Metric {name} is not a gauge")
        with self._lock:
            next_value = self._values.get(key, 0.0) + delta
            self._values[key] = max(0.0, next_value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        key, spec = self._key(name, labels)
        if spec.kind != "histogram" or value < 0:
            raise ValueError(f"Metric {name} is not a non-negative histogram")
        with self._lock:
            current = self._histograms.get(key, HistogramValue(0, 0.0, 0.0))
            self._histograms[key] = HistogramValue(
                current.count + 1,
                current.total + value,
                max(current.maximum, value),
            )

    def snapshot(self) -> dict[str, dict[tuple[tuple[str, str], ...], float | HistogramValue]]:
        result: dict[str, dict[tuple[tuple[str, str], ...], float | HistogramValue]] = {}
        with self._lock:
            for (name, labels), value in self._values.items():
                result.setdefault(name, {})[labels] = value
            for (name, labels), value in self._histograms.items():
                result.setdefault(name, {})[labels] = value
        return result

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._histograms.clear()

    def _key(
        self, name: str, labels: dict[str, str]
    ) -> tuple[tuple[str, tuple[tuple[str, str], ...]], MetricSpec]:
        spec = METRIC_SPECS.get(name)
        if spec is None:
            raise ValueError(f"Unknown voice metric: {name}")
        if set(labels) != set(spec.labels):
            raise ValueError(f"Metric {name} labels must be exactly {spec.labels}")
        for label, value in labels.items():
            if value not in LABEL_VALUES[label]:
                raise ValueError(f"Metric {name} label {label} has an unsafe value")
        ordered = tuple((label, labels[label]) for label in spec.labels)
        return (name, ordered), spec


voice_metrics = VoiceMetrics()
