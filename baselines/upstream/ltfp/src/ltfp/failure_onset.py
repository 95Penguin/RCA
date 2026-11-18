from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import DEFAULT_INTERVAL_SECONDS, TIME_COLUMN, TelemetryCase, clean_metric_frame, right_closed_resample


FAILURE_ONSET_SCHEMA = "ltfp.failure-onset-labels"
FAILURE_ONSET_SCHEMA_VERSION = 2
DEFAULT_LABEL_VERSION = "failure-onset-v2"


@dataclass(frozen=True)
class FailureOnsetConfig:
    """Configuration for auditable, train-independent onset labeling.

    Thresholds are fitted strictly before ``inject_time``.  A retrospective
    onset candidate is proposed only when at least ``required_breaches`` of
    the earliest qualifying ``window_bins`` completed buckets breach one or
    more selected SLIs.  Human review is required to accept the final label.
    """

    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    baseline_quantile: float = 0.995
    mad_multiplier: float = 6.0
    required_breaches: int = 3
    window_bins: int = 4
    min_baseline_bins: int = 8
    min_baseline_seconds: float = 120.0
    normal_baseline_fraction: float = 0.5
    max_followup_seconds: float | None = None
    preferred_service_tokens: tuple[str, ...] = (
        "frontend",
        "front-end",
        "front_end",
        "gateway",
        "ui-dashboard",
        "loadgenerator",
    )
    label_version: str = DEFAULT_LABEL_VERSION

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if not 0 < self.baseline_quantile <= 1:
            raise ValueError("baseline_quantile must be in (0, 1]")
        if self.mad_multiplier < 0:
            raise ValueError("mad_multiplier must be non-negative")
        if self.window_bins <= 0:
            raise ValueError("window_bins must be positive")
        if not 1 <= self.required_breaches <= self.window_bins:
            raise ValueError("required_breaches must be between 1 and window_bins")
        if self.min_baseline_bins <= 0:
            raise ValueError("min_baseline_bins must be positive")
        if self.min_baseline_seconds <= 0:
            raise ValueError("min_baseline_seconds must be positive")
        if not 0 < self.normal_baseline_fraction <= 0.5:
            raise ValueError("normal_baseline_fraction must be in (0, 0.5]")
        if self.max_followup_seconds is not None and self.max_followup_seconds <= 0:
            raise ValueError("max_followup_seconds must be positive when supplied")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["preferred_service_tokens"] = list(self.preferred_service_tokens)
        return payload


@dataclass(frozen=True)
class FailureOnsetResult:
    case_id: str
    scenario_id: str
    system: str | None
    inject_time: float | None
    failure_time: float | None
    censor_time: float | None
    confirmation_time: float | None
    observation_end: float | None
    event_observed: bool
    status: str
    reason: str
    sli_columns: tuple[str, ...]
    thresholds: dict[str, dict]
    breach_times: tuple[float, ...]
    confirming_window: tuple[dict, ...]
    label_version: str

    def to_record(self) -> dict:
        """Serialize an algorithmic candidate, never an accepted label.

        The 3-of-4 rule proposes an interval for blinded human review.  It
        cannot by itself create the independently reviewed ``failure_time``
        used by a confirmatory experiment.
        """

        audit_status = (
            "candidate_onset_pending_review"
            if self.event_observed
            else "candidate_censor_pending_review"
        )
        sli_boundaries = []
        for column in self.sli_columns:
            boundary = self.thresholds.get(column)
            if boundary is None:
                continue
            sli_boundaries.append(
                {
                    "name": column,
                    "kind": boundary.get("kind"),
                    "operator": boundary.get("operator"),
                    "threshold": boundary.get("threshold"),
                    "unit": boundary.get("unit", "source_native"),
                    "static_entry_weight": boundary.get("static_entry_weight"),
                    "source_columns": list(boundary.get("source_columns", [column])),
                    "provenance": boundary.get("provenance", {}),
                }
            )
        return {
            "case_id": self.case_id,
            "scenario_id": self.scenario_id,
            "repetition": self.case_id.rsplit("/", 1)[-1],
            "system": self.system,
            "inject_time": self.inject_time,
            "failure_time": None,
            "censor_time": self.censor_time,
            "confirmation_time": None,
            "observation_end": self.observation_end,
            "event_observed": None,
            "status": "pending_review",
            "audit_status": audit_status,
            "reason": "awaiting_independent_review",
            "candidate_failure_time": self.failure_time,
            "candidate_confirmation_time": self.confirmation_time,
            "candidate_event_observed": self.event_observed,
            "candidate_status": self.status,
            "candidate_reason": self.reason,
            # ``status`` is the survival outcome.  Audit/adjudication fields
            # are deliberately separate so a reviewer can change confidence
            # without silently changing the event/censor label.
            "review_status": "pending",
            "reviewer_decisions": [],
            "adjudicated": False,
            "adjudicator_decision": None,
            "adjudicator": None,
            "adjudication_notes": None,
            "sli": sli_boundaries,
            "sli_columns": list(self.sli_columns),
            "thresholds": self.thresholds,
            "breach_times": list(self.breach_times),
            "confirming_window": list(self.confirming_window),
            "label_version": self.label_version,
        }


def _sli_kind(column: str) -> str | None:
    name = column.lower()
    if "availability" in name or "success_rate" in name or "success-rate" in name:
        return "availability"
    if "latency" in name or "response-time" in name or "response_time" in name:
        return "latency"
    excluded_error_counters = (
        "container-memory-failures",
        "network-transmit-errors",
        "network-receive-errors",
        "filesystem",
        "fs-",
    )
    if any(token in name for token in excluded_error_counters):
        return None
    if "_trace_err_" in name or "istio-error" in name:
        return "error"
    if "failure_rate" in name or "failure-rate" in name or "failurerate" in name:
        return "error"
    if "_errors_" in name or name.endswith("_errors") or name.endswith("-errors"):
        return "error"
    if name.endswith("_error") or name.endswith("-error") or "_error_" in name or "-error-" in name:
        return "error"
    return None


def _is_failure_ratio(column: str) -> bool:
    name = str(column).lower()
    return any(token in name for token in ("request_failure_rate", "failure_rate", "error_rate"))


_ENTRY_SERVICE_PRIORITY = {
    "online-boutique": ("frontend-external", "frontend"),
    "online_boutique": ("frontend-external", "frontend"),
    "ob": ("frontend-external", "frontend"),
    "sock-shop": ("front-end",),
    "sock_shop": ("front-end",),
    "ss": ("front-end",),
    "train-ticket": ("ts-ui-dashboard",),
    "train_ticket": ("ts-ui-dashboard",),
    "tt": ("ts-ui-dashboard",),
}


def _service_prefix(column: str) -> str:
    name = str(column)
    suffixes = (
        "_request_failure_rate",
        "_availability",
        "_success_rate",
        "_latency-95",
        "_latency_p95",
        "_latency-90",
        "_latency_p90",
        "_latency",
        "_request_errors_total",
        "_request_error_total",
        "_errors_total",
        "_error_total",
        "_error",
        "-error",
    )
    for suffix in suffixes:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name.split("_", 1)[0]


def _counter_increments(values: pd.Series, column: str) -> pd.Series:
    """Convert explicit cumulative counters to interval increments.

    Compact RCAEval ``*_workload``/``*_error`` columns are already interval
    values.  Raw Prometheus-style ``*_total`` or ``*counter*`` columns are
    differenced, with counter resets treated as a new counter epoch.
    """

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    name = str(column).lower()
    explicit_counter = name.endswith("_total") or "counter" in name or name.endswith("_count_total")
    if not explicit_counter:
        return numeric.clip(lower=0)
    increments = numeric.diff()
    reset = increments < 0
    increments.loc[reset] = numeric.loc[reset]
    return increments.clip(lower=0)


def derive_request_failure_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add request-failure ratios from matched error/request interval counts."""

    derived = frame.copy()
    source_columns = dict(frame.attrs.get("sli_source_columns", {}))
    columns = [str(column) for column in frame.columns if column != TIME_COLUMN]
    for error_column in columns:
        if _sli_kind(error_column) != "error" or _is_failure_ratio(error_column):
            continue
        service = _service_prefix(error_column)
        exact_denominators = (
            f"{service}_workload",
            f"{service}_load",
            f"{service}_requests",
            f"{service}_request_count",
            f"{service}_requests_total",
            f"{service}_request_total",
        )
        denominator = next((column for column in exact_denominators if column in frame.columns), None)
        if denominator is None:
            denominator = next(
                (
                    column
                    for column in columns
                    if _service_prefix(column) == service
                    and any(token in column.lower() for token in ("request", "workload", "_load"))
                    and not any(token in column.lower() for token in ("error", "fail"))
                ),
                None,
            )
        if denominator is None:
            continue
        failures = _counter_increments(frame[error_column], error_column)
        requests = _counter_increments(frame[denominator], denominator)
        ratio = failures.div(requests.where(requests > 0))
        ratio = ratio.where(np.isfinite(ratio)).clip(lower=0, upper=1)
        derived_name = f"{service}_request_failure_rate"
        derived[derived_name] = ratio
        source_columns[derived_name] = [str(error_column), str(denominator)]
    derived.attrs["sli_source_columns"] = source_columns
    return derived


def _latency_preference(column: str) -> int | None:
    name = str(column).lower()
    if "latency-95" in name or "latency_p95" in name:
        return 0
    if "latency-90" in name or "latency_p90" in name:
        return 1
    if name.endswith("_latency") or name.endswith("-latency"):
        return 2
    return None


def _collapse_tail_latency(candidates: Sequence[str]) -> list[str]:
    selected = [column for column in candidates if _sli_kind(str(column)) != "latency"]
    by_service: dict[str, list[tuple[int, str]]] = {}
    for column in candidates:
        if _sli_kind(str(column)) != "latency":
            continue
        preference = _latency_preference(str(column))
        if preference is None:
            continue
        by_service.setdefault(_service_prefix(str(column)), []).append((preference, str(column)))
    for choices in by_service.values():
        selected.append(min(choices)[1])
    return selected


def select_sli_columns(
    frame: pd.DataFrame,
    preferred_service_tokens: Sequence[str] = FailureOnsetConfig.preferred_service_tokens,
    system: str | None = None,
) -> tuple[str, ...]:
    """Select registered user-facing error/latency SLIs.

    If any front-door service SLI exists, only those externally meaningful
    indicators are used.  For a registered benchmark system, absence of its
    front-door SLI is returned explicitly instead of silently substituting
    internal-service telemetry.  Such a case must be censored or handled by a
    separately versioned, auditable entry aggregation rule.  Low-level
    resource error counters are excluded.
    """

    candidates = [column for column in frame.columns if column != TIME_COLUMN and _sli_kind(str(column))]
    # When a request-normalized failure rate exists, do not also use its raw
    # error count: workload volume must not redefine service failure.
    rate_services = {
        _service_prefix(str(column))
        for column in candidates
        if _is_failure_ratio(str(column))
    }
    candidates = [
        column
        for column in candidates
        if not (
            _sli_kind(str(column)) == "error"
            and not _is_failure_ratio(str(column))
            and _service_prefix(str(column)) in rate_services
        )
    ]
    candidates = _collapse_tail_latency(candidates)
    normalized_system = str(system or "").strip().lower().replace(" ", "-")
    priorities = _ENTRY_SERVICE_PRIORITY.get(normalized_system, ())
    for service in priorities:
        entry = [column for column in candidates if _service_prefix(str(column)).lower() == service]
        if entry:
            return tuple(sorted(entry, key=lambda column: (_sli_kind(str(column)) == "latency", str(column))))
    if priorities:
        return ()
    preferred_tokens = tuple(str(token).lower() for token in preferred_service_tokens)
    preferred = [column for column in candidates if any(token in str(column).lower() for token in preferred_tokens)]
    selected = preferred or candidates
    return tuple(sorted(selected, key=lambda column: (_sli_kind(str(column)) != "error", str(column))))


def _complete_bucket_bounds(frame: pd.DataFrame, interval_seconds: int) -> tuple[float | None, float | None]:
    if frame.empty:
        return None, None
    times = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").dropna().to_numpy(dtype=float)
    if len(times) == 0:
        return None, None
    first_complete = math.ceil((float(times.min()) + interval_seconds) / interval_seconds) * interval_seconds
    last_complete = math.floor(float(times.max()) / interval_seconds) * interval_seconds
    if first_complete > last_complete:
        return None, None
    return float(first_complete), float(last_complete)


def completed_sli_frame(
    frame: pd.DataFrame,
    sli_columns: Sequence[str],
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> pd.DataFrame:
    """Return only fully observed right-closed SLI buckets."""

    clean = clean_metric_frame(frame)
    columns = [column for column in sli_columns if column in clean.columns]
    if not columns:
        return pd.DataFrame(columns=[TIME_COLUMN])
    first_complete, last_complete = _complete_bucket_bounds(clean, interval_seconds)
    if first_complete is None or last_complete is None:
        return pd.DataFrame(columns=[TIME_COLUMN, *columns])
    sampled = right_closed_resample(
        clean[[TIME_COLUMN, *columns]],
        interval_seconds=interval_seconds,
        agg="max",
        fill_empty=None,
    )
    mask = (sampled[TIME_COLUMN] >= first_complete) & (sampled[TIME_COLUMN] <= last_complete)
    return sampled.loc[mask].reset_index(drop=True)


def fit_sli_thresholds(
    sampled: pd.DataFrame,
    inject_time: float,
    sli_columns: Sequence[str],
    config: FailureOnsetConfig,
) -> dict[str, dict]:
    """Fit frozen disruption boundaries on a designated normal segment.

    Only the earliest configured fraction (at most 50%) of completed
    pre-injection buckets is eligible.  This keeps activation-adjacent drift
    out of the normal reference and requires at least 120 seconds by default.
    """

    pre_injection = sampled.loc[sampled[TIME_COLUMN] < float(inject_time)].sort_values(TIME_COLUMN)
    baseline_count = int(math.floor(len(pre_injection) * config.normal_baseline_fraction))
    baseline = pre_injection.iloc[:baseline_count]
    if len(baseline):
        baseline_duration = float(len(baseline) * config.interval_seconds)
        baseline_start = float(baseline.iloc[0][TIME_COLUMN] - config.interval_seconds)
        baseline_end = float(baseline.iloc[-1][TIME_COLUMN])
    else:
        baseline_duration = 0.0
        baseline_start = None
        baseline_end = None
    thresholds: dict[str, dict] = {}
    for column in sli_columns:
        if column not in baseline:
            continue
        values = pd.to_numeric(baseline[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < config.min_baseline_bins or baseline_duration < config.min_baseline_seconds:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = 1.4826 * mad
        kind = _sli_kind(str(column))
        if kind == "availability":
            tail_quantile = 1.0 - config.baseline_quantile
            quantile = float(np.quantile(values, tail_quantile))
            robust_limit = median - config.mad_multiplier * robust_scale
            threshold = min(quantile, robust_limit)
            operator = "<"
            estimator = "min(empirical_lower_quantile, median - mad_multiplier * 1.4826 * MAD)"
        else:
            tail_quantile = config.baseline_quantile
            quantile = float(np.quantile(values, tail_quantile))
            robust_limit = median + config.mad_multiplier * robust_scale
            threshold = max(quantile, robust_limit)
            operator = ">"
            estimator = "max(empirical_upper_quantile, median + mad_multiplier * 1.4826 * MAD)"
        operational_floor = None
        if kind == "latency":
            operational_floor = 1.5 * median
            threshold = max(threshold, operational_floor)
            estimator = f"max({estimator}, 1.5 * baseline_median)"
        elif _is_failure_ratio(str(column)):
            operational_floor = 0.05
            threshold = max(threshold, operational_floor)
            estimator = f"max({estimator}, 0.05)"
        thresholds[str(column)] = {
            "kind": kind,
            "operator": operator,
            "threshold": float(threshold),
            # RCAEval exports do not consistently declare physical units.
            # Preserve the native value rather than guessing ms, seconds, or
            # rate/count semantics from a column name.
            "unit": "ratio" if _is_failure_ratio(str(column)) else "source_native",
            "baseline_count": int(len(values)),
            "baseline_median": median,
            "baseline_mad": mad,
            "baseline_quantile": tail_quantile,
            "baseline_quantile_value": quantile,
            "robust_limit": float(robust_limit),
            "operational_floor": operational_floor,
            "provenance": {
                "selection": "designated_user_facing_sli",
                "fit_partition": "earliest_normal_fraction_strictly_pre_injection",
                "fit_end_exclusive": float(inject_time),
                "normal_baseline_fraction": config.normal_baseline_fraction,
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "baseline_duration_seconds": baseline_duration,
                "minimum_baseline_seconds": config.min_baseline_seconds,
                "estimator": estimator,
                "baseline_quantile": tail_quantile,
                "mad_multiplier": config.mad_multiplier,
                "frozen_before_model_scoring": True,
            },
        }
    return thresholds


def _empty_result(
    case: TelemetryCase,
    config: FailureOnsetConfig,
    *,
    status: str,
    reason: str,
    censor_time: float | None,
    observation_end: float | None,
    sli_columns: Sequence[str] = (),
    thresholds: Mapping[str, dict] | None = None,
) -> FailureOnsetResult:
    return FailureOnsetResult(
        case_id=str(case.case_id or case.name),
        scenario_id=str(case.scenario_id or case.case_id or case.name),
        system=case.system,
        inject_time=case.inject_time,
        failure_time=None,
        censor_time=censor_time,
        confirmation_time=None,
        observation_end=observation_end,
        event_observed=False,
        status=status,
        reason=reason,
        sli_columns=tuple(str(column) for column in sli_columns),
        thresholds=dict(thresholds or {}),
        breach_times=(),
        confirming_window=(),
        label_version=config.label_version,
    )


def _boundary_breach(values: pd.Series, boundary: Mapping) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = float(boundary["threshold"])
    operator = boundary.get("operator")
    if operator == ">":
        return numeric > threshold
    if operator == "<":
        return numeric < threshold
    raise ValueError(f"unsupported SLI boundary operator: {operator!r}")


def infer_failure_onset(case: TelemetryCase, config: FailureOnsetConfig | None = None) -> FailureOnsetResult:
    """Infer an SLI onset/censor candidate for subsequent blinded review."""

    config = config or FailureOnsetConfig()
    clean = clean_metric_frame(case.frame)
    if clean.empty:
        return _empty_result(
            case,
            config,
            status="invalid",
            reason="empty_telemetry",
            censor_time=None,
            observation_end=None,
        )

    raw_end = float(clean[TIME_COLUMN].max())
    last_complete = math.floor(raw_end / config.interval_seconds) * config.interval_seconds
    observation_end = float(last_complete)
    inject_time = case.inject_time
    if inject_time is None:
        return _empty_result(
            case,
            config,
            status="invalid",
            reason="missing_inject_time",
            censor_time=observation_end,
            observation_end=observation_end,
        )

    if observation_end < float(inject_time):
        return _empty_result(
            case,
            config,
            status="invalid",
            reason="telemetry_ends_before_inject_time",
            censor_time=None,
            observation_end=observation_end,
        )

    censor_time = observation_end
    if case.censor_time is not None:
        censor_time = min(censor_time, float(case.censor_time))
    if config.max_followup_seconds is not None:
        censor_time = min(censor_time, float(inject_time) + config.max_followup_seconds)

    sli_frame = derive_request_failure_rates(clean)
    sli_source_columns = dict(sli_frame.attrs.get("sli_source_columns", {}))
    sli_columns = select_sli_columns(
        sli_frame,
        config.preferred_service_tokens,
        system=case.system,
    )
    if not sli_columns:
        return _empty_result(
            case,
            config,
            status="right_censored",
            reason="no_sli_columns",
            censor_time=censor_time,
            observation_end=observation_end,
        )

    sampled = completed_sli_frame(sli_frame, sli_columns, interval_seconds=config.interval_seconds)
    thresholds = fit_sli_thresholds(sampled, float(inject_time), sli_columns, config)
    usable_columns = tuple(column for column in sli_columns if column in thresholds)
    if usable_columns:
        static_weight = 1.0 / len(usable_columns)
        for column in usable_columns:
            thresholds[column]["static_entry_weight"] = static_weight
            thresholds[column]["source_columns"] = list(
                sli_source_columns.get(column, [column])
            )
    if not usable_columns:
        return _empty_result(
            case,
            config,
            status="right_censored",
            reason="insufficient_pre_injection_baseline",
            censor_time=censor_time,
            observation_end=observation_end,
            sli_columns=sli_columns,
        )

    post = sampled.loc[
        (sampled[TIME_COLUMN] > float(inject_time)) & (sampled[TIME_COLUMN] <= censor_time),
        [TIME_COLUMN, *usable_columns],
    ].copy()
    if len(post) < config.window_bins:
        return _empty_result(
            case,
            config,
            status="right_censored",
            reason="insufficient_post_injection_followup",
            censor_time=censor_time,
            observation_end=observation_end,
            sli_columns=usable_columns,
            thresholds=thresholds,
        )

    per_column_breach = pd.DataFrame(index=post.index)
    for column in usable_columns:
        per_column_breach[column] = _boundary_breach(post[column], thresholds[column])
    post["_breach"] = per_column_breach.fillna(False).any(axis=1)
    breach_times = tuple(float(value) for value in post.loc[post["_breach"], TIME_COLUMN])

    for end in range(config.window_bins - 1, len(post)):
        window = post.iloc[end - config.window_bins + 1 : end + 1]
        if int(window["_breach"].sum()) < config.required_breaches:
            continue
        breached = window.loc[window["_breach"]]
        onset_time = float(breached.iloc[0][TIME_COLUMN])
        confirmation_time = float(window.iloc[-1][TIME_COLUMN])
        evidence = []
        for _, row in window.iterrows():
            breached_columns = [
                column
                for column in usable_columns
                if pd.notna(row[column])
                and bool(_boundary_breach(pd.Series([row[column]]), thresholds[column]).iloc[0])
            ]
            evidence.append(
                {
                    "time": float(row[TIME_COLUMN]),
                    "breach": bool(row["_breach"]),
                    "breached_sli_columns": breached_columns,
                }
            )
        return FailureOnsetResult(
            case_id=str(case.case_id or case.name),
            scenario_id=str(case.scenario_id or case.case_id or case.name),
            system=case.system,
            inject_time=float(inject_time),
            failure_time=onset_time,
            censor_time=censor_time,
            confirmation_time=confirmation_time,
            observation_end=observation_end,
            event_observed=True,
            status="observed",
            reason=f"first_{config.required_breaches}_of_{config.window_bins}_sli_breach",
            sli_columns=usable_columns,
            thresholds=thresholds,
            breach_times=breach_times,
            confirming_window=tuple(evidence),
            label_version=config.label_version,
        )

    return _empty_result(
        case,
        config,
        status="right_censored",
        reason="no_sustained_post_injection_sli_breach",
        censor_time=censor_time,
        observation_end=observation_end,
        sli_columns=usable_columns,
        thresholds=thresholds,
    )


def with_failure_onset(case: TelemetryCase, result: FailureOnsetResult) -> TelemetryCase:
    """Return a case carrying onset/censor labels without mutating telemetry."""

    metadata = dict(case.metadata or {})
    metadata.update(
        {
            "failure_time": result.failure_time,
            "censor_time": result.censor_time,
            "label_version": result.label_version,
            "label_status": result.status,
            "failure_onset_reason": result.reason,
            "failure_confirmation_time": result.confirmation_time,
        }
    )
    return replace(
        case,
        incident_time=result.failure_time,
        failure_time=result.failure_time,
        censor_time=result.censor_time,
        metadata=metadata,
    )


def build_failure_onset_manifest(
    cases: Iterable[TelemetryCase],
    config: FailureOnsetConfig | None = None,
    source_root: str | Path | None = None,
) -> dict:
    config = config or FailureOnsetConfig()
    pairs = [(case, infer_failure_onset(case, config)) for case in cases]
    pairs.sort(key=lambda pair: pair[1].case_id)
    results = [result for _, result in pairs]
    observed = sum(result.event_observed for result in results)
    records = []
    for case, result in pairs:
        record = result.to_record()
        record["root_label"] = case.root_label
        record["fault_type"] = case.fault_type
        record["topology"] = dict(case.topology or {})
        records.append(record)
    return {
        "schema": FAILURE_ONSET_SCHEMA,
        "schema_version": FAILURE_ONSET_SCHEMA_VERSION,
        "stage": "candidate_review",
        "ready_for_confirmatory_use": False,
        "label_version": config.label_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root) if source_root is not None else None,
        "config": config.to_dict(),
        "summary": {
            "case_count": len(results),
            "candidate_onset_count": int(observed),
            "candidate_censor_count": int(len(results) - observed),
            "pending_review_count": len(results),
            "accepted_onset_count": 0,
            "right_censored_count": 0,
        },
        "cases": records,
    }


_REVIEW_DECISION_ALIASES = {
    "onset": "onset",
    "accepted_onset": "onset",
    "no_onset": "no_onset",
    "censored_no_onset": "no_onset",
    "missing_telemetry": "missing_telemetry",
    "censored_missing_telemetry": "missing_telemetry",
    "ambiguous_timing": "ambiguous_timing",
    "censored_ambiguous_timing": "ambiguous_timing",
}


def _review_rows(review_decisions: str | Path | Mapping | Sequence[Mapping]) -> list[Mapping]:
    if isinstance(review_decisions, (str, Path)):
        payload = json.loads(Path(review_decisions).read_text(encoding="utf-8"))
    else:
        payload = review_decisions
    if isinstance(payload, Mapping):
        payload = payload.get("decisions", payload.get("reviews", payload))
        if isinstance(payload, Mapping):
            rows = []
            for case_id, decisions in payload.items():
                if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
                    raise ValueError(f"review decisions for {case_id!r} must be a list")
                for decision in decisions:
                    if not isinstance(decision, Mapping):
                        raise ValueError(f"review decision for {case_id!r} must be an object")
                    row = dict(decision)
                    row.setdefault("case_id", str(case_id))
                    rows.append(row)
            payload = rows
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("review decisions must be a list or a case_id-to-list mapping")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("every review decision must be an object")
    return list(payload)


def _normalise_review_decisions(
    review_decisions: str | Path | Mapping | Sequence[Mapping],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    seen: set[tuple[str, str, str]] = set()
    for raw in _review_rows(review_decisions):
        case_id = str(raw.get("case_id") or "").strip()
        reviewer_id = str(raw.get("reviewer_id", raw.get("reviewer", ""))).strip()
        role = str(raw.get("role", "reviewer")).strip().lower()
        decision_raw = str(raw.get("decision") or "").strip().lower()
        decision = _REVIEW_DECISION_ALIASES.get(decision_raw)
        if not case_id or not reviewer_id:
            raise ValueError("each review decision needs case_id and reviewer_id")
        if role not in {"reviewer", "adjudicator"}:
            raise ValueError(f"unsupported review role for {case_id!r}: {role!r}")
        if decision is None:
            raise ValueError(f"unsupported review decision for {case_id!r}: {decision_raw!r}")
        if raw.get("blinded_to_model_predictions") is not True:
            raise ValueError(
                f"review decision for {case_id!r}/{reviewer_id!r} must affirm "
                "blinded_to_model_predictions=true"
            )
        failure_time = raw.get("failure_time", raw.get("onset_time"))
        failure_time = None if failure_time in (None, "") else float(failure_time)
        if decision == "onset" and (failure_time is None or not math.isfinite(failure_time)):
            raise ValueError(f"onset decision for {case_id!r} needs a finite failure_time")
        if decision != "onset" and failure_time is not None:
            raise ValueError(f"non-onset decision for {case_id!r} cannot set failure_time")
        key = (case_id, reviewer_id, role)
        if key in seen:
            raise ValueError(f"duplicate {role} decision for {case_id!r}/{reviewer_id!r}")
        seen.add(key)
        normalized = {
            "case_id": case_id,
            "reviewer_id": reviewer_id,
            "role": role,
            "decision": decision,
            "failure_time": failure_time,
            "blinded_to_model_predictions": True,
            "notes": raw.get("notes"),
        }
        grouped.setdefault(case_id, []).append(normalized)
    return grouped


def _resolve_reviewed_case(record: Mapping, decisions: Sequence[Mapping], interval_seconds: int) -> dict:
    case_id = str(record["case_id"])
    reviewers = sorted(
        (dict(item) for item in decisions if item.get("role") == "reviewer"),
        key=lambda item: item["reviewer_id"],
    )
    adjudicators = sorted(
        (dict(item) for item in decisions if item.get("role") == "adjudicator"),
        key=lambda item: item["reviewer_id"],
    )
    if len(reviewers) != 2:
        raise ValueError(f"case {case_id!r} needs exactly two independent reviewer decisions")
    if reviewers[0]["reviewer_id"] == reviewers[1]["reviewer_id"]:
        raise ValueError(f"case {case_id!r} needs two distinct reviewers")

    agreed = reviewers[0]["decision"] == reviewers[1]["decision"]
    outcome = reviewers[0]["decision"] if agreed else None
    failure_time = None
    resolution_rule = "two_reviewer_exact_agreement"
    if outcome == "onset":
        onset_times = [float(item["failure_time"]) for item in reviewers]
        if abs(onset_times[0] - onset_times[1]) <= float(interval_seconds) + 1e-9:
            # Both bins are defensible under the registered one-bin agreement
            # tolerance; retain the earlier as the onset boundary.
            failure_time = min(onset_times)
            resolution_rule = "two_reviewer_agreement_within_one_bin_earliest"
        else:
            outcome = None

    adjudicated = outcome is None
    adjudicator_decision = None
    if adjudicated:
        if len(adjudicators) != 1:
            raise ValueError(
                f"case {case_id!r} has reviewer disagreement and needs exactly one adjudicator decision"
            )
        adjudicator_decision = adjudicators[0]
        outcome = str(adjudicator_decision["decision"])
        failure_time = (
            float(adjudicator_decision["failure_time"])
            if outcome == "onset"
            else None
        )
        resolution_rule = "third_reviewer_adjudication"
    elif adjudicators:
        raise ValueError(f"case {case_id!r} has an unnecessary adjudicator decision")

    inject_time = record.get("inject_time")
    censor_time = record.get("censor_time")
    if inject_time is None or censor_time is None:
        raise ValueError(f"case {case_id!r} lacks inject_time or censor_time and cannot become a final label")
    inject_time = float(inject_time)
    censor_time = float(censor_time)
    if censor_time < inject_time:
        raise ValueError(f"case {case_id!r} has censor_time before inject_time")
    if outcome == "onset":
        if failure_time is None or not (inject_time < failure_time <= censor_time):
            raise ValueError(
                f"reviewed onset for {case_id!r} must satisfy inject_time < failure_time <= censor_time"
            )
        grid_position = failure_time / float(interval_seconds)
        if abs(grid_position - round(grid_position)) > 1e-8:
            raise ValueError(f"reviewed onset for {case_id!r} is not on the {interval_seconds}s grid")

    final = deepcopy(dict(record))
    final.update(
        {
            "failure_time": failure_time if outcome == "onset" else None,
            "confirmation_time": record.get("candidate_confirmation_time") if outcome == "onset" else None,
            "event_observed": outcome == "onset",
            "status": "observed" if outcome == "onset" else "right_censored",
            "review_status": "complete",
            "reviewer_decisions": reviewers,
            "adjudicated": adjudicated,
            "adjudicator_decision": adjudicator_decision,
            "adjudicator": adjudicator_decision["reviewer_id"] if adjudicator_decision else None,
            "adjudication_notes": adjudicator_decision.get("notes") if adjudicator_decision else None,
            "resolution_rule": resolution_rule,
        }
    )
    if outcome == "onset":
        final["audit_status"] = "accepted_onset"
        final["reason"] = "independent_review_accepted_onset"
    else:
        reason_by_outcome = {
            "no_onset": ("censored_no_onset", "independent_review_no_onset"),
            "missing_telemetry": ("censored_missing_telemetry", "independent_review_missing_telemetry"),
            "ambiguous_timing": ("censored_ambiguous_timing", "independent_review_ambiguous_timing"),
        }
        if outcome not in reason_by_outcome:
            raise ValueError(f"unsupported resolved outcome for {case_id!r}: {outcome!r}")
        final["audit_status"], final["reason"] = reason_by_outcome[outcome]
    return final


def finalize_failure_onset_manifest(
    candidate_manifest: Mapping,
    review_decisions: str | Path | Mapping | Sequence[Mapping],
) -> dict:
    """Resolve a candidate manifest into labels after registered review.

    Every case requires two distinct reviewers blinded to model predictions.
    Onset times within one grid bin are accepted using the earlier boundary;
    all other disagreement requires exactly one adjudicator.
    """

    if candidate_manifest.get("schema") != FAILURE_ONSET_SCHEMA:
        raise ValueError("review finalization requires a failure-onset manifest")
    if candidate_manifest.get("schema_version") != FAILURE_ONSET_SCHEMA_VERSION:
        raise ValueError("review finalization requires the current failure-onset schema version")
    if candidate_manifest.get("stage") != "candidate_review":
        raise ValueError("only a candidate_review manifest can be finalized")
    config = candidate_manifest.get("config", {})
    interval_seconds = int(config.get("interval_seconds", DEFAULT_INTERVAL_SECONDS))
    grouped = _normalise_review_decisions(review_decisions)
    case_records = list(candidate_manifest.get("cases", []))
    listed_case_ids = [str(record["case_id"]) for record in case_records]
    if len(listed_case_ids) != len(set(listed_case_ids)):
        raise ValueError("candidate manifest case_id values must be unique")
    for record in case_records:
        if record.get("status") != "pending_review" or not str(record.get("audit_status", "")).startswith(
            "candidate_"
        ):
            raise ValueError(f"case {record.get('case_id')!r} is not a pending-review candidate")
        if record.get("failure_time") is not None:
            raise ValueError(f"candidate case {record.get('case_id')!r} already sets failure_time")
    case_ids = set(listed_case_ids)
    extra = sorted(set(grouped) - case_ids)
    if extra:
        raise ValueError(f"review decisions reference cases outside the candidate manifest: {extra[:5]}")
    missing = sorted(case_ids - set(grouped))
    if missing:
        raise ValueError(f"review decisions are missing candidate cases: {missing[:5]}")

    final_records = [
        _resolve_reviewed_case(record, grouped[str(record["case_id"])], interval_seconds)
        for record in case_records
    ]
    final_manifest = deepcopy(dict(candidate_manifest))
    final_manifest["stage"] = "reviewed_labels_pending_quality_control"
    final_manifest["ready_for_confirmatory_use"] = False
    final_manifest["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    final_manifest["review_protocol"] = {
        "independent_reviewer_count": 2,
        "agreement_tolerance_bins": 1,
        "within_tolerance_resolution": "earliest_reviewed_onset",
        "disagreement_resolution": "third_reviewer_adjudication",
        "model_prediction_blinding_required": True,
    }
    final_manifest["cases"] = final_records
    accepted = sum(record["event_observed"] for record in final_records)
    adjudicated = sum(record["adjudicated"] for record in final_records)
    final_manifest["summary"] = {
        "case_count": len(final_records),
        "accepted_onset_count": int(accepted),
        "right_censored_count": int(len(final_records) - accepted),
        "adjudicated_count": int(adjudicated),
        "pending_review_count": 0,
    }
    return final_manifest


def attach_failure_onset_quality_control(
    reviewed_manifest: Mapping,
    quality_report: Mapping,
    minimum_review_fraction: float = 0.20,
) -> dict:
    """Promote reviewed labels only after the registered agreement gate passes."""

    if reviewed_manifest.get("stage") != "reviewed_labels_pending_quality_control":
        raise ValueError("quality control can only be attached to fully reviewed pending labels")
    case_count = int(len(reviewed_manifest.get("cases", [])))
    n_reviewed = int(quality_report.get("n_reviewed", 0))
    minimum_count = int(math.ceil(case_count * float(minimum_review_fraction)))
    kappa = float(quality_report.get("cohens_kappa", float("nan")))
    median_difference = float(
        quality_report.get("onset_difference_median_seconds", float("nan"))
    )
    if n_reviewed < minimum_count:
        raise ValueError(
            f"quality-control sample is too small: {n_reviewed} reviewed, {minimum_count} required"
        )
    if (
        not bool(quality_report.get("passes", False))
        or not np.isfinite(kappa)
        or kappa < 0.80
        or not np.isfinite(median_difference)
        or median_difference > 30.0
    ):
        raise ValueError("quality-control report fails the registered kappa/onset-time gate")
    final_manifest = deepcopy(dict(reviewed_manifest))
    final_manifest["quality_control_audit"] = {
        **deepcopy(dict(quality_report)),
        "minimum_review_fraction": float(minimum_review_fraction),
        "minimum_review_count": minimum_count,
        "case_count": case_count,
    }
    final_manifest["stage"] = "final_labels"
    final_manifest["ready_for_confirmatory_use"] = True
    final_manifest["quality_control_accepted_at_utc"] = datetime.now(timezone.utc).isoformat()
    return final_manifest


def write_failure_onset_manifest(manifest: Mapping, path: str | Path) -> Path:
    """Atomically write a JSON label manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def read_failure_onset_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != FAILURE_ONSET_SCHEMA:
        raise ValueError(f"unsupported failure-onset manifest schema: {payload.get('schema')!r}")
    if payload.get("schema_version") != FAILURE_ONSET_SCHEMA_VERSION:
        raise ValueError(f"unsupported failure-onset schema version: {payload.get('schema_version')!r}")
    return payload


def apply_failure_onset_manifest(cases: Iterable[TelemetryCase], manifest: Mapping) -> list[TelemetryCase]:
    if manifest.get("stage") == "candidate_review" or not manifest.get("ready_for_confirmatory_use", False):
        raise ValueError("pending-review onset candidates cannot be applied as final failure labels")
    records = {}
    for record in manifest.get("cases", []):
        if not isinstance(record, Mapping) or record.get("case_id") is None:
            continue
        case_id = str(record["case_id"])
        if case_id in records:
            raise ValueError(f"duplicate case_id in final onset manifest: {case_id}")
        records[case_id] = record
    case_list = list(cases)
    missing = sorted(str(case.case_id or case.name) for case in case_list if str(case.case_id or case.name) not in records)
    if missing:
        raise KeyError(f"final onset manifest is missing cases: {missing[:5]}")
    labeled = []
    for case in case_list:
        record = records.get(str(case.case_id or case.name))
        result = FailureOnsetResult(
            case_id=str(record["case_id"]),
            scenario_id=str(record.get("scenario_id") or case.scenario_id or record["case_id"]),
            system=record.get("system"),
            inject_time=record.get("inject_time"),
            failure_time=record.get("failure_time"),
            censor_time=record.get("censor_time"),
            confirmation_time=record.get("confirmation_time"),
            observation_end=record.get("observation_end"),
            event_observed=bool(record.get("event_observed", record.get("failure_time") is not None)),
            status=str(record.get("status", "observed" if record.get("failure_time") is not None else "right_censored")),
            reason=str(record.get("reason", "manifest")),
            sli_columns=tuple(record.get("sli_columns", ())),
            thresholds=dict(record.get("thresholds", {})),
            breach_times=tuple(float(value) for value in record.get("breach_times", ())),
            confirming_window=tuple(record.get("confirming_window", ())),
            label_version=str(record.get("label_version", manifest.get("label_version", DEFAULT_LABEL_VERSION))),
        )
        labeled.append(with_failure_onset(case, result))
    return labeled
