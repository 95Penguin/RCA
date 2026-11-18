"""Case-level evaluation for failure onset prediction and localization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .dataset_v2 import FailureWindowDataset


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _exact_horizon_index(horizons: Sequence[float], requested: float) -> int:
    matches = np.flatnonzero(np.isclose(np.asarray(horizons, dtype=float), float(requested)))
    if len(matches) != 1:
        raise ValueError(f"primary horizon {requested:g}s is not present in {tuple(horizons)}")
    return int(matches[0])


def _alert_episode_count(alert: np.ndarray, eligible: np.ndarray) -> int:
    """Count rising alert episodes without joining across ineligible gaps."""

    positions = np.flatnonzero(eligible)
    if len(positions) == 0:
        return 0
    active = np.asarray(alert, dtype=bool)[positions]
    previous_active = np.r_[False, active[:-1]]
    contiguous = np.r_[False, np.diff(positions) == 1]
    starts = active & (~previous_active | ~contiguous)
    return int(starts.sum())


def evaluate_events(
    dataset: FailureWindowDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[list[dict], pd.DataFrame]:
    """Evaluate first alerts per physical case at every horizon.

    Only the first post-injection, pre-onset alert is scored.  A later timely
    alert cannot rescue an earlier too-early alert.  A too-early first alert or
    any first alert in a censored case is a false event prediction.  False
    alerts/hour counts rising alert episodes on the explicitly healthy
    pre-injection segment.
    """

    indices = np.asarray(indices, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(indices), len(dataset.horizons)):
        raise ValueError("probabilities shape does not match selected windows/horizons")
    case_values = np.asarray(dataset.case_ids, dtype=object)[indices]
    unique_cases = sorted(set(str(item) for item in case_values))
    event_rows: list[dict] = []
    per_case_rows: list[dict] = []

    for horizon_idx, horizon in enumerate(dataset.horizons):
        true_positive = 0
        false_positive = 0
        observed_events = 0
        leads: list[float] = []
        healthy_alert_episodes = 0
        healthy_alert_windows = 0
        healthy_windows = 0
        healthy_seconds = 0.0
        for case_id in unique_cases:
            local = np.flatnonzero(case_values == case_id)
            global_idx = indices[local]
            order = np.argsort(dataset.end_times[global_idx])
            local = local[order]
            global_idx = global_idx[order]
            end = dataset.end_times[global_idx]
            inject = float(dataset.inject_times[global_idx][0])
            failure = float(dataset.failure_times[global_idx][0])
            censor = float(dataset.censor_times[global_idx][0])
            observed = bool(np.isfinite(failure))
            score = probabilities[local, horizon_idx]
            alert = score >= float(threshold)
            valid_label = dataset.label_mask[global_idx, horizon_idx]
            healthy = end < inject
            healthy_valid = healthy & valid_label
            positive_diffs = np.diff(end)
            positive_diffs = positive_diffs[positive_diffs > 0]
            cadence = float(np.median(positive_diffs)) if len(positive_diffs) else 30.0
            case_healthy_windows = int(healthy_valid.sum())
            case_healthy_alert_windows = int((alert & healthy_valid).sum())
            case_healthy_episodes = _alert_episode_count(alert, healthy_valid)
            healthy_windows += case_healthy_windows
            healthy_alert_windows += case_healthy_alert_windows
            healthy_alert_episodes += case_healthy_episodes
            healthy_seconds += case_healthy_windows * cadence

            at_risk = (end >= inject) & (end < (failure if observed else censor)) & valid_label
            alert_positions = np.flatnonzero(alert & at_risk)
            predicted_event = len(alert_positions) > 0
            first = int(alert_positions[0]) if predicted_event else None
            first_lead = float(failure - end[first]) if predicted_event and observed else np.nan
            hit = bool(
                predicted_event
                and observed
                and first_lead > 0.0
                and first_lead <= float(horizon)
            )
            if observed:
                observed_events += 1
            if hit:
                true_positive += 1
                leads.append(first_lead)
            elif predicted_event:
                false_positive += 1

            per_case_rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": str(dataset.scenario_ids[global_idx[0]]),
                    "system": str(dataset.systems[global_idx[0]]),
                    "horizon": float(horizon),
                    "event_observed": observed,
                    "event_hit": hit,
                    "predicted_event": predicted_event,
                    "first_alert_time": float(end[first]) if predicted_event else np.nan,
                    "first_alert_lead_seconds": first_lead,
                    "first_alert_within_horizon": hit,
                    "healthy_false_alert_episodes": case_healthy_episodes,
                    "healthy_alert_windows": case_healthy_alert_windows,
                    "healthy_windows": case_healthy_windows,
                }
            )

        false_negative = max(observed_events - true_positive, 0)
        event_rows.append(
            {
                "horizon": float(horizon),
                "event_precision": _safe_divide(true_positive, true_positive + false_positive),
                "event_recall": _safe_divide(true_positive, observed_events),
                "event_true_positive": int(true_positive),
                "event_false_positive": int(false_positive),
                "event_false_negative": int(false_negative),
                "observed_events": int(observed_events),
                "first_alert_lead_median_seconds": float(np.median(leads)) if leads else np.nan,
                "first_alert_lead_q1_seconds": float(np.quantile(leads, 0.25)) if leads else np.nan,
                "first_alert_lead_q3_seconds": float(np.quantile(leads, 0.75)) if leads else np.nan,
                "false_alerts_per_hour": _safe_divide(healthy_alert_episodes, healthy_seconds / 3600.0),
                "healthy_false_alert_episodes": int(healthy_alert_episodes),
                "healthy_alert_windows": int(healthy_alert_windows),
                "healthy_windows": int(healthy_windows),
                "healthy_monitored_seconds": float(healthy_seconds),
            }
        )
    return event_rows, pd.DataFrame(per_case_rows)


def evaluate_first_alert_roots(
    dataset: FailureWindowDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    root_logits: np.ndarray,
    threshold: float,
    primary_horizon: float = 300.0,
) -> tuple[dict, pd.DataFrame]:
    """Rank every topology service once, at each case's first valid alert."""

    indices = np.asarray(indices, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    root_logits = np.asarray(root_logits, dtype=float)
    if root_logits.shape != (len(indices), len(dataset.schema.services)):
        raise ValueError("root_logits shape does not match test windows/services")
    horizon_idx = _exact_horizon_index(dataset.horizons, primary_horizon)
    case_values = np.asarray(dataset.case_ids, dtype=object)[indices]
    rows = []
    denominators = 0
    joint_hits = {1: 0, 3: 0}
    conditional_hits = {1: 0, 3: 0}
    conditional_reciprocal_ranks = []
    joint_reciprocal_rank_sum = 0.0
    for case_id in sorted(set(str(item) for item in case_values)):
        local = np.flatnonzero(case_values == case_id)
        global_idx = indices[local]
        failure = float(dataset.failure_times[global_idx][0])
        target = int(dataset.root_targets[global_idx][0])
        if not np.isfinite(failure) or target < 0:
            continue
        denominators += 1
        order = np.argsort(dataset.end_times[global_idx])
        local = local[order]
        global_idx = global_idx[order]
        end = dataset.end_times[global_idx]
        inject = float(dataset.inject_times[global_idx][0])
        lead = failure - end
        valid = (
            (end >= inject)
            & (end < failure)
            & (lead > 0.0)
            & dataset.label_mask[global_idx, horizon_idx]
        )
        alerts = (probabilities[local, horizon_idx] >= float(threshold)) & valid
        positions = np.flatnonzero(alerts)
        if len(positions) == 0:
            n_available_services = int(
                dataset.availability_mask[global_idx[0]].any(axis=-1).sum()
            )
            rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": str(dataset.scenario_ids[global_idx[0]]),
                    "alerted": False,
                    "alert_within_horizon": False,
                    "root_target": dataset.schema.services[target],
                    "rank": np.nan,
                    "n_root_candidates": int(len(dataset.schema.services)),
                    "n_available_services": n_available_services,
                }
            )
            continue
        first = int(positions[0])
        n_available_services = int(
            dataset.availability_mask[global_idx[first]].any(axis=-1).sum()
        )
        ranking = np.argsort(-root_logits[local[first]])
        rank = int(np.flatnonzero(ranking == target)[0]) + 1
        timely = bool(lead[first] <= dataset.horizons[horizon_idx])
        conditional_reciprocal_ranks.append(1.0 / rank)
        if timely:
            joint_reciprocal_rank_sum += 1.0 / rank
        for k in (1, 3):
            hit = rank <= min(k, len(dataset.schema.services))
            conditional_hits[k] += int(hit)
            joint_hits[k] += int(hit and timely)
        rows.append(
            {
                "case_id": case_id,
                "scenario_id": str(dataset.scenario_ids[global_idx[0]]),
                "alerted": True,
                "alert_within_horizon": timely,
                "alert_time": float(end[first]),
                "lead_seconds": float(lead[first]),
                "root_target": dataset.schema.services[target],
                "rank": rank,
                "top1_service": dataset.schema.services[int(ranking[0])],
                "top3_services": "|".join(dataset.schema.services[int(item)] for item in ranking[:3]),
                "n_root_candidates": int(len(dataset.schema.services)),
                "n_available_services": n_available_services,
            }
        )
    alerted = len(conditional_reciprocal_ranks)
    summary = {
        "primary_horizon": float(dataset.horizons[horizon_idx]),
        "eligible_cases": int(denominators),
        "alerted_cases": int(alerted),
        "joint_top1": _safe_divide(joint_hits[1], denominators),
        "joint_top3": _safe_divide(joint_hits[3], denominators),
        "conditional_top1": _safe_divide(conditional_hits[1], alerted),
        "conditional_top3": _safe_divide(conditional_hits[3], alerted),
        "joint_mrr": _safe_divide(joint_reciprocal_rank_sum, denominators),
        "conditional_mrr": (
            float(np.mean(conditional_reciprocal_ranks))
            if conditional_reciprocal_ranks
            else 0.0
        ),
        "n_root_candidates": int(len(dataset.schema.services)),
    }
    return summary, pd.DataFrame(rows)


def window_prediction_frame(
    dataset: FailureWindowDataset,
    indices: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    root_logits: np.ndarray,
) -> pd.DataFrame:
    indices = np.asarray(indices, dtype=int)
    payload = {
        "case_id": np.asarray(dataset.case_ids, dtype=object)[indices],
        "scenario_id": np.asarray(dataset.scenario_ids, dtype=object)[indices],
        "system": np.asarray(dataset.systems, dtype=object)[indices],
        "end_time": dataset.end_times[indices],
        "inject_time": dataset.inject_times[indices],
        "failure_time": dataset.failure_times[indices],
        "censor_time": dataset.censor_times[indices],
        "lead_time": dataset.lead_times[indices],
        "root_target": dataset.root_targets[indices],
    }
    for horizon_idx, horizon in enumerate(dataset.horizons):
        suffix = int(horizon) if float(horizon).is_integer() else horizon
        payload[f"label_{suffix}s"] = dataset.Y[indices, horizon_idx]
        payload[f"label_valid_{suffix}s"] = dataset.label_mask[indices, horizon_idx]
        payload[f"score_{suffix}s"] = raw_probabilities[:, horizon_idx]
        payload[f"probability_{suffix}s"] = calibrated_probabilities[:, horizon_idx]
    frame = pd.DataFrame(payload)
    for service_idx, service in enumerate(dataset.schema.services):
        frame[f"root_logit::{service}"] = root_logits[:, service_idx]
    return frame


def aggregate_seed_rows(rows: Sequence[dict], group_keys: Sequence[str]) -> pd.DataFrame:
    """Aggregate initialization seeds without treating them as independent cases."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column not in group_keys]
    grouped = frame.groupby(list(group_keys), dropna=False)
    mean = grouped[numeric].mean().add_suffix("_mean")
    std = grouped[numeric].std(ddof=1).fillna(0.0).add_suffix("_sd")
    return mean.join(std).reset_index()
