"""Independent-review utilities for candidate failure-onset labels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REVIEW_COLUMNS = (
    "case_id",
    "reviewer",
    "event_observed",
    "onset_time",
    "decision",
    "notes",
)


def stratified_audit_sample(
    records: pd.DataFrame,
    fraction: float = 0.20,
    seed: int = 7,
) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    required = {"case_id", "system", "scenario_id"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"candidate records are missing {sorted(missing)}")
    work = records.copy()
    work["subset"] = work["case_id"].astype(str).str.split("/", n=1).str[0]
    sampled = []
    rng = np.random.default_rng(seed)
    for _, group in work.groupby(["subset", "system"], sort=True):
        count = max(1, int(math.ceil(len(group) * fraction)))
        chosen = rng.choice(group.index.to_numpy(), size=min(count, len(group)), replace=False)
        sampled.append(work.loc[chosen])
    output = pd.concat(sampled).sort_values(["subset", "system", "scenario_id", "case_id"])
    output = output.reset_index(drop=True)
    output["reviewer"] = ""
    output["event_observed"] = ""
    output["onset_time"] = ""
    output["decision"] = ""
    output["notes"] = ""
    keep = [
        "case_id",
        "subset",
        "system",
        "scenario_id",
        "candidate_failure_time",
        "candidate_event_observed",
        "candidate_reason",
        "reviewer",
        "event_observed",
        "onset_time",
        "decision",
        "notes",
    ]
    return output[[column for column in keep if column in output]]


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "observed"}:
        return True
    if normalized in {"false", "0", "no", "censored"}:
        return False
    raise ValueError(f"invalid event_observed value: {value!r}")


def cohens_kappa(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("review decisions must be equally sized and non-empty")
    observed = float((left == right).mean())
    p_left = float(left.mean())
    p_right = float(right.mean())
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    if expected >= 1.0 - 1e-12:
        return 1.0 if observed >= 1.0 - 1e-12 else 0.0
    return float((observed - expected) / (1.0 - expected))


def evaluate_two_reviewer_audit(
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    minimum_kappa: float = 0.80,
    maximum_median_onset_difference_seconds: float = 30.0,
) -> dict:
    for name, frame in (("reviewer_a", reviewer_a), ("reviewer_b", reviewer_b)):
        missing = set(REVIEW_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
        if frame["case_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate case IDs")
    merged = reviewer_a.merge(reviewer_b, on="case_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(reviewer_a) or len(merged) != len(reviewer_b):
        raise ValueError("reviewer files must cover exactly the same audit sample")
    observed_a = np.asarray([_as_bool(value) for value in merged["event_observed_a"]])
    observed_b = np.asarray([_as_bool(value) for value in merged["event_observed_b"]])
    kappa = cohens_kappa(observed_a, observed_b)
    both_observed = observed_a & observed_b
    onset_a = pd.to_numeric(merged.loc[both_observed, "onset_time_a"], errors="coerce")
    onset_b = pd.to_numeric(merged.loc[both_observed, "onset_time_b"], errors="coerce")
    if onset_a.isna().any() or onset_b.isna().any():
        raise ValueError("both-observed reviews require finite onset_time values")
    differences = np.abs(onset_a.to_numpy(dtype=float) - onset_b.to_numpy(dtype=float))
    median_difference = float(np.median(differences)) if len(differences) else 0.0
    decision_agreement = observed_a == observed_b
    report = {
        "n_reviewed": int(len(merged)),
        "cohens_kappa": float(kappa),
        "minimum_kappa": float(minimum_kappa),
        "onset_difference_median_seconds": median_difference,
        "maximum_median_onset_difference_seconds": float(maximum_median_onset_difference_seconds),
        "event_decision_disagreements": int((~decision_agreement).sum()),
        "onset_pairs": int(len(differences)),
        "passes": bool(
            kappa >= minimum_kappa
            and median_difference <= maximum_median_onset_difference_seconds
        ),
        "requires_adjudication_case_ids": merged.loc[
            (~decision_agreement)
            | (
                both_observed
                & (
                    np.abs(
                        pd.to_numeric(merged["onset_time_a"], errors="coerce")
                        - pd.to_numeric(merged["onset_time_b"], errors="coerce")
                    )
                    > maximum_median_onset_difference_seconds
                )
            ),
            "case_id",
        ].astype(str).tolist(),
    }
    return report
