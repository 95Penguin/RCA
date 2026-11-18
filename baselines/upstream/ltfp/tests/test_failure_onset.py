from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.data import TIME_COLUMN, TelemetryCase, build_window_dataset  # noqa: E402
from ltfp.failure_onset import (  # noqa: E402
    FAILURE_ONSET_SCHEMA,
    FAILURE_ONSET_SCHEMA_VERSION,
    FailureOnsetConfig,
    attach_failure_onset_quality_control,
    apply_failure_onset_manifest,
    build_failure_onset_manifest,
    derive_request_failure_rates,
    finalize_failure_onset_manifest,
    infer_failure_onset,
    read_failure_onset_manifest,
    select_sli_columns,
    write_failure_onset_manifest,
)
from ltfp.protocol import load_label_manifest  # noqa: E402


def make_case(case_id: str = "RE1-SS/payment_cpu/1", sustained_breach: bool = True) -> TelemetryCase:
    times = np.arange(0.0, 541.0, 1.0)
    error = np.zeros_like(times)
    latency = np.full_like(times, 10.0)
    if sustained_breach:
        for start, end in ((300.0, 315.0), (315.0, 330.0), (345.0, 360.0)):
            error[(times > start) & (times <= end)] = 10.0
    frame = pd.DataFrame(
        {
            TIME_COLUMN: times,
            "front-end_error": error,
            "front-end_workload": np.full_like(times, 100.0),
            "front-end_latency": latency,
            "payment_container-memory-failures-total": np.zeros_like(times),
        }
    )
    return TelemetryCase(
        name=case_id,
        case_id=case_id,
        scenario_id=case_id.rsplit("/", 1)[0],
        system="sock-shop",
        frame=frame,
        incident_time=300.0,
        inject_time=300.0,
        root_label="payment",
        fault_type="cpu",
    )


class FailureOnsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FailureOnsetConfig(min_baseline_bins=8)

    def test_registered_default_boundary_configuration(self) -> None:
        self.assertEqual(self.config.baseline_quantile, 0.995)
        self.assertEqual(self.config.normal_baseline_fraction, 0.5)
        self.assertEqual(self.config.min_baseline_seconds, 120.0)

    def test_sli_selection_prefers_user_facing_indicators(self) -> None:
        selected = select_sli_columns(make_case().frame)
        self.assertEqual(selected, ("front-end_error", "front-end_latency"))

    def test_request_error_counts_are_normalized_by_workload(self) -> None:
        derived = derive_request_failure_rates(make_case().frame)
        self.assertIn("front-end_request_failure_rate", derived)
        self.assertEqual(
            derived.attrs["sli_source_columns"]["front-end_request_failure_rate"],
            ["front-end_error", "front-end_workload"],
        )
        value = derived.loc[
            derived[TIME_COLUMN] == 315.0,
            "front-end_request_failure_rate",
        ].iloc[0]
        self.assertAlmostEqual(float(value), 0.1)

    def test_cumulative_error_and_request_counters_are_differenced(self) -> None:
        frame = pd.DataFrame(
            {
                TIME_COLUMN: [0.0, 15.0, 30.0],
                "frontend-external_errors_total": [0.0, 5.0, 9.0],
                "frontend-external_requests_total": [0.0, 100.0, 200.0],
            }
        )
        derived = derive_request_failure_rates(frame)
        values = derived["frontend-external_request_failure_rate"].to_numpy()
        self.assertTrue(np.isnan(values[0]))
        np.testing.assert_allclose(values[1:], [0.05, 0.04])

    def test_latency_uses_registered_tail_priority_and_floor(self) -> None:
        frame = make_case().frame.copy()
        frame["front-end_latency-50"] = 1.0
        frame["front-end_latency-90"] = 11.0
        frame["front-end_latency-95"] = 12.0
        selected = select_sli_columns(frame, system="sock-shop")
        self.assertIn("front-end_latency-95", selected)
        self.assertNotIn("front-end_latency-90", selected)
        self.assertNotIn("front-end_latency-50", selected)
        result = infer_failure_onset(make_case(), self.config)
        self.assertEqual(result.thresholds["front-end_latency"]["threshold"], 15.0)

    def test_system_entry_priority_prefers_online_boutique_external_sli(self) -> None:
        frame = pd.DataFrame(
            {
                TIME_COLUMN: [0.0, 1.0],
                "frontend-external_error": [0.0, 1.0],
                "frontend-external_load": [100.0, 100.0],
                "frontend_latency-95": [1.0, 2.0],
            }
        )
        derived = derive_request_failure_rates(frame)
        selected = select_sli_columns(derived, system="online-boutique")
        self.assertEqual(selected, ("frontend-external_request_failure_rate",))

    def test_registered_system_does_not_substitute_internal_sli_for_missing_entry(self) -> None:
        frame = pd.DataFrame(
            {
                TIME_COLUMN: [0.0, 15.0],
                "ts-order-service_error": [0.0, 1.0],
                "ts-order-service_workload": [100.0, 100.0],
                "ts-order-service_latency-95": [1.0, 2.0],
            }
        )
        derived = derive_request_failure_rates(frame)
        self.assertEqual(select_sli_columns(derived, system="train-ticket"), ())

    def test_availability_uses_a_lower_boundary(self) -> None:
        case = make_case()
        frame = case.frame[[TIME_COLUMN]].copy()
        frame["front-end_availability"] = 1.0
        frame.loc[
            ((frame[TIME_COLUMN] > 300.0) & (frame[TIME_COLUMN] <= 330.0))
            | ((frame[TIME_COLUMN] > 345.0) & (frame[TIME_COLUMN] <= 360.0)),
            "front-end_availability",
        ] = 0.8
        availability_case = TelemetryCase(
            name=case.name,
            frame=frame,
            case_id=case.case_id,
            scenario_id=case.scenario_id,
            system=case.system,
            inject_time=case.inject_time,
        )
        result = infer_failure_onset(availability_case, self.config)
        self.assertEqual(result.failure_time, 315.0)
        self.assertEqual(result.thresholds["front-end_availability"]["operator"], "<")

    def test_first_sustained_three_of_four_breach_sets_onset_and_confirmation(self) -> None:
        result = infer_failure_onset(make_case(), self.config)
        self.assertTrue(result.event_observed)
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.failure_time, 315.0)
        self.assertEqual(result.confirmation_time, 360.0)
        self.assertEqual([row["breach"] for row in result.confirming_window], [True, True, False, True])
        # Post-injection values never influence the pre-injection threshold.
        rate_boundary = result.thresholds["front-end_request_failure_rate"]
        self.assertEqual(rate_boundary["threshold"], 0.05)
        self.assertEqual(rate_boundary["unit"], "ratio")
        self.assertEqual(rate_boundary["provenance"]["baseline_duration_seconds"], 135.0)
        self.assertTrue(rate_boundary["provenance"]["frozen_before_model_scoring"])
        self.assertEqual(
            rate_boundary["source_columns"],
            ["front-end_error", "front-end_workload"],
        )

    def test_absent_sustained_breach_is_right_censored_at_last_complete_bucket(self) -> None:
        result = infer_failure_onset(make_case(sustained_breach=False), self.config)
        self.assertFalse(result.event_observed)
        self.assertEqual(result.status, "right_censored")
        self.assertEqual(result.failure_time, None)
        self.assertEqual(result.censor_time, 540.0)
        self.assertEqual(result.reason, "no_sustained_post_injection_sli_breach")

    def test_manifest_round_trip_and_application_preserve_censoring(self) -> None:
        observed = make_case()
        censored = make_case("RE1-SS/payment_mem/2", sustained_breach=False)
        manifest = build_failure_onset_manifest([censored, observed], self.config, source_root="synthetic")
        self.assertEqual(manifest["schema"], FAILURE_ONSET_SCHEMA)
        self.assertEqual(manifest["schema_version"], FAILURE_ONSET_SCHEMA_VERSION)
        self.assertEqual(manifest["stage"], "candidate_review")
        self.assertFalse(manifest["ready_for_confirmatory_use"])
        self.assertEqual(manifest["summary"]["candidate_onset_count"], 1)
        self.assertEqual([record["case_id"] for record in manifest["cases"]], sorted([observed.case_id, censored.case_id]))

        with tempfile.TemporaryDirectory() as temporary:
            path = write_failure_onset_manifest(manifest, Path(temporary) / "labels.json")
            loaded = read_failure_onset_manifest(path)
        candidate_onset = next(record for record in loaded["cases"] if record["candidate_event_observed"])
        candidate_censor = next(record for record in loaded["cases"] if not record["candidate_event_observed"])
        self.assertEqual(candidate_onset["audit_status"], "candidate_onset_pending_review")
        self.assertEqual(candidate_censor["audit_status"], "candidate_censor_pending_review")
        self.assertIsNone(candidate_onset["failure_time"])
        self.assertEqual(candidate_onset["candidate_failure_time"], 315.0)
        self.assertFalse(candidate_onset["adjudicated"])
        self.assertEqual(
            candidate_onset["sli"][0]["provenance"]["fit_partition"],
            "earliest_normal_fraction_strictly_pre_injection",
        )
        with self.assertRaisesRegex(ValueError, "pending-review"):
            apply_failure_onset_manifest([observed, censored], loaded)

        decisions = []
        for case, decision, onset in (
            (observed, "onset", 315.0),
            (censored, "no_onset", None),
        ):
            for reviewer_id in ("reviewer-a", "reviewer-b"):
                decisions.append(
                    {
                        "case_id": case.case_id,
                        "reviewer_id": reviewer_id,
                        "decision": decision,
                        "failure_time": onset,
                        "blinded_to_model_predictions": True,
                    }
                )
        finalized = finalize_failure_onset_manifest(loaded, decisions)
        self.assertEqual(finalized["stage"], "reviewed_labels_pending_quality_control")
        self.assertFalse(finalized["ready_for_confirmatory_use"])
        finalized = attach_failure_onset_quality_control(
            finalized,
            {
                "n_reviewed": 2,
                "cohens_kappa": 1.0,
                "onset_difference_median_seconds": 0.0,
                "passes": True,
            },
        )
        self.assertEqual(finalized["stage"], "final_labels")
        self.assertTrue(finalized["ready_for_confirmatory_use"])
        with tempfile.TemporaryDirectory() as temporary:
            path = write_failure_onset_manifest(finalized, Path(temporary) / "labels-final.json")
            protocol_records = load_label_manifest(path, require_confirmatory=True)
        self.assertEqual(len(protocol_records), 2)
        labeled = apply_failure_onset_manifest([observed, censored], finalized)
        self.assertEqual(labeled[0].failure_time, 315.0)
        self.assertEqual(labeled[0].incident_time, 315.0)
        self.assertEqual(labeled[1].failure_time, None)
        self.assertEqual(labeled[1].incident_time, None)
        self.assertEqual(labeled[1].censor_time, 540.0)

    def test_telemetry_ending_before_injection_is_not_a_valid_censor_time(self) -> None:
        case = make_case()
        case = TelemetryCase(
            name=case.name,
            frame=case.frame.loc[case.frame[TIME_COLUMN] <= 270.0],
            inject_time=300.0,
            case_id=case.case_id,
            scenario_id=case.scenario_id,
            system=case.system,
        )
        result = infer_failure_onset(case, self.config)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.reason, "telemetry_ends_before_inject_time")
        self.assertIsNone(result.censor_time)
        self.assertEqual(result.to_record()["audit_status"], "candidate_censor_pending_review")

    def test_reviewer_disagreement_requires_third_reviewer_adjudication(self) -> None:
        case = make_case()
        candidate = build_failure_onset_manifest([case], self.config)
        reviewers = [
            {
                "case_id": case.case_id,
                "reviewer_id": "reviewer-a",
                "decision": "onset",
                "failure_time": 315.0,
                "blinded_to_model_predictions": True,
            },
            {
                "case_id": case.case_id,
                "reviewer_id": "reviewer-b",
                "decision": "no_onset",
                "blinded_to_model_predictions": True,
            },
        ]
        with self.assertRaisesRegex(ValueError, "adjudicator"):
            finalize_failure_onset_manifest(candidate, reviewers)
        finalized = finalize_failure_onset_manifest(
            candidate,
            reviewers
            + [
                {
                    "case_id": case.case_id,
                    "reviewer_id": "reviewer-c",
                    "role": "adjudicator",
                    "decision": "onset",
                    "failure_time": 330.0,
                    "blinded_to_model_predictions": True,
                }
            ],
        )
        record = finalized["cases"][0]
        self.assertTrue(record["adjudicated"])
        self.assertEqual(record["failure_time"], 330.0)
        self.assertEqual(record["audit_status"], "accepted_onset")

    def test_strict_windows_exclude_zero_lead_endpoint(self) -> None:
        case = TelemetryCase(
            name="case",
            frame=pd.DataFrame({TIME_COLUMN: [0.0, 1.0, 2.0], "metric": [0.0, 1.0, 2.0]}),
            incident_time=2.0,
        )
        dataset = build_window_dataset([case], horizons=[1.0], window_size=1, stride=1)
        self.assertEqual(dataset.end_times.tolist(), [0.0, 1.0])
        self.assertEqual(dataset.Y[:, 0].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
