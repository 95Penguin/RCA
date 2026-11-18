from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ltfp.data import TelemetryCase
from ltfp.dataset_v2 import (
    build_failure_windows,
    feature_exclusions_from_manifest,
    fit_feature_schema,
    standardize_train_only,
)
from ltfp.protocol import TopologyManifest


class FailureDatasetTests(unittest.TestCase):
    def topology(self) -> TopologyManifest:
        return TopologyManifest(
            version="test-v1",
            system="sock-shop",
            services=("front-end", "orders"),
            edges=(("front-end", "orders"),),
            entry_services=("front-end",),
            aliases={},
            sources=(),
            path="memory",
        )

    def case(self, case_id: str, failure_time=180.0, censor_time=240.0, include_log=True):
        times = np.arange(15.0, 255.0, 15.0)
        payload = {
            "time": times,
            "front-end_latency-95": np.linspace(1.0, 2.0, len(times)),
            "orders_cpu": np.linspace(0.0, 1.0, len(times)),
        }
        if include_log:
            payload["orders_log_orders_1"] = np.arange(len(times), dtype=float)
        return TelemetryCase(
            name=case_id,
            case_id=case_id,
            scenario_id=case_id.rsplit("/", 1)[0],
            system="sock-shop",
            frame=pd.DataFrame(payload),
            inject_time=90.0,
            failure_time=failure_time,
            censor_time=censor_time,
            root_label="orders",
        )

    def test_schema_is_fitted_only_from_supplied_train_cases(self):
        train = self.case("RE1-SS/orders_cpu/1", include_log=False)
        held_out = self.case("RE1-SS/orders_cpu/2", include_log=True)
        schema = fit_feature_schema([train], self.topology(), max_features=180)
        names = {item.name for item in schema.features}
        self.assertNotIn("orders_log_orders_1", names)
        self.assertNotIn(held_out.case_id, schema.fitted_case_ids)

    def test_post_failure_windows_are_excluded_and_labels_are_strict(self):
        case = self.case("RE1-SS/orders_cpu/1")
        schema = fit_feature_schema([case], self.topology())
        dataset = build_failure_windows(
            [case],
            schema,
            self.topology(),
            horizons=(30.0, 60.0),
            window_size=2,
            stride=1,
            grid_seconds=15,
        )
        self.assertTrue(np.all(dataset.end_times < 180.0))
        at_150 = np.flatnonzero(dataset.end_times == 150.0)[0]
        np.testing.assert_array_equal(dataset.Y[at_150], [1.0, 1.0])
        at_90 = np.flatnonzero(dataset.end_times == 90.0)[0]
        np.testing.assert_array_equal(dataset.Y[at_90], [0.0, 0.0])
        before_injection = np.flatnonzero(dataset.end_times == 75.0)[0]
        np.testing.assert_array_equal(dataset.Y[before_injection], [0.0, 0.0])

    def test_censored_horizons_are_masked_near_censor_time(self):
        case = self.case("RE1-SS/orders_cpu/1", failure_time=None, censor_time=210.0)
        schema = fit_feature_schema([case], self.topology())
        dataset = build_failure_windows(
            [case],
            schema,
            self.topology(),
            horizons=(30.0, 60.0),
            window_size=2,
            stride=1,
            grid_seconds=15,
            require_all_horizons_observed=False,
        )
        idx = np.flatnonzero(dataset.end_times == 180.0)[0]
        np.testing.assert_array_equal(dataset.label_mask[idx], [True, False])
        self.assertTrue(np.all(dataset.Y[idx] == 0))

    def test_standardization_keeps_missing_modalities_zero(self):
        first = self.case("RE1-SS/orders_cpu/1", include_log=True)
        second = self.case("RE1-SS/orders_cpu/2", include_log=False)
        schema = fit_feature_schema([first], self.topology())
        dataset = build_failure_windows(
            [first, second], schema, self.topology(), horizons=(60.0,), window_size=2, stride=2
        )
        first_indices = np.flatnonzero(np.asarray(dataset.case_ids) == first.case_id)
        X, _, _ = standardize_train_only(dataset, first_indices)
        second_indices = np.flatnonzero(np.asarray(dataset.case_ids) == second.case_id)
        log_modality = dataset.schema.modality_names.index("log")
        self.assertTrue(np.all(~dataset.availability_mask[second_indices, :, log_modality]))
        log_offset = sum(dataset.schema.modality_dims[:log_modality])
        log_width = dataset.schema.modality_dims[log_modality]
        self.assertTrue(np.all(X[second_indices, :, :, log_offset : log_offset + log_width] == 0.0))

    def test_target_sli_ablation_excludes_derived_ratio_sources(self):
        records = [
            {
                "sli": [
                    {
                        "name": "front-end_request_failure_rate",
                        "source_columns": ["front-end_error", "front-end_workload"],
                    },
                    {"name": "front-end_latency-95"},
                ]
            }
        ]
        self.assertEqual(
            feature_exclusions_from_manifest(records),
            (
                "front-end_error",
                "front-end_latency-95",
                "front-end_request_failure_rate",
                "front-end_workload",
            ),
        )


if __name__ == "__main__":
    unittest.main()
