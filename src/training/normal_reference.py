"""Fit observation-aware, train-only metric normal references."""

import numpy as np
import torch


def configure_metric_reference(model, dataset, train_ends):
    healthy = []
    for end in train_ends:
        start = end - dataset.history + 1
        if start == 0 or dataset.case_ids[start - 1] != dataset.case_ids[end]:
            continue
        if not dataset.failure_targets[start - 1:end + 1].any():
            healthy.append(end)
    if not healthy:
        raise ValueError("no fully healthy training windows for normal reference")
    windows = np.stack([
        dataset.features[end - dataset.history + 1:end + 1] for end in healthy
    ])
    shape = (dataset.history, len(dataset.services), len(model.metric_value_indices))
    median = np.zeros(shape, dtype=np.float32)
    scale = np.ones(shape, dtype=np.float32)
    missing = 0
    for slot in range(shape[0]):
        for service in range(shape[1]):
            for k, (value_col, mask_col) in enumerate(zip(
                model.metric_value_indices, model.metric_mask_indices
            )):
                values = windows[:, slot, service, value_col]
                if mask_col is not None:
                    values = values[windows[:, slot, service, mask_col] > 0]
                if not len(values):
                    missing += 1
                    continue
                center = np.median(values)
                mad = np.median(np.abs(values - center)) * 1.4826
                std = values.std()
                median[slot, service, k] = center
                scale[slot, service, k] = mad if mad > 1e-3 else (std if std > 1e-3 else 1.0)
    model.configure_normal_reference(torch.from_numpy(median), torch.from_numpy(scale))
    return {"healthy_train_windows": len(healthy), "unobserved_reference_cells": missing}
