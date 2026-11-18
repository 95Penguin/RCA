"""Use the project's case/split logic; expose inputs without future telemetry."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset
from models.factory import named_feature_groups
from training.engine import fit_feature_scaler


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HistoricalWindows(Dataset):
    """Never calls the source __getitem__, which also exposes next_node_features."""

    def __init__(self, source, ends):
        self.source = source
        self.ends = list(ends)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        d = self.source
        end = self.ends[index]
        start = end - d.history + 1
        adjacency = d.adjacency[start:end + 1] if d.adjacency.ndim == 3 else d.adjacency
        return {
            "x": torch.from_numpy(d.features[start:end + 1].copy()),
            "graph": adjacency.clone(),
            "y": torch.tensor(int(d.failure_targets[end])),
            "root": torch.from_numpy(d.node_targets[end].copy()),
            "root_valid": torch.tensor(d._root_label_valid.get(d.timestamps[end], bool(d.node_targets[end].any()))),
            "end": torch.tensor(end),
        }


class PreparedData:
    def __init__(self, directory: str | Path, fold: int, history: int = 5):
        self.directory = Path(directory)
        self.dataset = Re2WindowDataset(self.directory, history=history,
                                       split_policy="deployment", split_fold=fold)
        d = self.dataset
        self.groups = named_feature_groups(d.feature_columns)
        self.train, self.validation, self.test = map(list, d.predefined_splits)
        self.mean, self.scale = fit_feature_scaler(d, self.train)
        self.input_fit_indices = sorted({i for end in self.train for i in range(end-history+1, end+1)})
        self.splits = {"train": self.train, "validation": self.validation, "test": self.test}
        # Current business labels are used only to select normal TRAINING observations
        # for the auto-regressor; inference receives none of these labels.
        current = {}
        for stamp, target in d.label_target_times.items():
            if d._label_valid.get(stamp, False):
                current[target] = int(d.failure_targets[d._time_indices[stamp]])
        self.normal_train = [end for end in self.train
                             if all(current.get(d.timestamps[i]) == 0
                                    for i in range(end-history+1, end+1))]

    def windows(self, ends):
        return HistoricalWindows(self.dataset, ends)

    def identity(self):
        d = self.dataset
        records = []
        for split, ends in self.splits.items():
            for end in ends:
                records.append({"split": split, "case_id": d.case_ids[end],
                                "input_minute": d.timestamps[end],
                                "target_minute": d.label_target_times[d.timestamps[end]]})
        hashes = {name: file_hash(self.directory / name) for name in (
            "feature_schema.json", "label_schema.json", "next_minute_labels.csv",
            "case_groups.csv", "service_features_1m.csv", "service_adjacency_1m.npz")}
        return {"files_sha256": hashes, "split_report": d.split_report,
                "services": list(d.services), "feature_columns": list(d.feature_columns),
                "windows": records,
                "split_manifest_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
