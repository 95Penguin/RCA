from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def infer_service_name(metric_name: str) -> str:
    base = metric_name.split("__", 1)[0]
    for sep in ("_", ".", "/", ":"):
        if sep in base:
            return base.split(sep, 1)[0]
    return base


def service_feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feature_names):
        groups[infer_service_name(name)].append(idx)
    return dict(groups)


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer is not fitted")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class GraphFeatureBuilder:
    feature_names: Sequence[str]
    top_k: int = 2
    alpha: float = 0.35

    services_: list[str] | None = None
    groups_: dict[str, list[int]] | None = None
    adjacency_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "GraphFeatureBuilder":
        groups = service_feature_groups(self.feature_names)
        services = sorted(groups)
        service_matrix = self._service_matrix_from_groups(X, groups, services)
        if service_matrix.shape[1] <= 1:
            adjacency = np.zeros((service_matrix.shape[1], service_matrix.shape[1]))
        else:
            corr = np.corrcoef(service_matrix, rowvar=False)
            corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 0.0)
            adjacency = np.zeros_like(corr)
            for i in range(corr.shape[0]):
                if self.top_k <= 0:
                    continue
                neighbors = np.argsort(corr[i])[-self.top_k :]
                adjacency[i, neighbors] = corr[i, neighbors]
            row_sum = adjacency.sum(axis=1, keepdims=True)
            adjacency = np.divide(adjacency, row_sum, out=np.zeros_like(adjacency), where=row_sum > 0)
        self.services_ = services
        self.groups_ = groups
        self.adjacency_ = adjacency
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.services_ is None or self.groups_ is None or self.adjacency_ is None:
            raise RuntimeError("GraphFeatureBuilder is not fitted")
        service_matrix = self._service_matrix_from_groups(X, self.groups_, self.services_)
        smoothed = (1.0 - self.alpha) * service_matrix + self.alpha * (service_matrix @ self.adjacency_)
        max_score = smoothed.max(axis=1, keepdims=True)
        mean_score = smoothed.mean(axis=1, keepdims=True)
        return np.hstack([smoothed, max_score, mean_score])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def _service_matrix_from_groups(
        X: np.ndarray,
        groups: dict[str, list[int]],
        services: Sequence[str],
    ) -> np.ndarray:
        values = []
        abs_x = np.abs(X)
        for service in services:
            idx = groups[service]
            values.append(abs_x[:, idx].max(axis=1))
        return np.vstack(values).T if values else np.zeros((len(X), 0))
