from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ltfp.data import TIME_COLUMN, TelemetryCase


@dataclass(frozen=True)
class SyntheticConfig:
    n_cases: int = 120
    n_services: int = 8
    n_steps: int = 160
    step_seconds: int = 60
    incident_step_min: int = 105
    incident_step_max: int = 145
    early_signal_steps: int = 18
    seed: int = 13


FAULT_TYPES = ("cpu", "mem", "delay", "loss")


def make_synthetic_cases(config: SyntheticConfig = SyntheticConfig()) -> list[TelemetryCase]:
    rng = np.random.default_rng(config.seed)
    services = [f"svc{i}" for i in range(config.n_services)]
    edges = [(i, (i + 1) % config.n_services) for i in range(config.n_services)]
    edges += [(i, (i + 2) % config.n_services) for i in range(0, config.n_services, 2)]
    cases = []
    for case_idx in range(config.n_cases):
        root_idx = int(rng.integers(0, config.n_services))
        fault_type = str(rng.choice(FAULT_TYPES))
        incident_step = int(rng.integers(config.incident_step_min, config.incident_step_max + 1))
        frame = _simulate_case(
            rng=rng,
            services=services,
            edges=edges,
            root_idx=root_idx,
            fault_type=fault_type,
            incident_step=incident_step,
            n_steps=config.n_steps,
            step_seconds=config.step_seconds,
            early_signal_steps=config.early_signal_steps,
        )
        incident_time = incident_step * config.step_seconds
        root_service = services[root_idx]
        cases.append(
            TelemetryCase(
                name=f"synthetic_{root_service}_{fault_type}_{case_idx:03d}",
                frame=frame,
                incident_time=float(incident_time),
                root_label=root_service,
                fault_type=fault_type,
                metadata={"edges": edges},
            )
        )
    return cases


def _simulate_case(
    rng: np.random.Generator,
    services: Sequence[str],
    edges: Sequence[tuple[int, int]],
    root_idx: int,
    fault_type: str,
    incident_step: int,
    n_steps: int,
    step_seconds: int,
    early_signal_steps: int,
) -> pd.DataFrame:
    t = np.arange(n_steps)
    data: dict[str, np.ndarray] = {TIME_COLUMN: t * step_seconds}
    seasonal = 0.05 * np.sin(t / 7.0) + 0.03 * np.cos(t / 17.0)

    affected = _affected_services(root_idx, edges, max_hops=2)
    for service_idx, service in enumerate(services):
        distance = affected.get(service_idx)
        propagation_scale = 0.0 if distance is None else 0.65 ** distance
        noise = rng.normal(0, 0.04, size=(n_steps, 5))
        cpu = 0.35 + seasonal + noise[:, 0]
        mem = 0.45 + 0.02 * np.sin(t / 11.0 + service_idx) + noise[:, 1]
        latency = 80 + 7 * seasonal + noise[:, 2] * 15
        error_rate = 0.01 + np.maximum(noise[:, 3], -0.01) * 0.02
        qps = 120 + 10 * np.sin(t / 13.0 + service_idx) + noise[:, 4] * 25

        if propagation_scale > 0:
            start = max(0, incident_step - early_signal_steps - int(distance or 0) * 3)
            ramp = np.clip((t - start) / max(incident_step - start, 1), 0, 1) ** 1.4
            post = np.clip((t - incident_step) / 10, 0, 1)
            if fault_type == "cpu":
                cpu += propagation_scale * (0.35 * ramp + 0.2 * post)
                latency += propagation_scale * (25 * ramp + 35 * post)
            elif fault_type == "mem":
                mem += propagation_scale * (0.30 * ramp + 0.22 * post)
                latency += propagation_scale * (20 * ramp + 30 * post)
            elif fault_type == "delay":
                latency += propagation_scale * (45 * ramp + 65 * post)
                qps -= propagation_scale * (10 * ramp + 20 * post)
            elif fault_type == "loss":
                error_rate += propagation_scale * (0.05 * ramp + 0.12 * post)
                latency += propagation_scale * (15 * ramp + 20 * post)

        data[f"{service}_cpu"] = np.clip(cpu, 0, 1)
        data[f"{service}_mem"] = np.clip(mem, 0, 1)
        data[f"{service}_latency"] = np.clip(latency, 0, None)
        data[f"{service}_error"] = np.clip(error_rate, 0, 1)
        data[f"{service}_qps"] = np.clip(qps, 0, None)

    return pd.DataFrame(data)


def _affected_services(root_idx: int, edges: Sequence[tuple[int, int]], max_hops: int) -> dict[int, int]:
    neighbors: dict[int, list[int]] = {}
    for src, dst in edges:
        neighbors.setdefault(src, []).append(dst)
        neighbors.setdefault(dst, []).append(src)
    distances = {root_idx: 0}
    frontier = [root_idx]
    while frontier:
        current = frontier.pop(0)
        if distances[current] >= max_hops:
            continue
        for nxt in neighbors.get(current, []):
            if nxt in distances:
                continue
            distances[nxt] = distances[current] + 1
            frontier.append(nxt)
    return distances
