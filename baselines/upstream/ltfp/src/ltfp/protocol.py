"""Auditable manifests and split utilities for the failure-onset protocol.

The legacy experiments inferred identity, candidates, and preprocessing state
from whatever cases happened to be loaded.  This module makes those decisions
explicit files that can be archived with every result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


LABEL_MANIFEST_REQUIRED = {
    "case_id",
    "system",
    "scenario_id",
    "repetition",
    "inject_time",
    "failure_time",
    "censor_time",
    "sli",
    "audit_status",
}
SPLIT_MANIFEST_REQUIRED = {"case_id", "scenario_id", "fold"}


@dataclass(frozen=True)
class TopologyManifest:
    version: str
    system: str
    services: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    entry_services: tuple[str, ...]
    aliases: Mapping[str, str]
    sources: tuple[Mapping[str, str], ...]
    path: str


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, dtype={"case_id": str, "scenario_id": str, "repetition": str})
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("cases", payload))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must contain a record list: {source}")
    return pd.DataFrame(payload)


def load_label_manifest(
    path: str | Path,
    *,
    require_confirmatory: bool = False,
) -> pd.DataFrame:
    source = Path(path)
    metadata = None
    if source.suffix.lower() != ".csv":
        raw = json.loads(source.read_text(encoding="utf-8"))
        metadata = raw if isinstance(raw, dict) else None
    if require_confirmatory:
        if metadata is None:
            raise ValueError("confirmatory labels require a versioned JSON manifest")
        if metadata.get("schema") != "ltfp.failure-onset-labels":
            raise ValueError("confirmatory labels require the failure-onset manifest schema")
        if metadata.get("stage") != "final_labels" or not metadata.get(
            "ready_for_confirmatory_use", False
        ):
            raise ValueError("confirmatory labels must be finalized and ready_for_confirmatory_use")
        quality = metadata.get("quality_control_audit")
        if not isinstance(quality, Mapping) or not quality.get("passes", False):
            raise ValueError("confirmatory labels require a passing quality_control_audit")
    frame = _read_table(source)
    missing = sorted(LABEL_MANIFEST_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"label manifest is missing columns: {', '.join(missing)}")
    if frame["case_id"].astype(str).duplicated().any():
        duplicates = sorted(frame.loc[frame["case_id"].astype(str).duplicated(), "case_id"].astype(str).unique())
        raise ValueError(f"label manifest case_id values are not unique: {duplicates[:5]}")
    for column in ("inject_time", "failure_time", "censor_time"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    observed = frame["failure_time"].notna()
    invalid_order = observed & (
        (frame["failure_time"] <= frame["inject_time"]) | (frame["failure_time"] > frame["censor_time"])
    )
    if invalid_order.any():
        bad = frame.loc[invalid_order, "case_id"].astype(str).tolist()
        raise ValueError(f"failure/injection/censor ordering is invalid for: {bad[:5]}")
    if (frame["censor_time"] < frame["inject_time"]).any():
        bad = frame.loc[frame["censor_time"] < frame["inject_time"], "case_id"].astype(str).tolist()
        raise ValueError(f"censor_time precedes inject_time for: {bad[:5]}")
    if require_confirmatory:
        if "review_status" not in frame or not frame["review_status"].astype(str).str.lower().eq(
            "complete"
        ).all():
            raise ValueError("every confirmatory label must have review_status=complete")
        allowed_audit = frame["audit_status"].astype(str).str.lower().str.startswith(
            ("accepted_", "censored_")
        )
        if not allowed_audit.all():
            raise ValueError("confirmatory labels contain a non-final audit_status")
    return frame.reset_index(drop=True)


def make_scenario_folds(
    label_records: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    """Assign complete scenarios to balanced deterministic folds.

    Balance is greedy over case count and system count.  No label outcome or
    test performance participates in the assignment.
    """

    if n_folds < 3:
        raise ValueError("scenario-disjoint train/validation/test needs at least three folds")
    required = {"case_id", "scenario_id", "system"}
    missing = required - set(label_records.columns)
    if missing:
        raise ValueError(f"records are missing split fields: {sorted(missing)}")
    grouped = []
    for scenario, group in label_records.groupby("scenario_id", sort=True):
        systems = tuple(sorted(group["system"].astype(str).unique()))
        if len(systems) != 1:
            raise ValueError(f"scenario_id {scenario!r} spans multiple systems: {systems}")
        grouped.append((str(scenario), systems[0], int(len(group))))

    rng = np.random.default_rng(seed)
    jitter = {scenario: float(rng.random()) for scenario, _, _ in grouped}
    grouped.sort(key=lambda item: (-item[2], item[1], jitter[item[0]], item[0]))
    fold_sizes = np.zeros(n_folds, dtype=int)
    system_sizes: list[Counter] = [Counter() for _ in range(n_folds)]
    assignment: dict[str, int] = {}
    for scenario, system, count in grouped:
        scores = [
            (system_sizes[fold][system], int(fold_sizes[fold]), fold)
            for fold in range(n_folds)
        ]
        chosen = min(scores)[-1]
        assignment[scenario] = int(chosen)
        fold_sizes[chosen] += count
        system_sizes[chosen][system] += count

    output = label_records[["case_id", "scenario_id", "system"]].copy()
    output["fold"] = output["scenario_id"].astype(str).map(assignment).astype(int)
    output["split_seed"] = int(seed)
    output["n_folds"] = int(n_folds)
    output = output.sort_values(["fold", "system", "scenario_id", "case_id"]).reset_index(drop=True)
    validate_split_manifest(output, n_folds=n_folds)
    return output


def load_split_manifest(path: str | Path, label_records: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _read_table(path)
    missing = sorted(SPLIT_MANIFEST_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"split manifest is missing columns: {', '.join(missing)}")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    n_folds = int(frame["n_folds"].iloc[0]) if "n_folds" in frame and len(frame) else int(frame["fold"].max() + 1)
    validate_split_manifest(frame, n_folds=n_folds)
    if label_records is not None:
        labels = set(label_records["case_id"].astype(str))
        splits = set(frame["case_id"].astype(str))
        if labels != splits:
            raise ValueError(
                "label/split case sets differ: "
                f"missing_from_split={sorted(labels - splits)[:5]}, extra_in_split={sorted(splits - labels)[:5]}"
            )
    return frame.reset_index(drop=True)


def validate_split_manifest(frame: pd.DataFrame, n_folds: int = 5) -> None:
    if frame["case_id"].astype(str).duplicated().any():
        raise ValueError("each physical case must occur exactly once in the split manifest")
    scenario_folds = frame.groupby(frame["scenario_id"].astype(str))["fold"].nunique()
    if (scenario_folds != 1).any():
        raise ValueError("one or more scenarios cross fold boundaries")
    folds = set(frame["fold"].astype(int))
    expected = set(range(int(n_folds)))
    if folds != expected:
        raise ValueError(f"fold IDs must be exactly {sorted(expected)}, found {sorted(folds)}")


def fold_roles(split_manifest: pd.DataFrame, test_fold: int, n_folds: int | None = None) -> dict[str, set[str]]:
    if n_folds is None:
        n_folds = int(split_manifest["fold"].max()) + 1
    test_fold = int(test_fold)
    validation_fold = (test_fold + 1) % int(n_folds)
    roles = {
        "train": set(
            split_manifest.loc[~split_manifest["fold"].isin([test_fold, validation_fold]), "case_id"].astype(str)
        ),
        "validation": set(split_manifest.loc[split_manifest["fold"] == validation_fold, "case_id"].astype(str)),
        "test": set(split_manifest.loc[split_manifest["fold"] == test_fold, "case_id"].astype(str)),
    }
    if (roles["train"] & roles["validation"]) or (roles["train"] & roles["test"]) or (
        roles["validation"] & roles["test"]
    ):
        raise AssertionError("case roles overlap")
    scenario_by_case = dict(
        zip(split_manifest["case_id"].astype(str), split_manifest["scenario_id"].astype(str))
    )
    role_scenarios = {role: {scenario_by_case[item] for item in ids} for role, ids in roles.items()}
    if (role_scenarios["train"] & role_scenarios["validation"]) or (
        role_scenarios["train"] & role_scenarios["test"]
    ) or (role_scenarios["validation"] & role_scenarios["test"]):
        raise AssertionError("scenario roles overlap")
    return roles


def write_manifest_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def load_topology_manifest(path: str | Path, expected_system: str | None = None) -> TopologyManifest:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {"version", "system", "services", "edges", "entry_services", "sources"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"topology manifest is missing fields: {sorted(missing)}")
    services = tuple(str(item) for item in payload["services"])
    if len(services) != len(set(services)):
        raise ValueError("topology services must be unique")
    service_set = set(services)
    edges = tuple((str(edge[0]), str(edge[1])) for edge in payload["edges"])
    unknown = sorted({node for edge in edges for node in edge} - service_set)
    if unknown:
        raise ValueError(f"topology edges reference unknown services: {unknown}")
    entry_services = tuple(str(item) for item in payload["entry_services"])
    if set(entry_services) - service_set:
        raise ValueError("entry_services must be members of services")
    aliases = {str(key): str(value) for key, value in payload.get("aliases", {}).items()}
    if set(aliases.values()) - service_set:
        raise ValueError("topology alias targets must be members of services")
    system = str(payload["system"])
    if expected_system is not None and normalize_system(expected_system) != normalize_system(system):
        raise ValueError(f"topology system {system!r} does not match {expected_system!r}")
    return TopologyManifest(
        version=str(payload["version"]),
        system=system,
        services=services,
        edges=edges,
        entry_services=entry_services,
        aliases=aliases,
        sources=tuple(payload["sources"]),
        path=str(source),
    )


def topology_adjacency(manifest: TopologyManifest, bidirectional: bool = True) -> np.ndarray:
    index = {service: idx for idx, service in enumerate(manifest.services)}
    adjacency = np.zeros((len(index), len(index)), dtype=np.float32)
    for source, target in manifest.edges:
        adjacency[index[source], index[target]] = 1.0
        if bidirectional:
            adjacency[index[target], index[source]] = 1.0
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def combine_topology_and_correlation(
    dependency: np.ndarray,
    service_signal: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Union static dependency edges with symmetric train-only associations."""

    dependency = np.asarray(dependency, dtype=np.float32)
    service_signal = np.asarray(service_signal, dtype=float)
    if service_signal.ndim != 2 or service_signal.shape[1] != dependency.shape[0]:
        raise ValueError("service_signal must have shape [train observations, topology services]")
    if len(service_signal) < 2 or dependency.shape[0] < 2 or top_k <= 0:
        correlation = np.eye(dependency.shape[0], dtype=np.float32)
        return np.maximum(dependency, correlation), correlation
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_full = np.abs(np.corrcoef(service_signal, rowvar=False))
    correlation_full = np.nan_to_num(correlation_full, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation_full, 0.0)
    directed = np.zeros_like(correlation_full)
    for idx in range(len(correlation_full)):
        neighbors = np.argsort(correlation_full[idx])[-min(int(top_k), len(correlation_full) - 1) :]
        directed[idx, neighbors] = correlation_full[idx, neighbors]
    correlation = np.maximum(directed, directed.T).astype(np.float32)
    np.fill_diagonal(correlation, 1.0)
    return np.maximum(dependency, correlation).astype(np.float32), correlation


def normalize_system(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "ob": "online-boutique",
        "onlineboutique": "online-boutique",
        "ss": "sock-shop",
        "sockshop": "sock-shop",
        "tt": "train-ticket",
        "trainticket": "train-ticket",
    }
    return aliases.get(normalized, normalized)


def label_quality_gate(
    label_records: pd.DataFrame,
    minimum_subset_onset_fraction: float = 0.60,
    minimum_passing_subsets: int = 4,
) -> dict:
    """Apply the fixed automated portion of the onset quality gate."""

    if "subset" in label_records:
        subset_values = label_records["subset"].astype(str)
    elif "dataset" in label_records:
        subset_values = label_records["dataset"].astype(str)
    elif "case_id" in label_records:
        subset_values = label_records["case_id"].astype(str).str.split("/", n=1).str[0]
    else:
        raise ValueError("label quality gate needs subset, dataset, or case_id")
    working = label_records.assign(_derived_subset=subset_values)
    rows = []
    for subset, group in working.groupby("_derived_subset", sort=True):
        fraction = float(group["failure_time"].notna().mean())
        rows.append(
            {
                "subset": str(subset),
                "n_cases": int(len(group)),
                "n_observed": int(group["failure_time"].notna().sum()),
                "observed_fraction": fraction,
                "passes_onset_fraction": bool(fraction >= minimum_subset_onset_fraction),
            }
        )
    passing = sum(row["passes_onset_fraction"] for row in rows)
    return {
        "subsets": rows,
        "minimum_subset_onset_fraction": float(minimum_subset_onset_fraction),
        "minimum_passing_subsets": int(minimum_passing_subsets),
        "n_passing_subsets": int(passing),
        "route_status": "eligible_for_human_audit" if passing >= minimum_passing_subsets else "fallback_to_proxy",
        "human_audit_required": True,
    }


def check_horizon_support(
    train_labels: np.ndarray,
    horizons: Sequence[float],
    train_case_ids: Sequence[str],
    minimum_prevalence: float = 0.05,
    maximum_prevalence: float = 0.80,
    minimum_positive_events: int = 20,
) -> list[dict]:
    labels = np.asarray(train_labels, dtype=int)
    case_ids = np.asarray(train_case_ids, dtype=str)
    records = []
    for idx, horizon in enumerate(horizons):
        positives = labels[:, idx] > 0
        positive_events = len(np.unique(case_ids[positives]))
        prevalence = float(positives.mean()) if len(positives) else 0.0
        supported = minimum_prevalence <= prevalence <= maximum_prevalence and positive_events >= minimum_positive_events
        records.append(
            {
                "horizon": float(horizon),
                "prevalence": prevalence,
                "positive_events": int(positive_events),
                "supported": bool(supported),
            }
        )
    return records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(paths: Iterable[str | Path]) -> dict:
    """Hash executable source files even when the code tree is not a git repo."""

    files: list[tuple[str, Path]] = []
    allowed_suffixes = {".py", ".sh"}
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        if path.is_file() and path.suffix in allowed_suffixes:
            files.append((f"source-{index}/{path.name}", path))
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix in allowed_suffixes and "__pycache__" not in child.parts:
                    files.append((f"source-{index}/{child.relative_to(path).as_posix()}", child))
    digest = hashlib.sha256()
    per_file = {}
    for label, path in sorted(files):
        file_hash = sha256_file(path)
        per_file[label] = file_hash
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"combined_sha256": digest.hexdigest(), "files": per_file}


def runtime_metadata(
    command: Sequence[str],
    manifest_paths: Iterable[str | Path] = (),
    repository_root: str | Path | None = None,
    source_paths: Iterable[str | Path] = (),
) -> dict:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "sklearn", "torch"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[package] = None
    git_commit = None
    git_dirty = None
    if repository_root is not None:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repository_root), text=True, stderr=subprocess.DEVNULL
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=str(repository_root),
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
            git_dirty = None
    manifests = {
        str(Path(path)): sha256_file(path)
        for path in manifest_paths
        if path is not None and Path(path).exists()
    }
    return {
        "command": list(command),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": packages,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "manifest_sha256": manifests,
        "source_tree_sha256": sha256_source_tree(source_paths),
    }


def summarize_fold_composition(split_manifest: pd.DataFrame) -> list[dict]:
    output = []
    for fold, group in split_manifest.groupby("fold", sort=True):
        output.append(
            {
                "fold": int(fold),
                "n_cases": int(len(group)),
                "n_scenarios": int(group["scenario_id"].astype(str).nunique()),
                "systems": dict(Counter(group.get("system", pd.Series(dtype=str)).astype(str))),
            }
        )
    return output


def assert_no_role_overlap(role_by_case: Mapping[str, str], scenario_by_case: Mapping[str, str]) -> None:
    """Reusable integration assertion for model and data adapters."""

    cases_by_role: dict[str, set[str]] = defaultdict(set)
    scenarios_by_role: dict[str, set[str]] = defaultdict(set)
    for case_id, role in role_by_case.items():
        cases_by_role[str(role)].add(str(case_id))
        scenarios_by_role[str(role)].add(str(scenario_by_case[case_id]))
    roles = sorted(cases_by_role)
    for idx, left in enumerate(roles):
        for right in roles[idx + 1 :]:
            if cases_by_role[left] & cases_by_role[right]:
                raise AssertionError(f"case overlap between {left} and {right}")
            if scenarios_by_role[left] & scenarios_by_role[right]:
                raise AssertionError(f"scenario overlap between {left} and {right}")
