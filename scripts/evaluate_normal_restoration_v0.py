#!/usr/bin/env python3
"""Evaluate training-only normal-state restoration as an RCA attribution rule.

This is deliberately a V0 hypothesis test, not a new trained architecture.  It
loads the existing five-fold LTFP checkpoints and compares three explanations
of the *same* failure predictor: its supervised root head, zero masking, and
candidate-service restoration to a healthy prototype estimated only from the
training split.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.re2 import Re2WindowDataset
from models import LTFPBaseline
from rca.ranking import masking_importance, restoration_importance
from training.engine import choose_device, fit_feature_scaler, seed_everything
from training.metrics import rca_metrics


DATASETS = {
    "ob": ROOT / "artifacts/re2_ob_panel_fair_v1",
    "tt": ROOT / "artifacts/re2_tt_panel_fair_v1",
}


def _checkpoint(dataset: str, fold: int) -> Path:
    return ROOT / (
        f"artifacts/re2_{dataset}_ltfp_deployment_fold{fold}_"
        "fair_30ep_seed42_baseline_v1/best_model.pt"
    )


def _healthy_train_ends(dataset: Re2WindowDataset, train_ends: list[int]) -> list[int]:
    """Select windows whose observed history and next target are all normal.

    ``failure_targets[i]`` describes minute ``i+1``.  Thus labels from
    ``start-1`` through ``end`` cover every input minute and the forecast
    target.  Requiring the preceding index to belong to the same case prevents
    accidental use of a neighbouring case at a boundary.
    """
    healthy = []
    for end in train_ends:
        start = end - dataset.history + 1
        if start == 0 or dataset.case_ids[start - 1] != dataset.case_ids[end]:
            continue
        if not dataset.failure_targets[start - 1 : end + 1].any():
            healthy.append(end)
    return healthy


def _prototype(dataset: Re2WindowDataset, ends: list[int]) -> np.ndarray:
    windows = np.stack(
        [dataset.features[end - dataset.history + 1 : end + 1] for end in ends]
    )
    return np.median(windows, axis=0).astype(np.float32)


def _event_metrics(
    dataset: Re2WindowDataset,
    ends: list[int],
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int | None]:
    rows_by_event: dict[str, list[int]] = {}
    for row, end in enumerate(ends):
        event_id = dataset.event_ids[end]
        if event_id and targets[row].sum() > 0:
            rows_by_event.setdefault(event_id, []).append(row)
    event_targets = []
    event_scores = []
    for event_id, rows in rows_by_event.items():
        group_targets = targets[rows]
        if not np.all(group_targets == group_targets[0]):
            raise ValueError(f"root labels change within event {event_id}")
        event_targets.append(group_targets[0])
        event_scores.append(scores[rows].mean(axis=0))
    metric = rca_metrics(
        np.stack(event_targets),
        np.stack(event_scores),
    )
    metric["positive_events"] = metric.pop("positive_windows")
    return metric


def _score_diagnostics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    truth = targets.astype(bool)
    true_scores = scores[truth]
    return {
        "mean_true_root_risk_drop": float(true_scores.mean()),
        "median_true_root_risk_drop": float(np.median(true_scores)),
        "true_root_positive_drop_fraction": float((true_scores > 0).mean()),
        "all_services_tied_windows": int(
            np.isclose(scores.max(axis=1), scores.min(axis=1)).sum()
        ),
    }


def _run_fold(dataset_name: str, fold: int, device: torch.device, batch_size: int):
    seed_everything(42)
    dataset = Re2WindowDataset(
        DATASETS[dataset_name], history=5, split_policy="deployment", split_fold=fold
    )
    train_ends, _, test_ends = dataset.predefined_splits
    checkpoint_path = _checkpoint(dataset_name, fold)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["services"] != list(dataset.services):
        raise ValueError("checkpoint and dataset service orders differ")
    if checkpoint["feature_columns"] != list(dataset.feature_columns):
        raise ValueError("checkpoint and dataset feature columns differ")

    mean, scale = fit_feature_scaler(dataset, train_ends)
    if not (
        np.allclose(mean, checkpoint["feature_mean"], rtol=1e-5, atol=1e-6)
        and np.allclose(scale, checkpoint["feature_scale"], rtol=1e-5, atol=1e-6)
    ):
        raise ValueError("recomputed train scaler differs from the checkpoint")

    healthy_ends = _healthy_train_ends(dataset, train_ends)
    if not healthy_ends:
        raise ValueError("no fully healthy training windows for the prototype")
    prototype = torch.from_numpy(_prototype(dataset, healthy_ends)).to(device)

    config = checkpoint["model_config"]
    model = LTFPBaseline(
        feature_dim=config["feature_dim"], feature_groups=config["feature_groups"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    attributed_ends = [end for end in test_ends if dataset.node_targets[end].sum() > 0]
    index = {end: row for row, end in enumerate(dataset.sample_indices)}
    loader = DataLoader(
        Subset(dataset, [index[end] for end in attributed_ends]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    collected = {
        "targets": [],
        "learned_head": [],
        "zero_mask": [],
        "normal_restoration": [],
        "normal_restoration_nonnegative": [],
    }
    for batch in loader:
        features = batch["node_features"].to(device)
        adjacency = batch["adjacency"].to(device)
        with torch.no_grad():
            learned = model(features, adjacency)["node_scores"]
        collected["targets"].append(batch["node_targets"].numpy())
        collected["learned_head"].append(learned.cpu().numpy())
        collected["zero_mask"].append(
            masking_importance(model, features, adjacency).cpu().numpy()
        )
        collected["normal_restoration"].append(
            restoration_importance(
                model, features, adjacency, prototype, clamp_min_zero=False
            ).cpu().numpy()
        )
        collected["normal_restoration_nonnegative"].append(
            restoration_importance(
                model, features, adjacency, prototype, clamp_min_zero=True
            ).cpu().numpy()
        )
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    targets = arrays.pop("targets")
    methods = {}
    for name, scores in arrays.items():
        methods[name] = {
            "window": rca_metrics(targets, scores),
            "event": _event_metrics(dataset, attributed_ends, targets, scores),
            "diagnostics": _score_diagnostics(targets, scores),
        }
    return {
        "fold": fold,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "train_windows": len(train_ends),
        "healthy_prototype_windows": len(healthy_ends),
        "attributed_test_windows": len(attributed_ends),
        "attributed_test_events": len(
            {dataset.event_ids[end] for end in attributed_ends if dataset.event_ids[end]}
        ),
        "methods": methods,
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _summarize(folds: list[dict]) -> dict:
    summary = {}
    methods = folds[0]["methods"]
    for method in methods:
        summary[method] = {}
        for level in ("window", "event"):
            summary[method][level] = {}
            for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr"):
                values = [fold["methods"][method][level][metric] for fold in folds]
                summary[method][level][metric] = {**_mean_sd(values), "folds": values}
        for metric in (
            "mean_true_root_risk_drop",
            "median_true_root_risk_drop",
            "true_root_positive_drop_fraction",
            "all_services_tied_windows",
        ):
            values = [fold["methods"][method]["diagnostics"][metric] for fold in folds]
            summary[method].setdefault("diagnostics", {})[metric] = {
                **_mean_sd(values),
                "folds": values,
            }
    return summary


def _markdown(report: dict) -> str:
    lines = [
        "# 正常状态恢复归因 V0 实验",
        "",
        "## 目的",
        "",
        "在不重训故障预测器的条件下，比较同一 LTFP 模型的监督定位头、零值遮蔽和训练集正常原型恢复。该实验只检验‘恢复候选服务后故障风险下降可用于根因排序’这一核心假设。",
        "",
        "正常原型仅由每折训练集内历史与下一分钟均无用户可感知故障的窗口逐位置取中位数得到；测试折和测试标签不参与原型估计。",
        "",
    ]
    for dataset, result in report["datasets"].items():
        lines.extend([
            f"## {dataset}",
            "",
            "下表为五折非加权均值±样本标准差，定位按故障事件聚合（同一事件各正例窗口的服务分数取均值）。",
            "",
            "| 归因方法 | Hit@1 | Hit@3 | Hit@5 | MRR |",
            "|---|---:|---:|---:|---:|",
        ])
        for method, display in (
            ("learned_head", "LTFP监督定位头"),
            ("zero_mask", "零值遮蔽"),
            ("normal_restoration", "正常原型恢复（有符号）"),
            ("normal_restoration_nonnegative", "正常原型恢复（非负）"),
        ):
            metrics = result["summary"][method]["event"]
            values = [
                f'{metrics[key]["mean"]:.3f}±{metrics[key]["sd"]:.3f}'
                for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")
            ]
            lines.append(f"| {display} | " + " | ".join(values) + " |")
        signed = result["summary"]["normal_restoration"]["diagnostics"]
        lines.extend([
            "",
            f'- 每折正常原型训练窗口数：{[fold["healthy_prototype_windows"] for fold in result["folds"]]}。',
            f'- 真根因恢复后风险下降为正的窗口比例：{signed["true_root_positive_drop_fraction"]["mean"]:.3f}±{signed["true_root_positive_drop_fraction"]["sd"]:.3f}。',
            f'- 真根因平均风险下降：{signed["mean_true_root_risk_drop"]["mean"]:.4f}±{signed["mean_true_root_risk_drop"]["sd"]:.4f}。',
            "",
        ])
    lines.extend([
        "## 判定规则",
        "",
        "- 若正常恢复稳定优于零值遮蔽，且真根因恢复多数能降低风险，说明反事实输入比越界置零更合理，可进入条件生成/不确定性版本。",
        "- 若只接近监督定位头，说明它可作为无需额外定位头的归因方案，但还不足以单独构成创新。",
        "- 若不优于零值遮蔽或真根因恢复经常不降风险，应停止堆叠生成模型，先检查预测器是否真正依赖根因服务信号。",
        "",
        "完整逐折结果见 `artifacts/normal_restoration_v0/results.json`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ob", "tt"), default=("ob", "tt"))
    parser.add_argument("--folds", nargs="+", type=int, default=range(5))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds):
        parser.error("folds must be in 0..4")
    device = choose_device(args.device)
    report = {
        "protocol": {
            "task": "past five complete minutes predict next-minute user-visible failure state; attribute the root service",
            "backbone": "existing LTFP five-fold checkpoints; no retraining",
            "normal_reference": "training-only fully healthy windows; per-position median",
            "split": "deployment-group five-fold holdout",
            "device": str(device),
        },
        "datasets": {},
    }
    for dataset in args.datasets:
        folds = []
        for fold in args.folds:
            print(f"[{dataset}] fold {fold}: evaluating", flush=True)
            folds.append(_run_fold(dataset, fold, device, args.batch_size))
        report["datasets"][f"RE2-{dataset.upper()}"] = {
            "folds": folds,
            "summary": _summarize(folds),
        }
    output_dir = ROOT / "artifacts/normal_restoration_v0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    doc_path = ROOT / "docs/2026-11-12-normal-restoration-v0.md"
    doc_path.write_text(_markdown(report), encoding="utf-8")
    print(f"saved {output_path}")
    print(f"saved {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
