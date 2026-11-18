#!/usr/bin/env python3
"""Summarize future-telemetry auxiliary learning against the LTFP baseline."""

from __future__ import annotations

import json
from pathlib import Path

from summarize_transition_experiment import BASELINE, _paired_analysis


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
FORECAST = (
    "re2_{dataset}_ltfp_forecast_deployment_fold{fold}_first_visible_1m_"
    "fair_30ep_seed42_v1"
)


def _telemetry(dataset: str) -> dict[str, float]:
    model = []
    persistence = []
    for fold in range(5):
        path = ROOT / "artifacts" / FORECAST.format(dataset=dataset, fold=fold) / "metrics.json"
        values = json.loads(path.read_text(encoding="utf-8"))["telemetry_forecast"]
        model.append(values["model_rmse"])
        persistence.append(values["persistence_rmse"])
    return {
        "model_rmse_mean": sum(model) / len(model),
        "persistence_rmse_mean": sum(persistence) / len(persistence),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# 未来遥测辅助预测最小实验",
        "",
        f"结论：**{report['decision']['status']}**。",
        "",
        "固定LTFP编码器协议、first-visible 1分钟标签、deployment五折、30 epoch、seed 42，",
        "加入下一分钟服务遥测预测和0.2权重的masked MSE。",
        "",
        "| 数据集 | LTFP AP | Forecast AP | ΔAP | LTFP Hit@1 | Forecast Hit@1 | ΔHit@1 | Forecast RMSE | Persistence RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["datasets"].items():
        ap = values["fold_summary"]["ap"]
        hit1 = values["fold_summary"]["hit1"]
        telemetry = values["telemetry"]
        lines.append(
            f"| RE2-{name.upper()} | {ap['baseline_mean']:.3f} | "
            f"{ap['transition_mean']:.3f} | {ap['paired_delta']:+.3f} | "
            f"{hit1['baseline_mean']:.3f} | {hit1['transition_mean']:.3f} | "
            f"{hit1['paired_delta']:+.3f} | {telemetry['model_rmse_mean']:.3f} | "
            f"{telemetry['persistence_rmse_mean']:.3f} |"
        )
    lines.extend(["", "## 配对case-bootstrap", ""])
    for name, values in report["datasets"].items():
        interval = values["paired_case_bootstrap_95pct_ci"]
        lines.append(
            f"- RE2-{name.upper()}：pooled ΔAP={values['pooled']['ap_delta']:+.3f}，"
            f"95% CI [{interval['ap_delta'][0]:+.3f}, {interval['ap_delta'][1]:+.3f}]；"
            f"ΔHit@1={values['pooled']['hit1_delta']:+.3f}。"
        )
    lines.extend([
        "",
        "## 判定边界",
        "",
        "只有当至少一个数据集五折平均AP提升0.03、另一个下降不超过0.01且RCA不退化，",
        "才进入条件正常反事实rollout。遥测预测还必须与persistence forecast比较，",
        "否则只能称为辅助正则化，不能声称已经学到更准确的未来系统状态。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    datasets = {}
    for name in DATASETS:
        values = _paired_analysis(
            name,
            baseline_pattern=BASELINE,
            candidate_pattern=FORECAST,
        )
        values["telemetry"] = _telemetry(name)
        datasets[name] = values
    gate = any(
        values["fold_summary"]["ap"]["paired_delta"] >= 0.03
        for values in datasets.values()
    ) and all(
        values["fold_summary"]["ap"]["paired_delta"] >= -0.01
        and values["fold_summary"]["hit1"]["paired_delta"] >= -0.01
        for values in datasets.values()
    )
    forecast_beats_persistence = all(
        values["telemetry"]["model_rmse_mean"]
        < values["telemetry"]["persistence_rmse_mean"]
        for values in datasets.values()
    )
    status = (
        "GO: 可进入反事实rollout"
        if gate and forecast_beats_persistence
        else (
            "PARTIAL: 指标有收益但未来遥测预测尚未超过persistence"
            if gate
            else "NO-GO for the current forecast-auxiliary design"
        )
    )
    report = {
        "protocol": {
            "task": "first-visible failure within (t,t+1m]",
            "split": "deployment five-fold",
            "epochs": 30,
            "seed": 42,
            "telemetry_loss_weight": 0.2,
            "bootstrap": "5000 paired physical-case resamples stratified by fold",
        },
        "datasets": datasets,
        "decision": {
            "status": status,
            "metric_gate_passed": gate,
            "forecast_beats_persistence": forecast_beats_persistence,
        },
    }
    artifact = ROOT / "artifacts" / "ltfp_forecast_experiment_2026-11-14.json"
    document = ROOT / "docs" / "2026-11-14-ltfp-forecast-experiment.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
