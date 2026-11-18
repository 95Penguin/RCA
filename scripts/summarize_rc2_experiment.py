#!/usr/bin/env python3
"""Compare RC2 service-risk aggregation with the state-task LTFP baseline."""

from __future__ import annotations

import json
from pathlib import Path

from summarize_transition_experiment import _paired_analysis


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ob", "tt")
BASELINE = "re2_{dataset}_ltfp_deployment_fold{fold}_fair_30ep_seed42_baseline_v1"
RC2 = "re2_{dataset}_ltfp_rc2_deployment_fold{fold}_fair_30ep_seed42_v1"


def _display(value: float) -> str:
    return f"{value:+.3f}"


def _markdown(report: dict) -> str:
    lines = [
        "# RC²-Net根因一致风险分解最小实验",
        "",
        f"结论：**{report['decision']['status']}**。",
        "",
        "比较保持LTFP编码器、state 1分钟标签、deployment五折、30 epoch、seed 42",
        "以及相同的0.5权重listwise RCA loss。RC²-Net仅将独立预测/定位头替换为",
        "共享服务致障风险及其归一化log-sum-exp系统聚合。",
        "",
        "| 数据集 | LTFP AP | RC² AP | ΔAP | LTFP F1 | RC² F1 | LTFP Hit@1 | RC² Hit@1 | ΔHit@1 | LTFP MRR | RC² MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["datasets"].items():
        summary = values["fold_summary"]
        lines.append(
            f"| RE2-{name.upper()} | {summary['ap']['baseline_mean']:.3f} | "
            f"{summary['ap']['transition_mean']:.3f} | {_display(summary['ap']['paired_delta'])} | "
            f"{summary['f1']['baseline_mean']:.3f} | {summary['f1']['transition_mean']:.3f} | "
            f"{summary['hit1']['baseline_mean']:.3f} | {summary['hit1']['transition_mean']:.3f} | "
            f"{_display(summary['hit1']['paired_delta'])} | "
            f"{summary['mrr']['baseline_mean']:.3f} | {summary['mrr']['transition_mean']:.3f} |"
        )
    lines.extend([
        "", "## 配对case-bootstrap", "",
        "| 数据集 | Pooled ΔAP | ΔAP 95% CI | Pooled ΔHit@1 | ΔHit@1 95% CI |",
        "|---|---:|---|---:|---|",
    ])
    for name, values in report["datasets"].items():
        pooled = values["pooled"]
        intervals = values["paired_case_bootstrap_95pct_ci"]
        lines.append(
            f"| RE2-{name.upper()} | {_display(pooled['ap_delta'])} | "
            f"[{intervals['ap_delta'][0]:+.3f}, {intervals['ap_delta'][1]:+.3f}] | "
            f"{_display(pooled['hit1_delta'])} | "
            f"[{intervals['hit1_delta'][0]:+.3f}, {intervals['hit1_delta'][1]:+.3f}] |"
        )
    lines.extend([
        "", "## 预注册判定边界", "",
        "至少一个数据集的五折平均AP或Hit@1提升0.03；同时两个数据集的AP和Hit@1",
        "均不得下降超过0.01。通过后才加入故障类型风险维度和多随机种子实验。", "",
    ])
    return "\n".join(lines)


def main() -> int:
    datasets = {
        name: _paired_analysis(name, baseline_pattern=BASELINE, candidate_pattern=RC2)
        for name in DATASETS
    }
    material_gain = any(
        values["fold_summary"][metric]["paired_delta"] >= 0.03
        for values in datasets.values() for metric in ("ap", "hit1")
    )
    no_material_regression = all(
        values["fold_summary"][metric]["paired_delta"] >= -0.01
        for values in datasets.values() for metric in ("ap", "hit1")
    )
    gate = material_gain and no_material_regression
    report = {
        "protocol": {
            "task": "failure state and root service at exactly t+1m",
            "split": "deployment five-fold", "epochs": 30, "seed": 42,
            "node_bce_weight": 0.0, "root_ranking_weight": 0.5,
            "bootstrap": "5000 paired physical-case resamples stratified by fold",
        },
        "datasets": datasets,
        "decision": {
            "status": (
                "GO: add fault-type structured risk and multi-seed validation"
                if gate else "NO-GO for the current root-cause-consistent pooling design"
            ),
            "material_gain": material_gain,
            "no_material_regression": no_material_regression,
            "gate_passed": gate,
        },
    }
    artifact = ROOT / "artifacts" / "ltfp_rc2_experiment_2026-11-14.json"
    document = ROOT / "docs" / "2026-11-14-ltfp-rc2-experiment.md"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    document.write_text(_markdown(report) + "\n")
    print(_markdown(report))
    print(f"saved {artifact}")
    print(f"saved {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
