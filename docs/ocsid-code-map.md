# OCSID 代码地图

本文件说明“我们的方法”对应的代码位置。OCSID 分为**可训练预测**与**零训练归因**两部分，
代码按这个边界组织。

## 1. 包结构

```text
src/
  models/
    ocsid.py          OCSIDModel：服务时序 + 图 + operation 风险 + source/impact 头
    ltfp.py           复用的 LTFP 编码器、GAT、模态融合
    common.py         因果卷积、特征分组、图输入校验
  ocsid/
    __init__.py       包导出（Span / decompose_tree / replay / 特征函数 / OCSIDModel）
    features.py       19 维面板特征名、17 维 operation 特征、operation 表读写
    cpca.py           span 树分解、反事实重放、责任分、rank、事件指标
  data/
    re2.py            Re2WindowDataset（含 horizon 支持）
    dataset.py        通用窗口数据集
  training/
    metrics.py        AP / F1 / RCA 指标
```

## 2. 脚本入口

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `scripts/build_ocsid_operation_features.py` | 原始 trace → “分钟 × 入口 operation”特征表 | `ocsid.cpca` |
| `scripts/train_ocsid_joint.py` | 统一模型五折训练；预测 + 事件级 RCA；概率校准 | `ocsid.features`、`models.ocsid` |
| `scripts/audit_cpca_gate2.py` | CPCA 审计：正常参考、operation 选择（hard + soft，`--soft-tau`）、基线与逐故障类型 | `ocsid.cpca` |
| `scripts/fuse_ocsid_cpca.py` | 学习头与 CPCA 的 logit/RRF 融合（消融） | 结果 JSON |
| `scripts/train_ocsid_predictor.py` | 早期纯预测版本（operation/flat），保留作对照 | `ocsid.features` |
| `scripts/audit_span_cost_signature.py` | 单 case 归属规则验证（delay/CPU 的 post-gap 证据） | 独立 |

## 3. 数据流

```text
RE2 zip
  │  build_ocsid_operation_features.py
  ▼
artifacts/re2_{ob,tt}_ocsid_operation_features_v1.csv
  │  train_ocsid_joint.py
  ▼
OCSIDModel 训练（五折）
  ├── 预测：p
  └── 学习式 RCA：a_i（辅助/消融）

RE2 zip
  │  audit_cpca_gate2.py
  ▼
artifacts/cpca_gate2_*_scores_v1.json
  │  fuse_ocsid_cpca.py
  ▼
最终根因排序（CPCA 的 r_i）
```

## 4. 改代码时从哪里下手

| 想改什么 | 改哪里 |
|---|---|
| 模型结构/超参 | `src/models/ocsid.py`（`OCSIDModel`） |
| 服务时序/图编码 | `src/models/ltfp.py`（`LTFPBaseline`） |
| operation 特征定义 | `src/ocsid/features.py`（`OP_FEATURE_NAMES`、`operation_feature_vector`） |
| span 分解/归属规则 | `src/ocsid/cpca.py`（`decompose_tree`、`replay`） |
| 正常参考/operation 选择 | `scripts/audit_cpca_gate2.py` 的 `main` |
| 训练/校准/事件指标 | `scripts/train_ocsid_joint.py` |
| 融合策略 | `scripts/fuse_ocsid_cpca.py` |

## 5. 常用命令

```bash
# 特征表（单 case 调试加 --limit 1）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_ocsid_operation_features.py \
  --dataset tt --output artifacts/re2_tt_ocsid_operation_features_v1.csv

# 统一模型五折
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --output artifacts/ocsid_joint_tt_seed42_v3.json

# CPCA 审计（需要 case 列表 /tmp/tt_cases.txt）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 \
  --output artifacts/cpca_gate2_tt_first10_z2_scores_v1.json

# 融合/消融
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/fuse_ocsid_cpca.py \
  --dataset tt --ocsid artifacts/ocsid_joint_tt_seed42_v3.json \
  --cpca artifacts/cpca_gate2_tt_first10_z2_scores_v1.json

# 单元测试
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_ocsid -v
```

## 6. 已知边界

1. `OCSIDModel` 已从 `models` 包导出，但**尚未注册进 `src/models/factory.py`**；
   `run_re2.py train --model ocsid` 仍不可用，统一训练入口是 `scripts/train_ocsid_joint.py`。
   原因是通用训练循环不传 operation 特征，注册前需要扩展实验入口。
2. CPCA 不在 `OCSIDModel.forward` 内；最终排序由 `audit_cpca_gate2.py` +
   `fuse_ocsid_cpca.py` 在外部完成。这是有意的两阶段设计。
3. `train_ocsid_predictor.py` 是早期纯预测版本，保留用于对照；
   新实验应使用 `train_ocsid_joint.py`。
4. 结果 JSON 在 `artifacts/ocsid_joint_*_v3.json`、`artifacts/ocsid_summary_v3.json`、
   `artifacts/cpca_gate2_*_scores_v1.json`；软 operation 对比见
   `artifacts/cpca_soft_{ob,tt}_first10_z2_scores_v1.json` 与
   `docs/2026-11-24-cpca-soft-operation-result.md`。
