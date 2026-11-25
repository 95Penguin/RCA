# NextMinuteRCA

> **当前唯一主工程**：`/Users/95penguin/project/NextMinuteRCA`。下一分钟用户可感知故障状态预测、根因服务排序、baseline、新方法、RE2原始数据和实验产物都收口在本目录。详细映射见[当前代码与数据地图](docs/code-and-data-map.md)。

当前论文主实验使用 RCAEval RE2-OB、RE2-TT：给定过去 5 个完整分钟的
多模态微服务遥测数据，预测下一分钟是否出现用户可感知故障状态（包括持续故障），
并对根因服务排序。MicroSS/GAIA 仅保留为早期实验历史，不属于当前必跑数据集。
各数据集的业务标签与实验切分见[标签协议](docs/labels.md)。

预测任务和 horizon 采用统一定义：`state` 预测精确的 `t+h` 分钟故障状态与根因；
`first-visible` 预测 `(t, t+h]` 内是否首次出现用户可感知故障。当前主结果仍固定为
`state + 1min`，同一数据和模型入口已经支持 3min、5min 等后续实验。

## 环境与入口

```bash
uv sync
uv run python scripts/run_re2.py train --help
```

当前 RE2 入口支持 `label`、`prepare`、`train` 等子命令，每个子命令支持
`--help`。RE2 原始数据、正式标签和可直接训练的面板都已收口在本工程内。

## 当前主流程：RE2-OB / RE2-TT

`scripts/run_re2.py` 提供标签、面板构建和训练入口。两套原始 ZIP 位于
`data/RCAEval/`；具体命令和部署组五折设置见[数据集协议](docs/dataset-label-protocol.md)。

统一的 RE2 论文 baseline 五折命令为：

```bash
uv run python scripts/run_re2_baseline_matrix.py \
  --models instantops-adapted ltfp eadro-adapted \
  --datasets ob tt --folds 0 1 2 3 4 --epochs 30
```

单独运行不同状态预测 horizon：

```bash
# 当前主任务：精确预测 t+1min 状态
uv run python scripts/run_re2.py train \
  --data-dir artifacts/re2_ob_panel_fair_v1 \
  --output-dir artifacts/re2_ob_ltfp_state_1m_example \
  --model ltfp --task state --horizon-minutes 1

# 后续扩展：精确预测 t+3min 或 t+5min 状态
uv run python scripts/run_re2.py train \
  --data-dir artifacts/re2_ob_panel_fair_v1 \
  --output-dir artifacts/re2_ob_ltfp_state_3m_example \
  --model ltfp --task state --horizon-minutes 3
```

复用已有预测进行错误分析：

```bash
uv run python scripts/analyze_re2_errors.py
```

## 历史 MicroSS 流程

以下 MicroSS 命令和结果仅用于追溯早期实验。原始 MicroSS 分卷没有重复复制进
当前工程；如需重新运行，必须显式补齐 `data/MicroSS`，不能把历史结果当作当前
RE2 主实验。

## 本地复用已有数据训练

```bash
.venv/bin/python scripts/run_micross.py train \
  --data-dir data/processed/micross_mps_smoke_01 \
  --graph data/processed/micross_mps_smoke_01/service_adjacency_1m.npz \
  --labels data/processed/micross_mps_smoke_01/entry_failure_v2.csv \
  --output-dir artifacts/micross_example \
  --model gcn-gru --rca-method node-head --epochs 30 --device auto
```

新输出目录应与旧实验区分。业务标签生成方法见 [标签与评估](docs/labels.md)。
必须显式传入 `--labels`：不传时仍使用历史注入开始代理标签。
`train --model last-mlp|tcn|gru|gcn-gru|instantops-adapted|ltfp|eadro-adapted` 在同一数据、
标签与切分上运行通用基线、MTG 和论文架构基线。
`ltfp` 保留模态独立 TCN、门控融合、加权 GAT 和双任务头；
`eadro-adapted` 保留 Eadro 的多源编码、GLU、GATv2 风格传播、注意力池化和
联合检测/定位头，但将原始“当前异常检测”适配为“下一分钟预测”，不能表述为
Eadro 原任务的精确复现。
`audit` 导出故障事件与注入记录的候选关联表。

首次处理原始数据可逐步运行：

```bash
.venv/bin/python scripts/run_micross.py --check-inputs-only
.venv/bin/python scripts/run_micross.py prepare
.venv/bin/python scripts/run_micross.py graph --training-data-dir data/processed/micross_v2
```

全流程 `run` 支持 `--skip-prepare --skip-graph` 复用中间产物。
原始数据、中间产物、训练权重均不纳入 Git。

## RE2 结果与文档

`scripts/run_re2.py` 提供标签、面板构建和训练入口。两套原始 ZIP 放在
`data/RCAEval/`；具体命令和部署组五折设置见[数据集协议](docs/dataset-label-protocol.md)。
三数据集当前结果汇总见[固定基线](docs/three-dataset-baselines.md)。
四种论文方法的统一五折结果与组会讲稿见
[2026-11-12 四方法对比](docs/2026-11-12-four-method-group-report.md)。
当前固定的论文任务、数据集边界、已完成工作与后续顺序见
[研究目标与实验路线](docs/research-goal-and-roadmap.md)。
用于组会展示的任务定义、三种论文方法结果与 LTFP 首次任务对照见
[2026-11-13 组会汇报](docs/2026-11-13-group-report-task-and-results.md)。
四种方法的故障阶段、故障类型和根因服务诊断见
[RE2 错误分析](docs/2026-11-12-re2-error-analysis.md)。
“下一分钟状态”与 LTFP 式“首次用户可感知”标签的同模型对比见
[2026-11-12 LTFP 任务目标对比](docs/2026-11-12-ltfp-task-comparison.md)。

## 代码组织

完整粗细粒度模型使用 `--model joint-granularity`：从随机初始化端到端训练，
故障预测和最终 RCA 使用同一模型，不需要预训练或冻结 LTFP。
训练与统一 LTFP 对照入口为 `scripts/run_joint_granularity_experiment.py`，
模型结构与协议见[完整粗细粒度模型](docs/2026-11-18-joint-granularity-end-to-end.md)。

```text
scripts/run_re2.py          RE2 统一入口
scripts/run_micross.py      历史 MicroSS 流程编排
archive/                    只读历史快照和旧输出，不参与当前运行
src/data/                  数据适配、标签、调用图和滑动窗口
src/models/instantops.py   InstantOps/MTG 的 GCN＋GRU 主干
src/models/ltfp.py         LTFP 模态编码、图注意力和双任务头
src/models/joint_granularity.py  可从头联合训练的系统级—服务级交互模型
src/models/eadro.py        Eadro-adapted 编码、图传播和联合任务头
src/models/ocsid.py        OCSID 预测器：服务时序+图、operation 风险（归因由 CPCA）
src/models/baselines.py    LastValueMLP、ServiceTCN 通用基线
src/models/common.py       模型输入校验、特征分组和因果卷积公共组件
src/models/factory.py      统一模型注册、构建和模型专属训练策略
src/ocsid/                 OCSID 包：features / cpca / 代码地图见 docs/ocsid-code-map.md
src/training/              模型选择、训练、标准化和评估
src/training/tasks.py      state/first-visible 与 horizon 语义
src/rca/                   遮蔽归因与排名
scripts/build_ocsid_operation_features.py  trace → operation 特征表
scripts/train_ocsid_joint.py               OCSID 统一训练与事件级评估
scripts/audit_cpca_gate2.py                CPCA 归因审计
scripts/fuse_ocsid_cpca.py                 学习头与 CPCA 融合/消融
scripts/summarize_*.py     固定基线汇总
tests/                     单元与合成数据端到端测试
```

新增或修改模型时，进入对应的 `instantops.py`、`ltfp.py`、`eadro.py`；
完整粗细粒度模型集中在 `joint_granularity.py`。
预测 horizon 由任务和数据标签定义，后续 3 分钟、5 分钟任务不需要复制模型文件。

## 文档与验证

- [标签、评估和实验边界](docs/labels.md)
- [数据预处理约定](docs/preprocessing.md)
- [MicroSS、RE2-OB、RE2-TT 三数据集固定基线](docs/three-dataset-baselines.md)

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

产物包括 `best_model.pt`、`metrics.json`、`predictions.jsonl`。
故障预测报告 Precision/Recall/F1/AP；RCA 报告 Hit@K/MRR/MFR/MAR，
区分窗口与事后事件聚合结果，并报告可用根因标签数量。
