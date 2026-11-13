# 当前代码与数据地图

## 结论

当前实验的主工程是`/Users/95penguin/project/NextMinuteRCA`。新方法、新实验和新文档只在这里修改。
当前RE2原始ZIP、正式标签、模型面板、训练产物与报告都已位于该目录，日常RE2实验不再依赖旧`InstantOps`工程。
根目录原有的`deliver`、`deliver_next`和`tt_delivery`只是历史导出副本，已统一归档到`archive/legacy_exports/`，不作为运行入口。

## 代码职责

| 功能 | 当前文件 |
|---|---|
| RE2-OB/TT业务故障标签生成 | `src/data/re2_labels.py` |
| RE2分钟特征、面板、滑窗和任务适配 | `src/data/re2.py` |
| LTFP | `src/models/ltfp.py` |
| Eadro-adapted | `src/models/eadro.py` |
| InstantOps-adapted和MTG的GCN+GRU主干 | `src/models/instantops.py` |
| 通用简单baseline和模型公共组件 | `src/models/baselines.py`、`src/models/common.py` |
| 模型选择、训练、评价和预测导出 | `src/training/experiment.py` |
| 单次RE2标签/预处理/训练入口 | `scripts/run_re2.py` |
| 论文baseline五折批量运行 | `scripts/run_re2_baseline_matrix.py` |
| 故障阶段、故障类型和根因服务错误分析 | `scripts/analyze_re2_errors.py` |
| 状态转移数量与Persistence基线 | `scripts/analyze_transition_baselines.py` |
| 严格首分钟/恢复转移可学习性线性探针 | `scripts/probe_transition_learnability.py` |
| 根因服务与相邻症状服务时序审计 | `scripts/audit_root_symptom_timing.py` |

## 标签究竟在哪

### 标签生成代码

在当前主工程：

- `src/data/re2_labels.py`
- `src/data/re2.py`

### 当前模型实际使用的正式RE2标签源

- `/Users/95penguin/project/NextMinuteRCA/data/processed/re2_ob_labels_v1/next_minute_labels.csv`
- `/Users/95penguin/project/NextMinuteRCA/data/processed/re2_tt_labels_v1/next_minute_labels.csv`

这两个路径已写入主工程面板的`feature_schema.json`。训练和错误分析会跟随该记录，不会误用工作目录中的历史审计副本。

### RE2原始数据

- `data/RCAEval/RE2-OB.zip`
- `data/RCAEval/RE2-TT.zip`

两个ZIP已从旧数据仓库复制并做哈希校验，旧工程中的原件未删除。

MicroSS原始分卷没有重复复制到当前工程。相关代码和历史产物仍保留用于追溯，
但当前论文主实验及日常运行只要求上述两套RE2数据。

### 当前训练直接读取的模型面板

- `artifacts/re2_ob_panel_fair_v1/`
- `artifacts/re2_tt_panel_fair_v1/`

每个面板包含服务分钟特征、调用图、模型标签副本、服务列表和原始标签源记录。

## 当前结果在哪

| 结果 | 位置 |
|---|---|
| InstantOps/LTFP/Eadro/MTG每折模型和预测 | `artifacts/re2_*_deployment_fold*/` |
| 统一baseline汇总 | `artifacts/re2_next_minute_all_baselines_seed42_v1.json` |
| LTFP首次可感知1/10分钟对照 | `artifacts/re2_first_visible_*` |
| 错误分析JSON | `artifacts/re2_error_analysis_seed42_v1.json` |
| Persistence审计 | `artifacts/re2_transition_persistence_audit_v1.json` |
| 状态转移可学习性探针 | `artifacts/re2_transition_learnability_probe_v1.json` |
| 根因—症状时序审计 | `artifacts/re2_root_symptom_timing_audit_v1.json` |
| 组会汇报 | `docs/2026-11-13-group-report-task-and-results.md` |
| 错误分析报告 | `docs/2026-11-12-re2-error-analysis.md` |

## 其他工程与当前工程的关系

- `/Users/95penguin/project/InstantOps`：保留早期复现历史和原始数据备份，不再被当前RE2日常实验依赖。
- `/Users/95penguin/project/LTFP`：用于核对LTFP论文与官方/复现代码，不直接产生当前主表。
- `/Users/95penguin/project/PredictiveRCA`：保留早期故障检测、AIOps2025和RootFlow/MTG尝试，当前暂停。

## 以后如何运行

```bash
cd "/Users/95penguin/project/NextMinuteRCA"
uv sync

# 三个论文baseline的RE2五折
uv run python scripts/run_re2_baseline_matrix.py \
  --models instantops-adapted ltfp eadro-adapted \
  --datasets ob tt --folds 0 1 2 3 4 --epochs 30

# 复用已有预测生成错误分析
uv run python scripts/analyze_re2_errors.py
```

如果在终端看到当前目录是`/Users/95penguin/project/PredictiveRCA`或`/Users/95penguin/project/InstantOps`，应先切回上述主工程再运行。
