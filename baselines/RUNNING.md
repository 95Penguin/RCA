# 已实现的首批框架 baseline

2026-11-18：本轮实现 LTFP、InstantOps、Eadro、ART 的 RE2 任务适配与独立运行入口。其他根因/预测候选已固定作者源码快照，尚未编写算法适配器。不能将“已下载源码”理解成“全部 baseline 已能训练”。

## 运行

验证环境为 macOS ARM、CPU、Python 3.11，关键依赖 torch 2.2.2、DGL 1.1.3、numpy 1.26.4。完整版本见 `requirements-macos-cpu.lock.txt`。保留主工程原环境；作者旧 DGL 不安装进主工程 `.venv`。

独立环境位于 `baselines/.venv/`。如在新机器建立环境：

```bash
uv venv --python 3.11 baselines/.venv
uv pip sync --python baselines/.venv/bin/python baselines/requirements-macos-cpu.lock.txt
```

从主工程根目录运行：

```bash
# 检查源码哈希和依赖是否存在
baselines/.venv/bin/python scripts/run_baselines.py doctor

# 单方法、单折；输出目录必须尚不存在
baselines/.venv/bin/python scripts/run_baselines.py run \
  --method ltfp --dataset ob --fold 0 --epochs 30 \
  --output-dir artifacts/baselines/ltfp_author_ob_fold0_run1

# 四个方法、两个数据集、五折；这是正式规模运行命令，本轮尚未执行
baselines/.venv/bin/python scripts/run_author_baseline_matrix.py \
  --methods ltfp instantops eadro art \
  --datasets ob tt --folds 0 1 2 3 4 --epochs 30 \
  --output-root artifacts/baselines/frameworks_run1

# 本轮已验证的低成本运行规模，仅检查管线
baselines/.venv/bin/python scripts/run_author_baseline_matrix.py \
  --datasets ob tt --folds 0 --epochs 1 --max-windows 128 \
  --hidden 16 --repeats 2 \
  --output-root artifacts/baselines/frameworks_smoke_new
```

所有方法默认 CPU。`--max-windows` 会取各划分的前缀子集，结果带 `smoke_only=true`，不能作为论文性能。正式结果需要移除此参数，并固定训练预算和多个种子。

## 实现与差异

| CLI 方法 | 独立实现标识 | 实际复用 | 任务/输入差异 |
|---|---|---|---|
| ltfp | `ltfp-upstream-panel-state-v1` | 作者 `LTFPModel` 和 `combine_topology_and_correlation` | 单 horizon 的 exact t+1 业务标签；公共 19 特征面板；相关信号为标准化训练输入各服务的最大绝对特征值，top-k=2。与当前历史调用边取并集，不使用全测试图。 |
| instantops | `instantops-paper-panel-permutation-v1` | 当前工程的 GCN+GRU 论文适配实现 | 暂无确认的作者仓库。RCA 使用候选服务历史时间步联合置换的平均风险下降，固定其他服务和图，默认 16 次、保留符号；这是明确声明的置换变体，不声称原论文未发布的精确实现。 |
| eadro | `eadro-upstream-panel-state-v1` | 作者 `MainModel` 的多源编码器、DGL GATv2、注意力池化、实际注册的 `detecter/localizer` | 日志用共同面板的计数/覆盖特征均值，**尚未复现作者 Hawkes 日志强度**；Trace 用原模型要求的单个服务平均延迟通道；未来业务标签与根因标签独立，不从根因是否存在推导业务标签。 |
| art | `art-upstream-panel-score-lag-v1` | 作者 `AutoRegressor`、Transformer、GRU、图卷积 | 五个已观测分钟内，前四分钟预测观测 t；以 residual(t) 判断 y(t+1)，不是看到未来真实遥测后计算残差。按原 SLD 思路取最严重三个服务的通道残差和，按各服务 ILD 与 SLD 的余弦相似度排名；使用单窗口，无事后故障窗口，未做故障类型任务。 |

上表都是**统一任务下的适配版本**，不是作者原数据集/原任务结果的精确复现。旧 `ltfp`、`instantops-adapted`、`eadro-adapted` 的代码与结果保留，不能用新结果无说明替换旧表。

LTFP/Eadro/InstantOps 以验证 AP 最大选择同一个模型，平局取最早 epoch；无验证正例时明确失败。ART 在业务正常的训练历史上自回归训练，按最小训练重建 epoch loss 选模型，不使用根因标签选模型；验证业务标签仅用于后续阈值选择。分类采用训练正负比例权重，LTFP 的根因 CE 权重为 0.5，Eadro 检测/定位各 0.5，InstantOps 不用根因标签训练。

## 可追溯兼容修复

作者快照本身不改动。`source.py` 验证 `sources.lock.json` 中的 SHA256，加载时应用数量严格匹配的兼容替换。

- Eadro：给 `ConvNet` 接受原调用已经传入的多余 kwargs；修复 `trace_dropout/metric_dropout` 未定义变量（两者原本并未进入卷积计算）；绕开原 `forward` 中错误的任务头属性名，直接调用实际注册的作者任务头。
- ART：将图层列表注册为 `ModuleList`，使后续图层参与优化、状态保存和设备迁移；修复 Transformer 的 batch 维度，否则不同样本会相互参与注意力；将两个非连续张量 `view` 改为 `reshape`。这些补丁改变了有缺陷的原运行行为，因此明确记录，不称为未经改动的作者执行。
- ART 服务数量 11/27 不能被作者示例的 2 个头整除，此时配置为 1 个头；隐藏维度按命令行配置，保留该项记录。

## 数据边界与输出

`HistoricalWindows` 直接取截止 t 的张量，不调用会同时返回 `next_node_features` 的原数据 `__getitem__`。训练、验证、测试使用原有部署组五折；标准化与相关图均只拟合训练输入分钟。

每次运行包含：

- `data_manifest.json`：数据文件哈希、完整窗口划分、实际运行窗口索引和候选服务。
- `scaler.npz`：训练折拟合的标准化参数。
- `best_model.pt`：同一所选模型、模型配置、作者 commit、标准化参数。
- `metrics.json`：原始 AP、验证阈值 F1、概率模型的 F1@0.5、窗口 Hit/MRR、明确标为事后的事件聚合、时间、参数与源码信息。
- `predictions.jsonl`：输入/目标时刻、案例、业务标签、原始风险分数、服务分数与固定服务排序。

阈值仅使用验证集；异常分数不伪装为概率。窗口 RCA 对全部有根因标签的业务故障窗口评价，不以模型是否报警筛选。事后事件聚合平均原始服务分数，仅供历史对照，不等于提前定位表现。

## 当前验证状态

- 13 个上游来源已经固定 commit，保留 425 个源码/配置/说明文件（不含原始示例数据、模型权重、字节码或嵌套 Git 元数据）。以 `sources.lock.json` 的实际文件清单为准。
- 8 项新检查覆盖作者模型调用一致性、Eadro 分支梯度、ART 批量独立性和图层注册、未来遥测排除、置换归因批量独立性、验证阈值、真实 RE2 划分。
- 四个方法 × OB/TT，共 8 次 fold 0、1 epoch、每划分最多 128 窗口的真实数据管线运行通过。
- 尚未完成正式 30 epoch 五折与多种子实验；本轮小样本分数不用于比较论文效果。
- DiagFusion、TVDiag、TORAI、RCD、CIRCA、TraceRCA、MicroRank、Maat、AnoFusion、Hades 的后续接入见 [专用方法说明](SPECIALISTS.md)。运行这十个方法应使用新增完整环境锁，原四框架锁只包含首批依赖。
