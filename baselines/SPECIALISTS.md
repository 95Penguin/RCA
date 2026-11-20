# 预测与根因专用 baseline

接入日期：2026-11-19。当前提供 10 个作者代码适配器；与四个联合框架一起，形成 14 个可运行对照。下述均为本项目历史窗口任务的适配版本，不等于原论文原数据集上的逐项复现。

## 已实现的方法

| 入口 | 用途 | 实际复用 | 输入与适配 |
|---|---|---|---|
| `rcd` | RCA | 作者 localized RCD、离散化与 PC 骨架搜索 | 高频指标：训练正常参考对比允许的历史窗口；指标排名按服务去重。 |
| `circa` | RCA | 作者 CaseData、RHTScorer 与模型流程 | 高频指标；训练调用图构造静态先验，资源指标连接本服务延迟，被调用方延迟连接调用方延迟。此先验不证明因果。 |
| `tracerca` | RCA | 作者事务构建、关联挖掘和排序 | 请求级 Trace；训练操作耗时 3σ 或当前已完成 Span 错误标记异常。默认非 PRFL 模式仅挖单项，已检查与原完整挖掘排序一致。 |
| `microrank` | RCA / 随机游走 | 作者请求图、PageRank、Ochiai 谱排序 | 使用相同历史请求异常规则；要求窗口内正常、异常两类请求；操作分数按服务取最大值。它是具体的 Random Walk 对照，不另造一个泛称算法。 |
| `torai` | RCA | 作者维护 RCAEval 集成中的 TORAI | 高频指标与原始日志计数；本版未接入其可选 Trace 输入。`inject_time` 接口使用正常参考与历史段的合成分界，绝不传真实注入时刻。 |
| `diagfusion` | RCA | 作者 FastTextLab 增强、TAGClassifier 根因与类型分类器 | 将历史指标变化、日志模板与 Trace 操作转换为事件。训练词向量只用训练折；每类增强预算 64，向量 32 维。 |
| `tvdiag` | RCA | 作者 GraphSAGE 编码器、定位/类型头、图增强和四项自动加权损失 | 使用相同事件接口；保留根因/类型监督、任务内和模态间对比学习。原代码中只作诊断的 affinity 输出未接入。 |
| `anofusion` | 预测分数 | 作者 Net、GTN、GATGRU | 19 通道分钟面板，保留六类模态关系；MI 的分箱与拟合仅训练输入。四个已观测分钟重建第五个已观测分钟，服务最大残差作为对下一分钟标签的历史风险分数。 |
| `hades` | 预测 | 作者 MultiModel、时序/日志编码器和交叉注意力 | 指标与原始日志词向量；训练折每类固定取 20% 有标签样本，第二个 epoch 起加入置信度≥0.95的伪标签；目标改为下一分钟业务状态。 |
| `maat` | 预测 | 作者扩散 TrainingNetwork/PredictionNetwork、RandomForest 特征排序、IsolationForest | 训练输入观测率≥95%的指标通道；生成下一分钟指标，提取预测均值、相对当前变化和预测标准差。验证标签选择至多64特征，正常训练预测拟合检测器。实际未来观测只进入训练生成损失。 |

DiagFusion/TVDiag 使用训练根因和故障类型标签；这些比无监督 RCA 方法拥有更多监督，论文表格应披露。事件提取器是公共任务适配，未宣称与作者原始预处理完全一致。Hades 的预算/伪标签流程也是明确的外层训练适配。fastText 单线程运行；监督接口接收运行 seed，无监督接口不暴露 seed，使用库的默认种子。DiagFusion 的作者增强函数内部将 Python random 种子固定为0，该行为原样保留。

## 时间、监督与评价边界

所有方法看到的是过去 5 个完整分钟，以输入末分钟起点加 60 秒为严格可用截止点；输出对应精确下一分钟的业务状态或责任服务。高频指标/请求 Trace/原始日志直接从 `data/RCAEval/RE2-OB.zip` 或 `RE2-TT.zip` 读取，保留原精度。Span 以完成时间过滤；未完成或超出历史范围的父 Span 不参与关系构建。日志使用固定变量清洗器，不复用全案例拟合的模板列。

正常参考、标准化、事件词向量、MI 图和静态关系仅使用训练折。RCD/CIRCA/TORAI 将训练正常参考与当前历史候选段比较，候选段并不被预设为已发生故障；合成时间轴仅满足作者接口。统计 RCA 只在公共可归因业务正例窗口执行以节省时间，评价标签不传给算法；这是条件 RCA，不是报警门控的端到端指标。

TraceRCA/MicroRank 无可区分证据时明确返回不可用，并在相同 RCA 正例分母上记 Hit=0、RR=0，额外报告 coverage。禁止删除这些窗口后只报成功子集。严格五分钟输入下的这些失败属于适配设置的覆盖限制，不代表作者方法在原有故障后诊断协议中的表现。

RCA 专用方法不报告 AP，预测专用方法不报告 MRR，空值保留为 `null`。主 RCA 指标为窗口 Hit@1/3/5、MRR；同分按固定服务顺序打破，并额外报告单根因设置的 tie-expected MRR。这里不做事后事件聚合。阈值和模型选择不使用测试标签。

选择规则：DiagFusion/TVDiag 取最大验证窗口 MRR；Hades 取最大验证 AP；AnoFusion 取最小正常训练重建损失；Maat 取最小训练扩散损失。相同指标时保留更早 epoch。Maat 检测器的特征筛选只用验证标签，拟合只用正常训练预测。正式实验仍需固定超参数预算和训练轮数。

## 运行

使用独立 baseline 环境，新增完整环境锁为 `requirements-specialists-macos-cpu.lock.txt`。原四框架的环境锁仍保留，完整环境兼容四框架。Python 3.11、Torch 2.2.2、DGL 1.1.3、NumPy 1.26.4；Maat 使用作者要求的 GluonTS 0.10.0，与 pandas 1.5.3 配套。不要把这些旧版依赖装入主研究环境。

在项目根目录：

```bash
uv pip sync --python baselines/.venv/bin/python baselines/requirements-specialists-macos-cpu.lock.txt
baselines/.venv/bin/python scripts/run_baselines.py doctor

# 单方法；output-dir 必须是新目录
baselines/.venv/bin/python scripts/run_baselines.py specialist \
  --method diagfusion --dataset tt --fold 0 --epochs 30 \
  --output-dir artifacts/baselines/diagfusion_tt_fold0_seed42

# 全部专用方法，OB/TT × 五折；默认 seed 42，可传多个种子
baselines/.venv/bin/python scripts/run_specialist_baseline_matrix.py \
  --folds 0 1 2 3 4 --seeds 42 --epochs 30 --jobs 1 \
  --output-root artifacts/baselines/specialists_formal_v1

# 仅验收管线的短跑
baselines/.venv/bin/python scripts/run_specialist_baseline_matrix.py \
  --folds 0 --epochs 1 --max-windows 128 --hidden 16 \
  --reference-rows 256 --diff-steps 3 --forecast-samples 2 --jobs 2 \
  --output-root artifacts/baselines/specialists_smoke_new
```

正式 Maat 默认 100 扩散步、16 个预测样本；短跑配置只用于软件验收，不能据此判断方法优劣。高频统计 RCA 不做神经训练，因此 epochs 参数对其无效。每个进程缓存两个原始案例，优先小并发避免内存压力。

每次运行保存 `data_manifest.json`、`scaler.npz`、`metrics.json`、`predictions.jsonl`；需要的训练参考/词向量/模型/检测器按方法额外保存。矩阵脚本另存每次运行日志、`runs.json`、`summary.json` 和 `failures.json`，失败时最终退出码非零，不静默跳过失败方法。旧目录不会被覆盖。

## 可追溯的兼容处理

固定作者源码快照未修改，加载前验证 SHA256。`author_imports.py` 将通用 `models/core/config` 导入隔离在私有包名下，避免覆盖研究工程模块；以下修复在加载时按精确单次匹配应用：

- Hades：将时序编码器的普通列表注册为 `ModuleList`；限制 SelfAttention 的 squeeze 轴，使 batch=1 不丢失批维度。
- AnoFusion：去掉硬编码20时间维度；使不同样本使用自己的特征图；用图连接关系屏蔽注意力，保留梯度；初始化作者未初始化的 GTAggregation 参数。
- TVDiag：去掉未使用且缺失的模块导入；对比学习标签使用当前设备；零分母加数值下界。
- Maat：将未使用的 XGBoost 导入延迟到显式选择该分支时，默认 RandomForest 不依赖系统 libomp。本版不提供 XGBoost 分支的运行保证。
- RCD/TORAI：在各自 worker 内加载作者提供的两个 causal-learn 覆盖文件，不改安装目录；MicroRank 只优化请求 ID 列表的成员查找，不更改图和 PageRank。

这些修复有的会改变有缺陷的原程序行为，必须在复现实验说明中保留。算法分数与原论文不同不能仅归因于提出方法的优势。

## 尚未完成

正式五折、多种子、合理训练轮数与超参数预算实验仍未进行；未据短跑分数筛掉任何 baseline。CAD 仍是备选，未实现。TORAI 可选 Trace、DiagFusion/TVDiag 更贴近作者的事件预处理、Eadro 的 Hawkes 输入均可另作增强，但需要新版本标识和独立结果目录。
