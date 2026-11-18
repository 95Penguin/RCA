# Baseline 清点与接入计划

核对日期：2026-11-18。主工程：`/Users/95penguin/project/NextMinuteRCA`。

本目录管理 baseline 的作者源码、任务适配和运行配置。**首批四个框架适配器已经实现并通过 OB/TT 小样本运行**，详见 [运行说明与差异](RUNNING.md)。其他候选已固定源码，尚待适配。下方保留清点时的输入审计与接入计划；实时状态以 `registry.json` 为准。历史模型仍从原来的 `src/models/` 和 `scripts/` 运行。

选取原则：有作者仓库时优先复用其实现，在外层适配数据和评价。作者仓库、作者维护的 benchmark 集成、第三方复现、当前工程自写适配版分别记录；不能因名称相同就视为相同实现。仓库存在不代表已经能在 RE2 或本机运行。

## 目录

```text
baselines/
  README.md             人读清单、现有实现和接入顺序
  registry.json         方法登记表与逐方法接入状态
  sources.lock.json     13 个来源的 commit 和源码文件 SHA256
  RUNNING.md            当前可运行命令、适配差异及验证状态
  runner.py             四个框架的独立训练与公共输出
  INTEGRATION.md        数据、时间边界、训练与结果适配约定
  upstream/             作者源码固定版本快照
  adapters/             公共历史窗口与作者模型适配器
  requirements-macos-cpu.lock.txt  已验证独立环境
  configs/README.md     后续逐方法配置约定
```

训练产物继续放在主工程的 `artifacts/`，原始数据继续放在 `data/RCAEval/`，不重复复制。

## 方法池

四个框架类已由用户确定。根因侧为待接入候选池，MicroRank 是对“Random Walk”等方法的具体化建议。预测侧三个方法仍是建议候选，尚未代表最终论文名单；CAD 仅为备选。DeST 暂不选，Persistence/TCN 不加入本轮候选。

### 框架类：4 个

| 方法 | 优先实现来源 | 原始任务 | 当前工程状态 / 接入重点 |
|---|---|---|---|
| LTFP | [作者仓库](https://github.com/mlk12333/LTFP) | 未来时间窗故障预测与责任服务排序 | 已有 `src/models/ltfp.py` 架构适配版；本机另有作者仓库副本。核对混合图、特征选择、时间窗目标与原实现的差异。 |
| InstantOps | [论文](https://research.spec.org/icpe_proceedings/2024/proceedings/p119.pdf)；本次未确认作者代码仓库 | 故障预测与置换归因 RCA | 已有 `instantops-adapted`；目前使用服务遮蔽近似置换归因。后续先核对论文，再决定补齐；不可把用户自己的 InstantOps 工程当成作者发布。 |
| Eadro | [作者仓库](https://github.com/BEbillionaireUSD/Eadro) | 当前异常检测与根因定位 | 已有 `eadro-adapted`，含自写编码/图组件；后续优先核对 `codes/model.py`，从作者实现外包任务适配层。 |
| ART | [作者仓库](https://github.com/bbyldebb/ART) | 异常检测、故障类型识别与根因定位 | 主工程尚无实现。先核对作者数据预处理、模型和评分流程，再决定未来任务适配；原故障前后窗口不能直接当预测输入。 |

### 预测侧：3 个建议候选 + 1 个备选

| 方法 | 实现来源 | 原始任务 | 本工程需要解决的适配 |
|---|---|---|---|
| Maat | [作者仓库](https://github.com/BEbillionaireUSD/Maat) | 预测未来指标，再在预测指标上判断异常 | 最接近原生未来预测；仍需明确指标异常分数如何对应下一分钟业务退化，不能直接把指标异常标签等同于业务标签。 |
| AnoFusion | [作者仓库](https://github.com/AIOps-Lab-NKU/AnoFusion) | 服务实例多模态无监督故障检测 | 原分数依赖预测观测与真实观测的差异。先实现截至 t 的原始检测分数，再作为历史风险对照评价 t+1；若增加未来分类头，另命名 adapted 版本。 |
| Hades | [作者仓库](https://github.com/BEbillionaireUSD/Hades) | 指标与日志的半监督异常检测 | 需要日志语义/序列输入；未来目标需要显式改为 y(t+1)，并报告有标签、无标签预算及伪标签来源。 |
| CAD（备选） | [作者仓库](https://github.com/dawnvince/MTS_CAD) | 多变量指标异常检测 | 指标专用对照；采用截至 t 的检测分数，或明确标记预测头适配版，不能使用 t+1 的真实观测生成提前分数。 |

### 根因侧：7 个候选

| 方法 | 优先实现来源 | 输入与方法作用 | 接入重点 |
|---|---|---|---|
| DiagFusion | [作者仓库](https://github.com/AIOps-Lab-NKU/DiagFusion) | 多模态事件表示与依赖图；根因/故障类型诊断 | 需要从原始遥测构造事件，核对 fastText 与图输入；类型标签的额外使用需记录。 |
| TVDiag | [作者仓库](https://github.com/WHU-AISE/TVDiag) | 任务相关多模态表示；根因定位与类型识别 | 需要事件提取、图和任务标签；其类型识别任务不属于未来系统故障预测。 |
| TORAI | [作者维护的 RCAEval 集成](https://github.com/phamquiluan/RCAEval/blob/main/docs/TORAI.md) | 多源异常评分、GMM 分组、RCD 簇内细化 | 不是已确认的独立 TORAI 仓库。核对专用多源文件和 patched causal-learn；不能直接套另一个数据集或复制其示例分数。 |
| RCD | [作者仓库](https://github.com/azamikram/rcd) | 正常/异常指标分布与因果发现 | 保留作者算法，统一正常参考数据和历史截止时间；核对短历史的有效样本量与指标到服务的聚合。 |
| CIRCA | [作者仓库](https://github.com/NetManAIOps/CIRCA) | 指标、因果图、业务 SLI 与干预识别 | 原接口包含检测时刻与分析窗口，需审计是否访问预测时刻之后的数据；不能传入未来故障时间或真实根因构图。 |
| TraceRCA | [作者仓库](https://github.com/NetManAIOps/TraceRCA) | 请求 Trace 特征、异常请求与可疑服务集 | 需要请求级原始 Trace，分钟均值不足以保留原算法；只接入截至 t 已完成可用的观测。 |
| MicroRank（Random Walk 的具体化建议） | [作者仓库](https://github.com/IntelligentDDS/MicroRank) | Trace 延迟、PageRank 与谱分析 | 保留请求/操作层图，固定到服务层的映射；不能用随手实现的随机游走冒充作者算法。 |

## 当前已有实现核对

| 项目 | 现有位置 | 已确认的边界 |
|---|---|---|
| LTFP 架构版 | `src/models/ltfp.py` | 模态时序编码、门控融合、加权图注意力、两个任务头；当前主表使用公共因果调用图，不等于完整复现作者的依赖 + 训练相关性混合图。 |
| InstantOps 适配版 | `src/models/instantops.py`、`src/rca/ranking.py` | GCN + GRU；RCA 为 masking importance，不是已核对的原论文置换流程。 |
| Eadro 适配版 | `src/models/eadro.py` | 自写多源编码、GLU、GATv2 风格图传播和两个任务头；未来标签适配，不是直接调用作者仓库。 |
| 注册与训练 | `src/models/factory.py`、`src/training/experiment.py` | 工程入口可复用，但不应要求所有外部算法都迁移进同一 PyTorch 训练循环。 |
| 现有批量入口 | `scripts/run_re2_baseline_matrix.py` | 支持现有模型；未接入上述新候选。脚本默认模型选择设置与后续统一比较规则需要显式对齐。 |
| 指标 | `src/training/metrics.py` | AP、F1@0.5、窗口排名和事后事件聚合已有实现。新方法需导出原始分数，由公共评价计算。 |

本机 LTFP 作者仓库副本：`/Users/95penguin/project/LTFP`，origin 为 `git@github.com:mlk12333/LTFP.git`，2026-11-18 读取 HEAD 为 `be900faf4830dd4c5429d03eb8e7b279b75f11b7`。它有未跟踪实验文件，不能将整个工作目录当作干净的上游快照复制。后续从固定 commit 读取/获取源码。

主工程当前 HEAD 为 `e7168868fd0068208f57dd844f8adb817e48208c`，存在大量未提交研究变更；这个 HEAD 不能单独代表本次审计所见的完整代码状态。现有实现与结果全部保留，今后作者版使用独立实现标识，不覆盖旧版。

## 数据现状

- 原始 ZIP：`data/RCAEval/RE2-OB.zip`、`data/RCAEval/RE2-TT.zip`。
- 现有面板：`artifacts/re2_ob_panel_fair_v1/`、`artifacts/re2_tt_panel_fair_v1/`。
- OB 有 11 个候选服务，TT 有 27 个；每个服务每分钟 19 个特征。
- 指标值为 CPU、内存、磁盘 IO、socket，另有观测掩码；日志为总量/错误计数/覆盖；Trace 为计数/错误/平均耗时，另有入口 p95 和历史滚动退化特征。
- 调用图来自父子 span 均已完成的边，现有面板存的是对称计数图。需要方向或请求结构的方法必须重新从原始 Trace 导出，不能从对称矩阵恢复方向。
- 面板不包含完整日志文本、日志事件序列、请求级 Trace 或高频指标序列。原始 ZIP 是这些输入的待审计来源；当前未逐方法确认所有字段均满足作者算法。

## 后续实现顺序

1. **先核对已有框架**：固定 LTFP/Eadro 作者版本，逐项核对现有适配差异；InstantOps 单独记录代码来源缺口与置换归因差异。保留旧结果。
2. **补齐 ART**：优先跑作者最小示例，确认原评分的时间依赖，再接 RE2 原始数据导出。
3. **根因侧分输入接入**：RCD/CIRCA 的原始指标导出、TraceRCA/MicroRank 的请求 Trace 导出、DiagFusion/TVDiag 的事件导出、TORAI 的专用多源导出。可按输入成熟度推进，不以最终分数决定是否收录。
4. **预测侧**：优先 Maat 和 AnoFusion，再补 Hades；CAD 为备选。先固定检测到预测的适配模式，避免一边看测试结果一边换定义。
5. 每个方法经历“固定来源 → 作者示例/最小运行 → RE2 小样本 → 同划分正式实验”；新增接口的检查针对时间泄漏、标签/服务对齐与分数导出，不为本轮文档清点训练模型。

实现和评价细节见 [INTEGRATION.md](INTEGRATION.md)，方法来源和未完成项见 [registry.json](registry.json)。
