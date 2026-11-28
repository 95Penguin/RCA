# OCSID / CPCA 论文前代码审查与重构

更新：后续 corrected-v2 已修复下述四个实现问题，并增加显式入口过滤与可选有界时钟对齐。
本文保留发现时的证据，不代表当前实现仍原样保留这些错误。
当前方法见 [四模块说明](2026-11-25-method-v2.md)。

日期：2026-11-25。结论：目录整理已完成；当前数值实现仍有正确性问题，
不能以“实验已跑完”为由直接冻结论文主表。本次仅保留历史行为重构，未修正下列算法，
未覆盖旧实验、未重跑正式训练。新增 JSON 标记 legacy-v1。

## 1. 已完成的目录整理

特征构建、五折训练、CPCA 审计、历史融合分别迁入 `src/ocsid/` 的
`operation_builder.py`、`training.py`、`audit.py`、`fusion.py`；原脚本保留为薄入口。
增加 `python -m ocsid` 分发，保留 tabular baseline 对 `_build_arrays` 的兼容导入。
`from ocsid import Span` 不再加载 PyTorch；`OCSIDModel` 保留懒加载导出。
更新代码地图、README；审计新增 `scores_by_method`，旧 `cpca_scores` 含义保持 hard。
训练/审计报告增加参数记录，正常参考的 protocol 反映实际 CLI 参数。
没有迁移 baseline/upstream、旧实验、原始 ZIP 或 artifacts。

## 2. 写论文前必须修正并验证

### P0：CPCA 串行重放不守恒

位置：`src/ocsid/cpca.py:decompose_tree`、`replay`。
相邻两个子调用之间的同一空档，同时计入前一个 child 的 post 和后一个 child 的 pre。
local 被截断为 0，无法抵消这段双计时。因此“不并行”不等于“精确重放”。

最小用例：root=[0,100]，child A=[10,30]，child B=[50,70]。
中间 20 被算两遍；不恢复任何服务时，replay=120，观测 duration=100。
即使正常参考缺失，仍可能引入额外树误差；恢复效果的分子因此不再只是该服务贡献。

真实 case 抽查（完整分组、未只挑正例）：

| 数据 | case | 可解析请求 | 不守恒请求 | 最大多计时间 |
|---|---|---:|---:|---:|
| OB | currencyservice_delay/1 | 23667 | 16539 | 1640186 μs |
| TT | ts-train-service_delay/1 | 5765 | 5072 | 1497904 μs |

这是抽查，不代表全库比例，但已证明并非只有人工构造才出现。

修正时必须为兄弟间空档规定唯一归属：归前一 child 的 post、后一 child 的 pre，
或显式切分但总量只能计一次。这个选择改变归因假设，不能当格式化直接替换。
必须覆盖无干预守恒、多子节点、缺失正常参考、多次同服务调用；另对并行树拒绝或实现
关键路径重放。然后重算正常参考、CPCA、hard/soft 消融、事件统计、bootstrap 和 horizon。
依赖该 local/pre/post 分解的对照也需重算。

### P1：Operation GRU 混合时间轴和接口轴

位置：`src/models/ocsid.py:forward`。
输入 `[B,T,O,D]` 直接 reshape 为 `[B*O,T,D]`，没有先交换 T/O。
例如两个 operation 的真实历史为 `[0,1,2]`、`[100,101,102]`，
GRU 实际接收 `[0,100,1]`、`[101,2,102]`。
应使用 `permute(0,2,1,3).reshape(B*O,T,D)`，并验证 operation 重排等变性。
OB 只有一个 operation 时该问题不触发；TT 必须重训预测与 operation 相关消融。
旧结果的 q_o 不能解释为独立 operation 风险历史。

### P1：Operation 标准化使用 held-out 数据

位置：`src/ocsid/training.py:_standardize`。
已经算出 `train_positions`，但实际用 `rows=O[M>0]` 拟合均值和标准差，包含验证/测试。
最小测试中仅放大 held-out operation 值，训练集标准化结果就改变。
这是无标签的测试分布泄漏，影响幅度需修正后实测，不能预判提升或下降。
应只从训练位置对应的有效 operation 行拟合，再变换所有分区；必要时对重叠窗口
去重拟合需另行说明。OB/TT 预测及消融均需重训。
早期 `train_ocsid_predictor.py` 也明确在全样本拟合缩放，它只能作历史结果。

### P1：观测标记没有保留原始二值

同一 `_standardize` 先 z-score，再对 observed_* 和 log_observed 四舍五入。
这不等于保留二值：训练 0/1 比例为 4:1 时，1 会变成 round(2)=2。
应直接复制原始 X 的标记列。与上一个修正一起重训即可。

## 3. 论文口径和部署边界

1. **预测与定位分别评估**：audit 使用真实正例标签选择待定位窗口，且过滤注入前分钟。
   当前结果证明的是正例条件下的 RCA；没有验证预测阈值触发后的全链路召回/误报。
   框架图的“高风险触发”是部署流程，不能直接当成已测端到端结果。
2. **soft/hard 输出不一致**：旧 audit 的 `cpca_scores` 始终是 hard；融合、H@5、
   bootstrap 等脚本默认读它。soft 的事件排名与 hard JSON 重算结果不能混作同一配置。
   本次追加 `scores_by_method` 便于后续显式选配置，但旧产物未补造分数。
3. **未覆盖口径不一致**：compute_cpca_hit5 对未覆盖事件的 reciprocal rank 贡献为 0；
   paired_bootstrap 使用 N+1（贡献为 1/(N+1)）；fusion 对无 CPCA 的窗口置零；
   audit 自身统计通常跳过无排名。论文须指定统一事件全集、候选服务全集、缺失/平分规则。
   这部分可对既有分数重新汇总，无需所有 baseline 重新训练。
4. **辅助头仍会在 eval 运行**：训练脚本用 aux_root=True；forward 未按 training 开关，
   推理结果虽未使用该分数，但并非自动物理删除。父类还创建了 root_head。
   应表述为“辅助头不参与最终归因”，真正无头部署需要另做权重过滤与预测等价检查。
5. **没有完整部署 checkpoint**：训练只保存 JSON，best state 保留在内存。
   后续应保存权重、模型参数、服务/operation 顺序、fold scaler、阈值、参考和实现版本。
6. **前十分钟参考**：只有参考窗口结束后才可称在线可用；需要核对评估窗口是否排除
   warm-up。pre 默认值利用注入时间，应仅用于 oracle 对照，正式命令显式 first10。
7. **因果表述**：span 包含关系是时间核算结构；边等待归被调用方是建模假设。
   不足以证明真实干预效果可被精确识别。写“结构反事实估计”并列出假设；不要直接称
   “观测树提供真实因果方程，因此修复效果精确”。

## 4. 优化建议的顺序

先修正上述守恒、轴排列和 train-only 标准化，固定 hard/soft 与 coverage 口径；
验证通过后再做性能优化：fixed 参考无需构建 rolling 的全量 minute_buckets；
每折重复读取/组窗可缓存；历史融合入口与正式定位入口明确分开。
这些工程优化不会替代正确性修正。当前不建议再加网络模块或寻找额外创新点。

最低重算范围：OCSID OB/TT 正式预测、对应消融和 horizon；CPCA 两库及其对照、
soft/hard、bootstrap、分类型与 horizon；baseline 的已有预测可复用，统一重算 RCA 指标。
CPCA 树分解改变后特征构建目前只用 root 的统计不受 local 数值变化直接影响，
但应先逐行验证特征表相同，再决定是否复用。不能因为重构通过就宣称历史主表已验证。

## 5. 验证记录

原有 6 项方法测试通过；新加 3 项包/API 测试通过；4 个正确性契约被标记为
expectedFailure，明确记录尚未修复的问题（不是“算法测试全部通过”）。
旧脚本与新包的 8 个 CLI --help 成功；对比迁移前后非 CLI 函数 AST 全部相同；
模型和 CPCA 核心文件与迁移前快照逐字节相同。旧实验产物未更改。

复现：

```bash
.venv/bin/python -B -m unittest tests.test_ocsid_contracts -v
```

测试的 expectedFailure 标记只能在修正实现并更新实验版本后移除。
