# 下一版模型设计：关键路径反事实归因（CPCA）

日期：2026-11-23

状态：设计提案，尚未实现；先过 Gate D 再写模型代码。

**定位说明（2026-11-23 更新）：** 本文档定义的是 OC-SID 独立模型的**归因头**，
不是“LTFP + 残差插件”。完整独立模型（operation 级预测、source–impact 编码器、
损失与实验协议）见 `docs/2026-11-23-ocsid-independent-model-design.md`。
LTFP 在最终方案中只作为 baseline。

## 0. 结论先行

1. **预测侧不要再加结构。** `state + 1min` 的主指标已被 persistence 上界压住
   （persistence 审计中的同口径 LTFP 对照）：
   naive persistence 在 OB 的 F1/AP 为 `0.921/0.884`，LTFP 为 `0.833/0.881`；
   TT 为 `0.894/0.844` 对 `0.785/0.806`。严格 onset 在 OOF 中只有 10/20 个正例。
   在 55/64 个测试事件、每折 6–18 个事件的规模上，增加参数只会增加方差。
2. **RCA 侧还没有被真正解决。** 监督头事件级 MRR 是 `0.878`（OB）/ `0.932`（TT），
   但所有无监督局部证据（metric 偏差 `0.757/0.892`、图传播偏差 `0.665/0.649`、
   零值遮蔽 `0.541/0.653`、正常原型恢复 `0.540/0.717`）都明显更低。
   已有的方向方法、门控、任务独立图、fault-type aux、粗细粒度联合都没有跨数据集稳定收益。
3. **方法建议：CPCA（Critical-Path Counterfactual Attribution，关键路径反事实归因）。**
   根因标签是“注入服务”，它的语义不是“异常最大的服务”，而是
   “把它恢复正常后，入口退化会消失的服务”。这是一个反事实量。
   CPCA 用请求级 span DAG 的精确结构方程（span 时长 = 自身时间 + 子区间并集）
   做重放，直接估计“恢复服务 v 后入口时延下降的比例”，并用路径覆盖率与估计稳定性
   做解析式可靠性收缩。归因器零训练参数，可与任意预测器组合。
4. **一句话创新点：** 不再给异常打分，而是用调用链自身的守恒关系把
   “根因（高偏差且位于关键路径）”与“症状（高偏差但无关键路径松弛量）”分开；
   且它不是学习出来的因果图，而是可精确重放的观测 DAG。

## 1. 现在的真正瓶颈

### 1.1 统计功效

| 项目 | RE2-OB | RE2-TT |
|---|---:|---:|
| case 数 | 90 | 90 |
| 有业务影响 case | 51 | 53 |
| 事件数 | 55 | 64 |
| 正例分钟 | 455 / 1944 | 436 / 1879 |
| 每折测试事件数 | 6, 8, 10, 13, 18 | 9, 11, 13, 15, 16 |
| LTFP 事件 MRR 逐折范围 | 0.688–1.000 | 0.700–1.000 |
| LTFP 事件 MRR 五折均值±SD | 0.878±0.119 | 0.932±0.131 |

五折 SD 约为种子 SD 的 2–10 倍（gru/gcn-gru 三种子 SD 约 `0.007–0.078`）。
**结论：任何小于约 0.03 的事件级 MRR 差异，用“五折均值比较”都不可判定。**
后续所有结论必须用**同事件配对**统计（bootstrap over events、McNemar），
而不是比较两个五折均值。

### 1.2 预测任务的天花板

- persistence 上界高于 LTFP（见 §0）。
- 严格 onset 只有 10（OB）/ 20（TT）个 OOF 正例；线性探针 AP 只有 `0.027/0.107`。
- `first-visible` 同模型 F1 掉到 `0.491/0.308`。
- 注入到可感知退化的中位提前量是 1 分钟。

因此“下一分钟是否故障”这个头继续做深没有空间；它的主要信息就是“当前是否已经退化”。
论文必须把它作为 **persistence + onset 分解** 报告，而不是继续包装成预警能力。

### 1.3 RCA 的共同失败模式

已失败的 20 个方向（方向图、stage route、directional residual、future-risk query、
null anchor、protected gradient、task-graph、fault-type aux、joint granularity、
candidate verifier、OC-SID Gate A 等）有一个共同点：

> 它们都在改进“异常证据如何表示/如何融合”，也就是在给**异常大小**排序；
> 而根因标签要求的是**反事实影响**。异常大不等于可移除的时延多。

Gate A 已经给出直接证据：operation exclusive 在 CPU/loss 上是好的排序量，
在 delay 上明显更差（TT delay 事件 Hit@1 `0.455` 对 inclusive `0.909`）。
原因不是模型不够强，而是**归属对象错了**：网络注入的等待时间出现在调用方的
self time 里，而不是被调用方的 exclusive 里。任何“把服务自身 exclusive 当根因分数”的
做法都会在不同故障类型之间摇摆。

## 2. 创新点从哪里来

分三层，按可辩护性从高到低：

### 2.1 机制层（主贡献）：CPCA

用观测到的请求 DAG 作为结构因果模型，根因 = 使入口时延下降最多的干预。
结构方程来自 span 区间的守恒关系，不需要学习因果图，也不需要在 55 个事件上训练。

### 2.2 任务/度量层（配套贡献）

- persistence 分支与 onset 分支分开报告，主概率按“当前是否已退化”确定性路由；
- 事件级配对统计 + 逐故障类型表；
- persistence 基线、first-visible 任务和 10 个经典 RCA 方法放在同一协议下。
  这一层单独就足以构成“benchmark / measurement”论文的一半。

### 2.3 系统层（加分项）

- OB（11 服务）与 TT（27 服务）跨系统稳定性；
- 无业务影响注入 case 上的源识别（source identifiability），
  用来证明方法不是在拟合“影响”标签。

## 3. CPCA 方法

### 3.1 结构分解

对一次请求 $\tau$ 的 span 树，对每个 span $s$：

$$
d_s = \text{self}_s + \left|\bigcup_{c\in \mathrm{children}(s)} [\text{start}_c,\text{end}_c]\right|
$$

其中 `self_s` 是父区间扣掉直接子区间并集后的部分，它混合了两类物理时间：

- **本地计算/IO 时间**：属于服务 $v(s)$；
- **边等待时间**：父 span 发起调用到子 span 开始之间的前置空档，
  以及子 span 结束后父 span 继续之前的后置空档。这部分主要是网络 RTT 与对端排队，
  **通常**应归被调用方（§3.1.1 讨论它不能完全归给被调用方的情形）。

因此把成本拆成三类，并明确归属：

| 成本 | 定义 | 默认归属 |
|---|---|---|
| `local_s` | 服务 $v$ 的 self 时间中，不与任何调用空档重叠的部分 | 服务 $v$ |
| `gap_pre(c)` | 紧邻 `start_c` 之前、属于父 self 的空档，`clamp(≥0)` | 归被调用服务 $v(c)$（请求路径） |
| `gap_post(c)` | 紧邻 `end_c` 之后、属于父 self 的空档，`clamp(≥0)` | 归被调用服务 $v(c)$（返回路径） |

空档按父 self 的连通段分配：每个空档段只分给与它相邻的那一次调用，
没有相邻调用的段留在父服务（例如父自身的计算）。这样
`local_s + Σ_c(gap_pre+gap_post) = self_s` 不重不漏。

对每个服务 $v$，其单请求成本为

$$
c_v(\tau)=\sum_{s:v(s)=v}\text{local}_s
+\sum_{c:v(c)=v}\big(\text{gap}_{pre}(c)+\text{gap}_{post}(c)\big)
$$

即：**local 归自身，进出该服务的两条边等待都归被调用方**；
$\sum_v c_v(\tau)$ 覆盖整棵树的所有 self 时间，不重不漏。
“pre-gap 归调用方”的变体见 §3.1.1。

**这一步是相对 Gate A 的关键修正：** Gate A 把网络等待留在调用方的 exclusive 里，
所以 delay 类故障被系统性归错；CPCA 把边等待归给被调用方，使
“CPU/disk → local 上升”和“delay → child duration + post-gap 上升”落在同一个服务分数上，
同时保留可解释的来源（local vs edge）。

### 3.1.1 在真实 trace 上先验证归属规则

在写模型之前，直接对原始 `traces.csv` 做了单 case 的中位数对比
（注入前 vs 注入后，只取目标服务的直接父子边；单位 ms）。
这不是完整审计，只用于确认“时间到底落在谁身上”：

| case | 故障 | 子 span 时长 | pre-gap | post-gap | 父 self |
|---|---|---:|---:|---:|---:|
| `ts-train-service_delay/1` | delay | 4.5 → **405.1** | 1.5 → 1.5 | 0.7 → **200.8** | 2.2 → **202.2** |
| `ts-route-service_delay/2` | delay | 2.8 → **203.3** | 1.3 → 1.3 | 0.6 → **200.6** | 1.8 → **201.9** |
| `checkoutservice_delay/1`（OB） | delay | 99.6 → **2249.9** | 1.8 → 1.7 | 1.4 → **201.5** | 3.3 → **203.3** |
| `ts-travel-service_cpu/3` | CPU | 9.1 → **25.0** | 1.5 → 1.6 | 0.6 → 0.6 | 1.9 → 2.1 |
| `ts-travel-service_disk/3` | disk | 11.9 → **24.7** | 2.1 → 2.2 | 0.2 → 0.4 | 2.4 → 2.6 |
| `ts-auth-service_cpu/2`（根 span） | CPU | 根时长 220 → **748**，其中根 self 217 → **716** | — | — | — |
| `ts-auth-service_mem/1`（根 span） | mem | 根时长 226 → **1095**，其中根 self 222 → **1000** | — | — | — |
| `ts-train-service_loss/1` | loss | 4.1 → 4.5 | 1.3 → 1.4 | 0.6 → 0.8 | 2.0 → 2.2 |
| `ts-route-service_loss/2` | loss | 2.8 → 2.9 | 1.3 → 1.3 | 0.6 → 0.7 | 1.8 → 1.9 |

四条直接结论：

1. **delay 的等待主要出现在子 span 时长和 `post-gap`（返回路径），`pre-gap` 不变；TT 与 OB 都复现。**
   200ms 是 TT 的注入量级，OB 的子 span 达 2.25s，但归因结构相同。
   所以“把 post-gap 归给被调用方”不是猜测，是这份数据的直接事实；
   而 Gate A 的 exclusive = 父 self，恰好把 post-gap 算给了调用方，delay 因此排错。
2. **CPU/disk 只抬高服务自身时长与 self，不抬高 gap。** 这部分用 local 归属即可。
3. **根 span 没有父节点，但根服务的 self 仍然可测**（auth 的 CPU/mem 把根 self 抬高
   3–4.5 倍），所以 `local` 归属必须独立于父子边存在。
4. **loss 的时延签名很弱（+4% ~ +9%）**，CPCA 可能对 loss 无效；
   loss 必须单独报告，或补 span 数/超时/重试类特征，不能假设它和 delay 同机制。

由此把默认归属定为 `local → 自身`、`pre-gap / post-gap → 被调用方`。
“pre-gap 归调用方”只作为审计变体：**父服务在调用前的自身计算与 pre-gap 无法区分**，
但实测中 delay 与 CPU 都没有明显抬高 pre-gap（CPU 的 travel 案例只 +9%），
所以这个混淆在第一版里是二阶的；若后续看到 CPU 的 pre-gap 异常，再启用“多边一致性”变体。

复现（每次只读一个 case 的 trace，约 1 分钟）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_span_cost_signature.py \
  --dataset tt --case ts-train-service_delay/1 --service ts-train-service

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_span_cost_signature.py \
  --dataset tt --case ts-auth-service_cpu/2 --service ts-auth-service
```

### 3.2 入口时延与松弛量：用重放，不用最长路 DP

入口时延就是根 span 的时长，而每个 span 的时长递归定义为：

$$
D(\tau)=d_{\text{root}},\qquad
d_s=\text{local}_s+\sum_{c\in ch(s)}\text{gap}_s(c)
+\left|\bigcup_{c\in ch(s)}[\text{start}_c,\text{end}_c]\right|
$$

其中 `local_s + Σ gap_s(c) = self_s`，gap 是父 self 连通段按相邻调用分配的互不重叠子段。
注意这**不是**“把全树 local 求和”：并发子区间重叠时，按整段时长求和会重复计数
（例如两个各 10ms、重叠 5ms 的子调用，并集是 15ms，而路径和是 20ms）。
正确做法是保留区间语义，用**递归重放**计算 $D$ 与反事实 $D^{cf}$：

```text
replay(node, v):
    for c in children(node):
        dur_c = replay(c, v)                     # 自底向上
    # v 自己的 local 用正常参考替换；边 (node->c) 的 gap
    # 只在 “c 属于 v” 时替换（因为 gap 归被调用方）
    local = normal_local(node) if node.service == v else obs_local(node)
    gaps  = [normal_gaps(node, c) if c.service == v else obs_gaps(node, c)
             for c in children(node)]
    intervals = place_children(children(node), dur_c, gaps)
    return local + sum(gaps) + union_length(intervals)
```

在“其他 span 的相对开始时刻不变”的假设下，这个重放是精确的；
它同时覆盖了串行和并发，不需要最长路 DP。`slack` 定义为

- `slack_v(τ) = D(τ) − D_{−v}(τ)`，其中 $D_{−v}$ 是把 $v$ 的 local 与 gap 全部置零后的重放结果；
- 若 $v$ 不在该请求的任何 span 上，则 `slack_v = 0`。

`slack` 就是“这个服务是否真的在决定入口时延”。并发分支上非关键的慢服务
（典型症状服务）会有很大的异常偏差但 `slack≈0`；根因则有高偏差且 `slack>0`。
这正是异常大小排序无法表达、而结构重放可以直接算出的量。

### 3.3 反事实与分数

对服务 $v$，把它的成本替换为同 `(service, operation)`（边则用
`(parent_service, parent_op, child_service, child_op)`）的正常参考中位数，
重放得到 $D^{cf}_v(\tau)$：

$$
\Delta_v(\tau)=D(\tau)-D^{cf}_v(\tau),\qquad
r_v=\frac{\sum_\tau \Delta_v(\tau)}{\sum_\tau D(\tau)}
$$

$r_v$ 的语义是“本窗口入口总时延中，可通过把 $v$ 恢复正常移除的比例”，
单位一致、跨服务可比、跨请求可按流量加权。实现上有两个简化：

1. **只有位于关键区间上的服务才有非零 $\Delta$。** 对串行链，
   $\Delta_v=\min(\delta_v,\text{slack}_v)$；对并发分支，若 $v$ 的区间完全被其他
   分支覆盖，$\Delta_v=0$。重放自动得到这个结果，不需要显式判断。
2. **可以按候选局部重放。** 每个请求先做一次 decompose（得到 local/gap/slack/D），
   再对“该请求路径上出现过的服务”逐个做一次局部重放；
   候选平均 3–8 个，成本可控。不要对所有服务重放，未出现的服务 $\Delta_v=0$。

建议同时记录三个诊断量，供融合与解释使用：

- `r_v`：总体可移除比例（主排序分数）；
- `cov_v`：经过 $v$ 的入口请求时间占比；
- `slack_share_v = Σ slack_v / Σ D`：结构上的关键性。

审计时比较两种聚合：
(a) population（未经过的请求记 0，与入口 p95 的定义一致）；
(b) covered-only（只在经过 $v$ 的请求内取相对下降中位数，避免低流量服务被稀释）。
先固定比较，不按数据集选择。

### 3.4 正常参考（必须因果）

三种参考，按可用性依次回退：

1. **case 内因果滚动**（主）：只用 $t$ 之前的分钟，robust median/MAD，
   偏差 winsorize 到 3 倍尺度；冷启动时向训练期参考收缩
   （$\hat m = \lambda m_{case}+(1-\lambda)m_{train}$，$\lambda$ 由已观测分钟数决定）。
2. **训练期正常参考**（对照）：只在训练折的正常窗口上估计。
3. **case 注入前 10 分钟**（仅上界对照）：Gate A 用的口径，用来量化乐观偏差，
   不能作为正式结果。

要求在审计报告中同时给出 1 与 3 的差距；如果差距很大，说明方法依赖了
“知道哪段是正常”的隐含信息，必须改用 1。

### 3.5 可靠性收缩（解析式，不学习）

历史上前几版学习门控全部饱和到 1（0.986–1.000），因为门控没有独立监督。
CPCA 的可靠性直接由可观测统计量给出：

$$
\tilde r_v = r_v \cdot \min\!\left(1,\frac{cov_v}{\kappa}\right)\cdot \exp(-\gamma\cdot \widehat{se}_v)
$$

- `se_v`：对窗口内请求做 bootstrap 得到的 $r_v$ 标准误；
- `κ`：最小有效覆盖率（建议 0.05，验证集选定）；
- `γ`：固定小常数，不按数据集调。

当覆盖率低、请求少、span 解析率低或 bootstrap 区间跨 0 时，$\tilde r_v\to 0$，
排序自动退回神经排序头。**这是“路径证据不可靠时严格回退”的实现，
不是再学一个门。**

### 3.6 与预测器的组合

三档，按顺序验证，不一次全上：

| 方案 | 公式 | 说明 |
|---|---|---|
| A. CPCA-only | 按 $\tilde r_v$ 排序 | 零训练，用来验证机制本身 |
| B. Rank fusion | RRF(LTFP 排序, CPCA 排序) | 两个来源都保留，无参数 |
| C. Bounded residual | $\text{logit}^{LTFP}_v+\lambda\tanh(\beta\tilde r_v)$ | $\lambda$ 只在验证集从 `{0,0.5,1,2}` 选，允许 0 |

主结果先报 A 和 B；C 只在 A/B 显示互补错误时才做，并且必须报“同 checkpoint 关闭残差”的
事后消融（参考 directional residual 的做法），避免把联合训练轨迹差异当成方法收益。

### 3.7 一个具体例子

`frontend → checkout → payment`，在 payment 上注入 200ms 网络延迟：

- payment 的 span duration 可能只小幅上升；
- checkout 的 self time（等待 payment 返回的空档）大幅上升；
- 现有 exclusive 分数会把 checkout 排到第一（Gate A 在 delay 上正是这个现象）；
- CPCA 把 checkout→payment 的 `gap_pre/gap_post` 归给 payment，
  恢复 payment 的边等待后重放，入口时延下降最多的是 payment，排序正确。

若注入的是 payment 的 CPU 故障，则 payment 的 `local` 上升，同样归给 payment。
**同一套分数覆盖两类机制，这是它相对单一 exclusive/inclusive 指标的结构性优势。**

## 4. 与已失败方向的区别

| 已失败方向 | 失败机制 | CPCA 的不同 |
|---|---|---|
| operation exclusive / inclusive（Gate A） | 单一量，且边等待归错对象 | 用结构重放同时使用 local 与 edge，不需要二选一 |
| 有向图 / directional residual | 方向只是关系，1 分钟内根与邻居同时越阈值 | 不学传播，用区间守恒直接算关键路径 |
| normal-restoration v0 | 在**模型输入**上恢复，受预测器依赖影响（Hit@1 0.31/0.56） | 在**结构时延**上恢复，不改变预测器 |
| CIRCA 类因果图方法 | 需要学习/发现因果图 | 观测 DAG 本身就是结构方程，不需要发现 |
| candidate evidence verifier | 55 个事件上训练小网络，方差大于收益 | 零训练参数，只有 2–3 个解析超参 |
| fault-type aux | 故障类型不可识别（OB config 精度 0.028） | 不分类；local/edge 分解天然给出机制解释 |
| stage route / task graph | 仍在共享表示与梯度上做文章 | 归因器与预测器解耦，不共享表示 |
| null anchor / RC² / joint granularity | 把预测与定位绑在一个标量上 | 预测与归因是两个可独立验证的模块 |

**与相关工作必须显式比较的对象**（baselines/ 已 vendored）：
CIRCA、TraceRCA、MicroRank、TVDiag、Hades、RCD、DiagFusion。
CPCA 的主张不是“分离根因与症状”（Topo-MFP/DeST 已覆盖），
而是“用请求 DAG 的结构方程估计可移除时延这一反事实量”。
这个 delta 必须在 related work 里逐条写清，否则会被审稿人认为是 CIRCA 的变体。

## 5. 验证计划（Gate D）

### Step 0：先补齐可比较的基线（1 周，不做模型）

- 10 个 specialist 与全部框架 baseline 的正式五折结果（当前只有 1-epoch smoke）；
- LTFP / StageRoute / InstantOps / Eadro / persistence 全部补 3 个种子；
- 所有实验输出落到 `artifacts/`（当前 stage route、directional、OC-SID、
  candidate verifier 的结论只存在于 `/private/tmp` 和文档里，仓库内没有 metrics，
  不可复现也不能进论文）；
- 评估改为：事件级为主、窗口级为辅、逐折范围 + 配对 bootstrap CI。

### Step 1：CPCA 离线审计（不训练，1–2 周）

复用 `scripts/audit_operation_source_impact.py` 的流式 trace 解析，
新增 `scripts/audit_critical_path_counterfactual.py`：

1. 对每个 case 的每分钟重建 span 成本树，输出 per-request 的
   `local / gap / slack / D`；
2. 计算 Gate A 的三个基线分数（global mean duration、inclusive、exclusive）作为对照；
3. 计算 CPCA 的变体：local-only、edge-only、local+edge；edge 再分
   `both gaps→child`（默认）、`pre-gap→caller`、多边一致性三种归属；
4. 正常参考分别用 case 滚动 / 训练期 / oracle 前 10 分钟；
5. 评价：事件级 Hit@1/MRR、首分钟、逐故障类型、无影响 case 的源识别。

**预先固定的通过线：**

- CPCA 在 OB 和 TT 都必须优于三个单视图基线；
- 与 LTFP 融合后，至少一个数据集事件级 MRR 提升 ≥ `+0.02`，
  且配对 bootstrap 95% CI 不含 0；
- 另一个数据集下降 ≤ `0.01`；
- 换 3 个种子后方向不变（CPCA 本身确定性，变化来自 LTFP 侧）。

### Step 2：若通过

- 集成 A/B 两种组合，做同 checkpoint 消融；
- 逐故障类型表：预期 CPU/disk 由 local 主导，delay/loss 由 edge 主导，
  memory/socket 介于两者之间。这张表是机制成立与否的关键证据；
- 跨数据集：OB 与 TT 分别报告，不做“平均后领先”的表述。

### Step 3：若不通过

按 `2026-11-23-re2-data-driven-method-design.md` 的停损线执行：
停止在 RE2 上继续设计复杂 RCA 结构，把论文定位为
“严格任务定义 + 完整基线（10 specialist）+ 数据边界与负结果分析”。
这仍然是一篇可投的 measurement/benchmark 论文，而不是失败。

## 6. 预期收益与风险

### 6.1 合理的预期

- TT：Gate A 的“global + inclusive + exclusive”组合把事件 Hit@1 从 `0.719` 提到 `0.875`，
  说明路径证据在 TT 上有真实增量；但那是相对弱规则，相对 LTFP（`0.894/0.932`）
  的增量会更小，合理预期是事件 MRR `+0.02 ~ +0.05`。
- OB：Gate A 的组合只改善 Hit@1（`0.345→0.436`），MRR 基本持平，
  且首分钟变差；OB 的 trace 只有固定 17 个 service-operation 序列，
  operation 区分度低。**OB 有可能不涨，这是可以接受但要如实报告的。**
- 最大价值可能不在均值，而在**逐故障类型的一致性**：
  如果 delay 类从 `0.455` 提升到 0.8 以上，机制主张就成立，即使总体均值只动一点。

### 6.2 风险

| 风险 | 表现 | 缓解 |
|---|---|---|
| span 采样/时钟偏斜 | 子 span 早于父 span、缺父节点 | 解析率作为可靠性项；裁剪负 gap |
| 排队反馈 | 恢复 v 后其他服务时延也会变 | 明确写成假设；对 $\Delta$ 做敏感性上限 |
| 低流量根因 | population 加权后 $r_v$ 很小 | 同时报 covered-only；覆盖率门槛 |
| 事件太少 | +0.03 也可能不显著 | 事件级配对检验；报每事件 win/loss；三种子 |
| loss 类时延签名弱 | 实测只 +4%~+9%，CPCA 可能无效 | 单独报告 loss；必要时补 span 数/超时/重试特征 |
| 与 CIRCA 撞车 | 审稿人认为只是反事实 RCA | related work 逐条 delta；强调 DAG 结构方程与 slack |

## 7. 这个方案里“模型”到底在哪

如果目标是深度学习会议/期刊，纯解析归因会被质疑“没有模型”。可学习的位置只有两个，
且都必须保持低容量：

1. **融合层**：把 `[r_v, cov_v, slack_share_v, LTFP logit_v, metric deviation_v]`
   送入一个 4–6 参数的单调打分器，用事件级 pairwise ranking loss 训练，
   非负约束 + 强正则；每个训练折只有约 40–50 个事件，参数超过 10 个就不可信。
2. **正常参考层**：用分位数回归或轻量状态空间模型估计每个
   `(service, operation)` 的正常时延分布，替代中位数/MAD。
   这是唯一有足够样本（数十万请求）支撑的学习点。

编码器、图传播、门控都不应该是这一版的创新点——它们已经被证明在这份数据上不稳定。

## 8. 不做清单

- 不再加图方向、边门控、节点门控、任务独立 GAT；
- 不再加 fault-type、forecast、transition 辅助头；
- 不再做端到端联合训练（joint granularity 已证明 AP 双降）；
- 不用单种子五折均值下结论；
- 不用 `/private/tmp` 里没有落盘的结果；
- 不在测试折上选阈值、选 $\lambda$、选 checkpoint。

## 9. 两周任务清单

| 天 | 任务 | 产出 |
|---|---|---|
| 1–2 | 补齐 persistence 基线、3 seeds、10 specialist 正式五折 | `artifacts/` 可复现表 |
| 3–5 | span 成本树构建（local/gap/slack/D） | `artifacts/re2_{ob,tt}_span_costs_v1` |
| 6–8 | CPCA 三种变体 + 三种正常参考审计 | `scripts/audit_critical_path_counterfactual.py` |
| 9 | 逐故障类型、首分钟、无影响 case 分析 | 审计报告 |
| 10 | Gate D 判定 | 通过 / 停止 |
| 11–14 | 若通过：融合 A/B + 同 checkpoint 消融 + 三种子 | 论文主表 |

## 10. 复现入口（计划）

```bash
# Step 0：统一基线（该入口的种子参数是 --seed，单数，需外层循环）
for seed in 42 123 2026; do
  uv run python scripts/run_re2_baseline_matrix.py \
    --models instantops-adapted ltfp eadro-adapted --datasets ob tt \
    --folds 0 1 2 3 4 --epochs 30 --seed "$seed"
done

# Step 1：CPCA 审计
PYTHONDONTWRITEBYTECODE=1 baselines/.venv/bin/python \
  scripts/audit_critical_path_counterfactual.py \
  --datasets ob tt --folds 0 1 2 3 4 \
  --normal-reference rolling \
  --output artifacts/re2_cpca_gate_d_v1.json
```

实现时优先复用：

- `src/data/trace_paths.py`：`Span`、`interval_union_length`、`service_latencies`、`entry_span`；
- `scripts/audit_span_cost_signature.py`：单 case 归属验证（§3.1.1 的证据，已实现）；
- `scripts/audit_operation_source_impact.py`：流式 zip 解析、case→member 映射、robust z；
- `src/training/metrics.py`：`rca_metrics`（事件级聚合已在
  `src/training/experiment.py:_event_rca_metrics` 实现）。

## 11. 备选创新线（CPCA 不通过时）

三条线可以独立成立，按性价比排序：

### B1. 跨系统零样本 RCA（最容易被忽略、也最容易被审稿人认可）

- 训练 TT（27 服务）、测试 OB（11 服务），反之亦然；
- 输入只用**相对自身正常参考的偏差**、服务无关的排序与覆盖率统计，
  不使用服务身份 embedding、不使用绝对资源值、不使用固定拓扑；
- 主张：现有 RCA 方法都假设固定服务集合，跨系统无法迁移；
  我们给出首个 OB↔TT 零样本结果和失败边界。
- 风险：绝对指标可能很差，但“迁移后仍显著高于随机/异常大小排序”就是一个结论；
  需要先做 1 天探针确认不是完全随机。

### B2. persistence / onset 分解 + 告警策略（配套，不足以单独成文）

- 主概率分两支：当前已退化 → persistence；当前正常 → 低容量 hazard；
- 事件级告警策略（连续 k 分钟、恢复抑制），报告检测延迟 vs 每 case 误报；
- 价值：把“下一分钟预测”从被 persistence 支配的指标里解救出来，
  让 onset 能力可被单独评价。适合作为主方法的一节，而不是全部。

### B3. 统一 benchmark / measurement 论文（保底）

- 4 个框架 baseline + 10 个 specialist + persistence + first-visible，
  全部五折三种子、事件级配对统计；
- 结论：现有方法在严格下一分钟协议下普遍失效；persistence 支配预测；
  log 近似噪声；方向/图传播/门控的增益不可复现；
- 这不是“失败论文”，而是这个方向目前最缺的东西。投
  EMSE / TOSEM / ISSRE / ICSME 一类偏系统与经验的 venue 比投 DL 会议更合适。

**建议顺序：** CPCA（Gate D）→ 若通过，B2 作为主方法的一节；
若 CPCA 不通过，B1 先做一天探针，再决定 B1 还是 B3。
三条都不要再回头做“再加一个图模块”的路线。
