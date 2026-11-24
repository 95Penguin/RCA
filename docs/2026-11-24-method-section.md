# 方法章节：OCSID 框架（带解释版）

> 本文档按论文 Method 章节组织，但每一节都先用日常语言说明“在做什么、为什么”，
> 再给公式和逐项解释。所有公式与仓库实现一致
> （`src/models/ocsid.py`、`scripts/audit_cpca_gate2.py`）。
>
> 全篇用一个例子贯穿：一次请求 `frontend → checkout → payment`，
> 入口总时延 1000 ms，`payment` 自己处理了 600 ms（正常只要 50 ms）。

---

## 0. 先读这一页

**任务。** 看过去 5 分钟的监控数据，回答两个问题：

1. 下一分钟系统会不会出现用户能感知的故障？（预测）
2. 如果会，是哪个服务导致的？（定位）

**方法一句话。** 用“服务时序 + 调用图 + 入口 operation”预测故障；
用“把某个服务恢复正常，再看这次请求能快多少”来定位根因。

**五个步骤。**

1. 把过去 5 分钟每个服务的 19 个指标编码成向量；
2. 让服务沿着调用图互相交换信息，得到“系统整体状态”；
3. 对每个入口 operation（比如“登录”“查余票”）单独估计一个风险；
4. 把“系统风险”和“operation 风险”融合成下一分钟故障概率；
5. 对已经发生的请求，逐个服务做“恢复正常”的反事实重放，谁能让入口时延降得最多，谁就是根因。

**运行例子。** 一次请求 `frontend → checkout → payment`，总耗时 1000 ms：

| 服务 | 自己的处理时间 | 等别人的时间 | 恢复正常后请求能快多少 |
|---|---:|---:|---:|
| payment | 620 ms（正常 50） | 0 | **740 ms** |
| checkout | 0 ms（正常 0） | 800 ms | **0 ms** |
| frontend | 0 ms（正常 0） | 1000 ms | **0 ms** |

异常大小排序会把三个服务都排得很高（checkout 的 span 也是 800 ms，因为它包含了 payment 的时间）。
但只有 payment 恢复正常能真正缩短请求——这就是 CPCA 要算的东西。

---

## 1. 问题定义

### 1.1 场景

我们把系统看成 $N$ 个微服务，它们通过 RPC 互相调用。每条用户请求从某个**入口 operation**
（例如 `POST /api/v1/users/login`）进入，经过若干服务，最后返回。

监控数据有三类：

- **Metric**：每个服务的 CPU、内存、磁盘 IO、socket 等资源指标；
- **Log**：每个服务每分钟的日志量、错误数；
- **Trace**：每条请求的 span 树，记录每个服务调用了谁、各自花了多久。

### 1.2 输入

对分钟 $t$，模型看过去 $T=5$ 个完整分钟：

- $\mathbf{X}_t \in \mathbb{R}^{T \times N \times F}$：每个服务每分钟 $F=19$ 个面板特征。
  这 19 个特征包括 4 个资源指标、4 个观测掩码、3 个 trace 统计（span 数、错误数、平均时延）、
  3 个日志统计、5 个入口统计（入口 p95、退化分等）。
- $\mathbf{A}_t \in \mathbb{R}^{T \times N \times N}$：服务调用图，$\mathbf{A}_{t}[u,v]>0$ 表示
  $u$ 调用了 $v$。
- $\mathbf{O}_t \in \mathbb{R}^{T \times M \times F_o}$：每个**入口 operation** 每分钟
  $F_o=17$ 个 trace 特征，包括请求数、p50/p95/p99、错误率、相对正常基线的比值、退化分等。
- $\mathbf{m}_t \in \{0,1\}^{T \times M}$：这个 operation 在该分钟是否有足够的请求。

**为什么要有 operation 这一路输入？** 因为 TT 的标签本身就是按 operation 定义的：
“同一个入口 operation 的 p95 连续两分钟超过它自己的注入前基线”。
如果把 15 个 operation 混成一个系统指标，一个 operation 的抖动会被其他 operation 稀释，
也说不清“是哪个入口先坏”。

### 1.3 输出与监督

- $p_t \in [0,1]$：下一分钟系统故障概率；
- $\hat{\mathbf{R}}_t \in \mathbb{R}^{N}$：根因服务排序。

训练标签：$y_{t+1}\in\{0,1\}$ 是下一分钟是否故障；$r_{t+1}$ 是注入服务（根因）。
每个正例窗口还带一个事件编号 $e$，用于事件级评估（见第 10 节）。

### 1.4 符号表

| 符号 | 人话 |
|---|---|
| $N$ | 服务数（OB 11，TT 27） |
| $M$ | 入口 operation 数（OB 1，TT 约 17） |
| $T$ | 看过去几分钟（5） |
| $\mathbf{X},\mathbf{A},\mathbf{O},\mathbf{m}$ | 服务指标、调用图、operation 指标、掩码 |
| $\tau$ | 一条请求 |
| $s$ | 一个 span（一次服务调用） |
| $v(s)$ | span 属于哪个服务 |
| $C(s)$ | span 的直接子 span |
| $d_s$ | span 的时长 |
| $\mathrm{self}_s$ | span 扣掉子调用区间后的“自身时间” |
| $\mathrm{local}_s$ | 自身时间里真正用于计算/IO 的部分 |
| $\mathrm{pre}_s(c)$ | 调用子 $c$ 之前的等待空档 |
| $\mathrm{post}_s(c)$ | 子 $c$ 返回之后的等待空档 |
| $D(\tau)$ | 入口时延（根 span 时长） |
| $r_i$ | CPCA 给服务 $i$ 的结构责任分 |
| $a_i$ | 学习式根因分（辅助/消融） |

---

## 2. 框架总览

### 2.1 两阶段

OCSID 分两阶段：

```text
阶段一：预测（可训练）
  X, A ──► 服务时序编码 ──► 图传播 ──► p_sys ─┐
  O, m ──► operation 编码 ──► q_o ──► p_op ──┴──► p

阶段二：归因（零训练）
  traces ──► span 树 ──► 反事实重放 ──► r_i ──► 根因排序
```

### 2.2 为什么不做成一个端到端网络

这是数据逼出来的选择，不是省事：

- 预测需要图传播，因为“上游在等下游”这类信息只有跨服务才能看到；
- 但定位需要**请求级**的结构归属。图传播会把根因的局部异常扩散到症状服务，
  学出来的根因分反而会跟着症状走。
- 我们试过把两者合成一个网络：学习式根因分在 OB 上只有 0.738/0.833（事件级），
  而结构式 CPCA 是 0.855/0.920。融合曲线也显示，学习分的权重越大越差，
  最优是“完全用 CPCA”。

所以最终框架是：**学习式预测 + 结构式归因**。两者共享同一个时间窗口和同一批 trace，
但一个负责“会不会坏”，一个负责“为什么坏”。

---

## 3. 阶段一：预测模块

### 3.1 服务时序编码

**在做什么。** 对每个服务，把它过去 5 分钟的 19 个指标压成一个向量，
表示“这个服务现在是什么状态”。

**怎么做。** 按模态分组（metric / log / trace），每组过一个因果卷积网络（TCN），
再用时间注意力把 5 个时刻汇总：

$$
\mathbf{h}^{\text{local}}_i
= \mathrm{Fuse}\Big(
\mathrm{TCN}_m(\mathbf{X}_{i,t-T+1:t}^{(m)}) \ \big|\ m\in\{\text{metric},\text{log},\text{trace}\}
\Big).
\tag{1}
$$

**逐项读法。**

- $\mathbf{X}_{i,t-T+1:t}^{(m)}$：服务 $i$ 第 $m$ 类模态的 5 分钟序列；
- $\mathrm{TCN}_m$：因果卷积，保证只用过去，不看未来；
- $\mathrm{Fuse}$：模态门控 + 拼接，把三路表示合成一个 64 维向量；
- $\mathbf{h}^{\text{local}}_i$：只描述服务 $i$ 自己，还没和邻居交互。

**为什么按模态分。** 三类数据的量纲和噪声完全不同：CPU 是 0–1 的利用率，
日志是每分钟几百条，trace 是毫秒级时延。混在一起过一层网络，弱模态会淹没强模态。
分组编码再融合，至少让模型自己学“这个窗口该信谁”。

### 3.2 图传播

**在做什么。** 让每个服务看一眼它的调用方和被调用方，把邻居的状态融进来。

**怎么做。** 在调用图上做两层图注意力（GAT）：

$$
\mathbf{h}^{(\ell+1)}_i
= \mathrm{LN}\Big(
\mathbf{h}^{(\ell)}_i
+ \mathrm{Dropout}\big(\mathrm{GAT}(\mathbf{h}^{(\ell)}, \mathbf{A}_t)\big)
\Big),\quad \ell=1,2.
\tag{2}
$$

**逐项读法。**

- $\mathbf{h}^{(0)}_i=\mathbf{h}^{\text{local}}_i$；
- $\mathrm{GAT}$：按邻居的重要性加权求和，权重由注意力学出来；
- 残差 $\mathbf{h}^{(\ell)}_i+\dots$ 和 LayerNorm：防止深层图网络把自身信息冲掉；
- 输出 $\mathbf{h}^{\text{graph}}_i$ 同时包含自己和对端的信息。

**为什么需要图。** 举一个预测的例子：payment 变慢时，checkout 的时延也会变慢，
但 checkout 自己并没有问题。如果模型只看每个服务自己的指标，它会把 checkout 也当成异常；
看了调用图之后，它有机会学到“checkout 的慢是从 payment 传来的”。
消融实验也支持这一点：去掉图之后，TT 的 AP 从 0.853 掉到 0.824。

**为什么 OB 上图不重要。** OB 只有一个入口 operation，拓扑简单、调用链短，
图能提供的额外关系很少；去掉图后 OB 的 AP 只从 0.968 掉到 0.965。
这是数据集边界，不是方法缺陷。

### 3.3 系统风险头

**在做什么。** 把所有服务压成一个“系统状态”，再判断系统会不会坏。

$$
\mathbf{g}
= \mathrm{MLP}\Big(\big[\tfrac{1}{N}\textstyle\sum_i \mathbf{h}^{\text{graph}}_i,\
\max_i \mathbf{h}^{\text{graph}}_i\big]\Big),
\qquad
\mathbf{z}^{\text{sys}} = \mathrm{MLP}(\mathbf{g}) \in \mathbb{R}^{2},
\tag{3}
$$

$$
p_{\text{sys}} = \mathrm{softmax}(\mathbf{z}^{\text{sys}})_1 .
\tag{4}
$$

**逐项读法。**

- 先对所有服务做平均池化和最大池化：平均池化看“整体水位”，最大池化看“最严重的那个服务”；
- 拼起来过一个 MLP 得到系统表示 $\mathbf{g}$；
- 再接一个二分类头，输出“正常/故障”两个 logit，取故障类的 softmax 概率。

**为什么要 mean + max。** 只用 mean 会漏掉“一个服务炸了但其他都正常”的情况；
只用 max 又会被单个噪声服务带偏。两者拼接是最便宜也最稳的做法。

### 3.4 Operation 风险头

**在做什么。** 对每个入口 operation 单独判断“它下一分钟会不会退化”。

**怎么做。** 所有 operation 共享一个 GRU：

$$
\mathbf{u}_o = \mathrm{GRU}\big(\mathbf{O}_{o,t-T+1:t}\big),\qquad
q_o = \sigma\big(\mathbf{w}_o^\top \mathrm{LN}(\mathbf{u}_o)\big).
\tag{5}
$$

**逐项读法。**

- $\mathbf{O}_{o,t-T+1:t}$：operation $o$ 过去 5 分钟的 17 维特征；
- GRU 把它压成 32 维；
- 一个线性层 + sigmoid 输出 $q_o\in[0,1]$，表示这个 operation 下一分钟退化的概率。

系统级 operation 风险有两种聚合：

$$
p_{\text{op}} =
\begin{cases}
\max_{o:\,m_o=1} q_o, & \text{(max)}\\[2pt]
1-\prod_{o:\,m_o=1}(1-q_o), & \text{(product)}
\end{cases}
\tag{6}
$$

**怎么选。** 默认用 max：系统的定义是“任一入口 operation 持续退化”，
所以“最危险的那个入口”最能代表系统。product 假设各 operation 独立，
但 TT 的 17 个 operation 往往由同一个根因同时影响，product 会系统性高估；
它只作为消融。

**为什么 operation 这一路必要。** 消融显示，去掉 operation 风险头后，
OB 的 AP 从 0.968 掉到 0.940，TT 从 0.853 掉到 0.819。
一方面它提供了 per-operation 的监督信号，另一方面它让预测可解释：
“是登录入口风险 0.8，不是查票入口”。

### 3.5 融合

**在做什么。** 把“系统图视角”和“入口 operation 视角”合成一个概率。

$$
p = \sigma\Big(
w_1 \cdot \mathrm{logit}(p_{\text{sys}})
+ w_2 \cdot \mathrm{logit}(p_{\text{op}})
\Big).
\tag{7}
$$

**逐项读法。**

- $\mathrm{logit}(p)=\log\frac{p}{1-p}$ 把概率变到实数轴；
- 两个视角在 logit 空间加权求和，再 sigmoid 回概率；
- $w_1,w_2$ 由训练学出来，模型自己决定更信哪一边。

**为什么在 logit 空间融合。** 概率空间直接加权会被 0 和 1 附近饱和；
logit 空间是线性的“证据相加”，更稳定。

### 3.6 辅助 source–impact 头（只用于解释和消融）

**在做什么。** 给每个服务算一个“自身证据 + 传播上下文”的学习式根因分，
方便解释，也作为 CPCA 的对照。

$$
\mathbf{s}_i = \mathrm{MLP}\big(\mathbf{h}^{\text{local}}_i\big),\qquad
\mathbf{e}_i = \mathrm{MLP}\big(\mathbf{h}^{\text{graph}}_i\big),
\tag{8}
$$

$$
a_i = \mathrm{MLP}\big([\mathbf{s}_i,\ \mathbf{e}_i,\ \mathbf{s}_i\odot\mathbf{e}_i]\big).
\tag{9}
$$

**逐项读法。**

- $\mathbf{s}_i$：只看服务自己的局部表示，回答“它自己有没有异常”；
- $\mathbf{e}_i$：看图传播后的表示，回答“它在不在受影响的上下文里”；
- $\mathbf{s}_i\odot\mathbf{e}_i$：两者的交互项；
- $a_i$：一个标量根因分。

**重要：最终排序不用 $a_i$。** 融合实验（第 6 节）显示，
$a_i$ 和 CPCA 的 $r_i$ 高度冗余，而且 $a_i$ 单独用更差。
保留它只是为了可解释性和消融对照。

### 3.7 一个预测例子

假设当前窗口：

- 登录 operation 的 p95 是 250 ms，和正常一样 → $q_{\text{login}}\approx 0.1$；
- 查余票 operation 的 p95 是 800 ms，正常 400 ms → $q_{\text{query}}\approx 0.9$；
- 图上看，checkout 和 payment 的 trace 时延都在上升 → $p_{\text{sys}}\approx 0.7$；
- 融合后 $p\approx 0.8$，并且模型能说出“是查余票这个入口风险高”。

---

## 4. 阶段二：CPCA 结构反事实归因

### 4.0 直觉：修好再重放

CPCA 只做一件事：

> 假设把服务 $i$ 恢复正常，重新走一遍这次请求，入口时延会变成多少？
> 下降得越多，$i$ 越像根因。

它不需要训练，因为请求的 span 树本身就给出了“时间是怎么组成的”。

### 4.1 Span 树与三类时间

**先看一个真实请求。** `frontend → checkout → payment`：

| span | 总时长 $d_s$ | 子调用并集 | 自身时间 $\mathrm{self}_s$ | local | pre | post |
|---|---:|---:|---:|---:|---:|---:|
| payment | 600 | 0 | 600 | 600 | — | — |
| checkout | 800 | 600 | 200 | 20 | 10 | 170 |
| frontend | 1000 | 800 | 200 | 0 | 20 | 180 |

每一行都满足：

$$
d_s = \mathrm{self}_s
+ \Big|\bigcup_{c\in C(s)} [\mathrm{start}_c,\ \mathrm{end}_c]\Big|.
\tag{10}
$$

**逐项读法。**

- $d_s$：这个 span 从开始到结束的总时长；
- 子调用并集：所有直接子 span 占用的时间区间（并发时取并集，不重复算）；
- $\mathrm{self}_s$：总时长里“不属于任何子调用”的部分。

$\mathrm{self}_s$ 里其实混了两种东西：服务自己算的时间，和它等子调用返回的时间。
所以再拆一次：

$$
\mathrm{self}_s
= \mathrm{local}_s
+ \sum_{c\in C(s)} \big(\mathrm{pre}_s(c) + \mathrm{post}_s(c)\big).
\tag{11}
$$

**逐项读法。**

- $\mathrm{local}_s$：真正用于计算/IO 的时间；
- $\mathrm{pre}_s(c)$：从父 span 发起调用，到子 span 开始之间的空档
  （请求在路上 + 对端排队）；
- $\mathrm{post}_s(c)$：子 span 结束，到父 span 继续之间的空档（返回在路上）。

在例子里：

- payment 的 620 ms 全是 local；
- checkout 的 self = 180 ms = local 0 + pre 10 + post 170；
- frontend 的 self = 200 ms = local 0 + pre 20 + post 180。

**为什么拆这么细。** 因为不同类型的故障落在不同的项里：

- CPU / 内存 / 磁盘：抬高服务的 **local**；
- 网络 delay / loss：抬高调用方的 **pre/post gap**；
- 症状服务：自己的 local 正常，只是它的 span 包含了子服务的时长。

如果只用一个“服务平均时延”，这三种情况分不开。

### 4.2 归属规则：时间算在谁头上

对服务 $i$，把“它自己的 local”和“进出它的边等待”加在一起：

$$
c_i(\tau)
= \sum_{s:\,v(s)=i} \mathrm{local}_s
+ \sum_{c:\,v(c)=i} \big(\mathrm{pre}_{p(c)}(c) + \mathrm{post}_{p(c)}(c)\big).
\tag{12}
$$

**逐项读法。**

- 第一项：所有属于 $i$ 的 span 的 local；
- 第二项：所有“被调用方是 $i$”的边，把父 span 上的 pre/post 空档算给 $i$；
- $p(c)$ 是 $c$ 的父 span。

**一句话：local 归自己，进出自己的边等待也归自己。**
这样所有服务的成本加起来正好等于整棵树的 self 时间，不重不漏。

**为什么边等待归被调用方。** 我们直接在 RE2 原始 trace 上验证过：
delay 注入时，`post-gap` 从 0.7 ms 涨到 200.8 ms，而 `pre-gap` 完全不变；
CPU/disk 注入只抬高服务自身时长与 local。
如果把这段等待留给调用方（Gate A 的 exclusive 做法），delay 类故障就会被算到调用方头上，
根因排错——这正是 Gate A 在 delay 上失败的原因。

### 4.3 入口时延与 slack

入口时延就是根 span 的时长：

$$
D(\tau) = d_{\text{root}(\tau)}.
\tag{13}
$$

对服务 $i$ 定义**松弛量**：

$$
\mathrm{slack}_i(\tau) = D(\tau) - D_{-i}(\tau),
\tag{14}
$$

**逐项读法。**

- $D_{-i}(\tau)$：把 $i$ 的 local 和进出它的 gap 全部置零后，重放得到的入口时延；
- $\mathrm{slack}_i$：$i$ 的“成本上限”——把它的成本整段清零，最多能省多少入口时延。

**它解决什么问题。** 它回答“这个服务是否在关键路径上”：

- $\mathrm{slack}=0$：它不在关键路径上，恢复它也不会让请求变快（比如并行分支上较短的服务）；
- $\mathrm{slack}>0$：它确实在决定入口时延，最多能移除 $\mathrm{slack}$ 这么多。

**但 slack 不是最终的根因判据。** 一个服务可以 $\mathrm{slack}>0$ 却 $\Delta=0$：
它在关键路径上，但自身成本本来就在正常范围（比如 checkout，它只是等 payment）。
真正区分“根因”和“正常服务/症状”的是 $\Delta$（恢复正常后能移除多少）：

- 根因：$\Delta$ 大（自身异常，且在关键路径上）；
- 症状：$\Delta\approx0$（要么自身正常，要么 $\mathrm{slack}=0$）。

例子里：

- payment：$\mathrm{slack}=800$ ms，$\Delta=740$ ms（自身异常 + 在关键路径）；
- checkout：$\mathrm{slack}=200$ ms（它毕竟有 200 ms 自己的成本），但 $\Delta=0$（成本本来正常）；
- frontend：$\mathrm{slack}=0$（根节点没有入边，自身 local 也是 0）；
- 并行分支的 recommendation 即使慢 500 ms，只要 checkout 分支 800 ms 更长，
  它的 $\mathrm{slack}=0$，$\Delta=0$。

### 4.4 反事实重放

对服务 $i$，把它的 local 和进出它的 gap 换成正常参考，然后自底向上重算整棵树：

$$
\widetilde{\mathrm{local}}_s =
\begin{cases}
\mathrm{med}^{\text{normal}}_{v(s),\,op(s)}, & v(s)=i,\\
\mathrm{local}_s, & \text{otherwise},
\end{cases}
\qquad
\widetilde{\mathrm{gap}}_s(c) =
\begin{cases}
\mathrm{med}^{\text{normal}}_{\text{edge}}, & v(c)=i,\\
\mathrm{gap}_s(c), & \text{otherwise},
\end{cases}
\tag{15}
$$

$$
\tilde d_s = \widetilde{\mathrm{local}}_s
+ \sum_{c\in C(s)} \big(\widetilde{\mathrm{pre}}_s(c) + \tilde d_c + \widetilde{\mathrm{post}}_s(c)\big).
\tag{16}
$$

**逐项读法。**

- 如果 span 属于 $i$，它的 local 用正常中位数替换；
- 如果边的被调用方是 $i$，这条边的 pre/post 用正常中位数替换；
- 其他 span 和边保持观测值；
- 从叶子往上重算每个 span 的时长，最后得到新的入口时延 $\tilde D_i(\tau)$。

**例子里恢复 payment：**

| 量 | 观测 | 正常 | 重放后 |
|---|---:|---:|---:|
| payment local | 620 | 50 | 50 |
| checkout→payment pre | 10 | 5 | 5 |
| checkout→payment post | 170 | 5 | 5 |
| checkout local | 0 | 0 | 0 |
| frontend→checkout pre/post | 20 / 180 | 20 / 180 | 20 / 180 |
| 入口时延 | 1000 | — | **260** |

所以 $\Delta_{\text{payment}}=740$ ms。恢复 checkout 或 frontend 时，重放结果不变，
$\Delta=0$。

**为什么串行重放是精确的。** 在抽查的 RE2-TT case 上，同一父节点下的子调用完全不重叠
（18,099 个多子节点，overlap 全为 0），所以父 span 的时长就等于
“local + 每个子调用的 pre + 子时长 + post”依次相加，不需要处理并发并集。

### 4.5 责任分

$$
\Delta_i(\tau) = D(\tau) - \tilde D_i(\tau),
\qquad
r_i = \frac{\sum_{\tau} \Delta_i(\tau)}{\sum_{\tau} D(\tau)}.
\tag{17}
$$

**逐项读法。**

- $\Delta_i(\tau)$：恢复 $i$ 后，这次请求能快多少毫秒；
- $r_i$：本窗口所有请求里，恢复 $i$ 能移除的入口时延占比，范围 $[0,1]$。

例子里只有一个请求，$r_{\text{payment}}=740/1000=0.74$，checkout 和 frontend 都是 0。
所以排序是 payment 第一。

**为什么用比例而不是毫秒。** 不同服务的正常时延量级差很多（login 250 ms，查票 50 ms），
直接用毫秒会让高延迟服务占便宜；用“占入口时延的比例”就跨服务可比。

### 4.6 Operation 条件化

TT 的标签是按 operation 定义的，所以每个窗口先选“当前最异常、请求数又够”的入口 operation：

$$
o^\star_t
= \arg\max_{o:\,n_o(t)\ge n_{\min}}
\frac{\mathrm{p95}_o(t)}{\mathrm{p95}_o^{\text{normal}}},
\tag{18}
$$

**逐项读法。**

- $n_o(t)$：这个 operation 在当前窗口的请求数，要求至少 20；
- 分子是当前 p95，分母是它在正常期的 p95；
- 硬选择版本只在这个 operation 的请求上算 $r_i$；
- 默认实现用软加权，避免“根因不在最风险 operation 上”时丢分：

$$
w_o=\frac{\exp(\mathrm{ratio}_o/\tau)}{\sum_{o'}\exp(\mathrm{ratio}_{o'}/\tau)},
\qquad
r_i=\sum_o w_o\,r_i^{(o)},\qquad \tau=1.
\tag{18a}
$$

**为什么。** 如果把 17 个 operation 的请求混在一起，根因只出现在其中一部分请求里，
它的效应会被稀释。硬选择只保留一个 operation，简单但依赖 argmax；
软加权按风险比例保留所有 operation，根因不在最风险 operation 上时仍有分数。
OB 只有 1 个入口 operation，这一步自动退化为全量；
所以 OB 的增益不是 operation 选择带来的，而是归属规则带来的。

TT 事件级：硬选择 $0.952/0.974$（63 个事件），软加权 $0.953/0.974$（64 个事件），
多覆盖一个事件，MRR 持平；OB 两者一致。

### 4.7 覆盖率与回退

如果根因不在所选 operation 的路径上，CPCA 根本看不到它。这时用覆盖率
$c_i$（经过 $i$ 的请求时延占比）判断：

- 覆盖足够：用 $\bar r_i$ 排序；
- 覆盖不足：部署时回退到学习式 $a_i$。

本文的事件级评估**不丢弃**未覆盖事件，而是按“排最后”计入，
避免高估 CPCA。覆盖率约 90%（OB 90.3%，TT 91.5%）。

### 4.8 算法

```text
Algorithm 1  CPCA 对一个窗口的归因
输入：窗口内已完成请求、正常参考、候选服务集合
输出：服务排序

1. 用式 (18) 选当前最异常的入口 operation o*
2. 初始化 delta[i]=0，total_D=0
3. 对 o* 的每条请求 tau：
4.     建 span 树；用式 (11) 算 local、pre、post
5.     D = 根 span 时长；total_D += D
6.     对 tau 里出现的每个服务 i：
7.         用式 (15)(16) 恢复 i 并重放，得到 D_cf
8.         delta[i] += max(D - D_cf, 0)
9. r[i] = delta[i] / total_D          # 式 (17)
10. 按 r 降序返回
```

---

## 5. 训练与推理

### 5.1 损失

预测阶段用三项损失：

$$
\mathcal{L}
= \mathcal{L}_{\text{sys}}
+ \alpha\,\mathcal{L}_{\text{op}}
+ \beta\,\mathcal{L}_{\text{rank}}.
\tag{19}
$$

**逐项解释。**

$$
\mathcal{L}_{\text{sys}} = \mathrm{BCE}(p_t,\ y_{t+1}),\qquad
\mathcal{L}_{\text{op}} = \mathrm{BCE}(p_{\text{op}},\ y_{t+1}),
\tag{20}
$$

- $\mathcal{L}_{\text{sys}}$：最终故障概率要拟合标签；
- $\mathcal{L}_{\text{op}}$：operation 聚合出来的概率也要拟合标签，
  保证 operation 头不是摆设；
- 权重 $\alpha=0.2$，让 operation 头有梯度但不主导。

$$
\mathcal{L}_{\text{rank}}
= -\frac{1}{|\mathcal{E}^+|}\sum_{t\in\mathcal{E}^+}
\log \mathrm{softmax}(a_t)_{r_{t+1}}.
\tag{21}
$$

- 只在“正例且有根因标签”的窗口算；
- 让注入服务的 $a_{r}$ 在 softmax 里尽量大；
- $\mathcal{E}^+$ 按事件等权（每个事件总权重为 1），避免长事件重复贡献梯度。

权重 $\beta=0.5$。

### 5.2 训练协议

- 五折 deployment-group 留出：test=fold $k$，validation=fold $(k+1)\bmod 5$，其余训练；
- Adam，学习率 $10^{-3}$，batch 32，30 epochs；
- 以验证集 AP 选 checkpoint；
- 标准化只在训练折拟合；
- 阶段二零训练，正常参考只取 case 前 10 分钟（因果 warm-up，不看注入时间）。

### 5.3 推理与校准

1. 阶段一前向：得到 $p$、$q_o$、$a_i$；
2. 阶段二：对同一窗口的 trace 算 $\bar r_i$；
3. 最终排序：按 $\bar r_i$；覆盖不足时回退 $a_i$；
4. 概率校准：在验证集上选 F1 最大阈值，并拟合温度缩放；测试集只评估一次。

**为什么要校准。** 模型 AP 很高，但 0.5 阈值下的 F1 会随种子大幅波动
（OB 从 0.658 到 0.927）。原因是概率整体偏移，不是排序坏了。
用验证集选阈值/温度后，OB 的 F1 收敛到 0.872–0.917。

---

## 6. 设计理由与消融

| 模块 | 为什么存在 | 去掉会怎样（实测） |
|---|---|---|
| 服务时序编码 | 每个服务自己的 5 分钟演化是预测主信号 | 未单独消融；去掉整条时序/图路径的 no-graph LTFP 在 fold 0 只有 AP 0.357 |
| 图传播 | 捕捉“上游等下游” | TT AP 0.853→0.824；OB 0.968→0.965（单入口，无额外关系） |
| Operation 风险头 | TT 标签是 per-operation | OB 0.968→0.940；TT 0.853→0.819 |
| Source/impact 学习头 | 可解释辅助分与消融对照 | 最终排序不用它：与 CPCA 冗余（融合 λ 单调趋近 CPCA-only） |
| CPCA local 归属 | 自身计算/IO 应归自己 | CPU/disk 无法定位 |
| CPCA 边等待归被调用方 | 返回路径等待是调用方在等被调用方 | delay 被算给调用方（exclusive 的失败模式） |
| 关键路径 slack | 判断服务是否在关键路径上 | 不在关键路径的服务会被排上来 |
| 零训练归因 | 只有 55/64 个事件，学习式 restoration 方差大于收益 | normal-restoration v0 的 Hit@1 只有 0.31/0.56 |

**关于“为什么最后不用融合”。** 我们试过
$\mathrm{score}_i=\log a_i+\lambda r_i$ 和 RRF 两种融合。
$\lambda$ 从 0 加到 8，指标单调上升，最后等于“完全用 CPCA”；
RRF 也没有超过 CPCA-only。这说明学习式 $a_i$ 在这份数据上没有互补信息。
所以最终方法直接输出 $r_i$，$a_i$ 只作为解释和消融。

---

## 7. 假设与边界

CPCA 的结构反事实依赖四个假设，论文要显式声明：

1. **单服务注入**：每个事件只有一个注入服务，根因标签就是干预目标。
   如果一次故障由多个服务共同导致，$r_i$ 会把责任分给多个服务，排序仍然可用，
   但“唯一根因”的语义不再成立。
2. **无排队反馈**：恢复 $i$ 后，其他服务的时延不变。
   真实系统里恢复根因会降低下游排队，所以 CPCA 的 $\Delta$ 是保守估计。
3. **请求混合稳定**：窗口内请求的 operation 和路径分布不发生剧变。
   如果故障导致流量结构变化（比如重试打向另一个服务），重放会失真。
4. **trace 覆盖充分**：入口 span 和父子关系可恢复。
   覆盖率不足时回退学习头。

**它不是完整的因果识别。** 我们估计的是“在观测 DAG 的结构方程下，
把 $i$ 恢复正常能移除多少入口时延”，不是任意干预下的反事实。
这个 estimand 与标签语义（注入服务）对齐，是方法成立的前提。

其他边界：

- OB 只有 1 个入口 operation，operation 级建模退化为系统级；
- loss 类故障的时延签名弱（实测 +4%~+9%），CPCA 在 loss 上不是最优；
- CPCA 覆盖约 90% 的正例窗口，未覆盖事件按失败计入。

---

## 8. 与现有方法的关系

- **CIRCA**：服务/指标级因果图 + 反事实干预。它需要学习或先验因果图，
  服务级指标又难区分“自己慢”和“继承慢”。CPCA 用观测到的请求 DAG，
  结构方程已知，不需要因果发现。
- **TraceRCA / MicroRank / TVDiag**：请求/图上的异常定位或传播排序。
  它们给“异常大小”排序，CPCA 给“可移除入口时延”排序。
- **LTFP / InstantOps / Eadro**：分钟级多模态/图模型。
  CPCA 不替代预测器，而是在同一窗口上提供结构归因。

---

## 9. 实现细节

| 项 | 取值 |
|---|---|
| 历史窗口 $T$ | 5 分钟 |
| 面板特征 $F$ | 19（metric 4 + observed 4 + trace 3 + log 3 + entry 5） |
| operation 特征 $F_o$ | 17 |
| 服务编码 hidden | 64，时间层 2 |
| 图注意力 | 2 层 GAT |
| operation GRU hidden | 32 |
| 聚合 | max（product 为消融） |
| 优化器 | Adam，lr $10^{-3}$，batch 32 |
| epochs | 30，验证 AP 选模 |
| 损失权重 | $\alpha=0.2,\ \beta=0.5$ |
| CPCA 每窗口最多请求 | 100 |
| 正常参考 | case 前 10 分钟中位数，按 (service, operation) 与边分别估计 |
| 事件级评估 | 同 event 窗口分数取均值后排序一次 |

---

## 10. 常见疑问

**Q1：窗口级和事件级有什么区别？用哪个？**
窗口级把每个 5 分钟窗口当样本（455/436 个）；事件级把同一故障事件的窗口聚合后排一次名
（55/64 个事件）。论文主表用事件级：决策单位是“一次故障”，事件之间近似独立，
置信区间可信；窗口级放附录，用来说明逐分钟稳定性，并与已有窗口级工作对比。
显著性检验要按事件聚类，不能把 455 个窗口当独立样本。

**Q2：$a_i$ 和 $r_i$ 是什么关系？**
$a_i$ 是学习出来的根因分，$r_i$ 是结构反事实算出来的责任分。
两者在实验里高度冗余，融合没有增量，所以最终排序用 $r_i$。

**Q3：为什么 OB 上图不重要，TT 上图重要？**
OB 只有 1 个入口 operation、调用链短，图提供不了额外关系；
TT 有 17 个入口 operation、27 个服务，跨服务关系更重要。
这是数据边界，不是方法不一致。

**Q4：为什么不做端到端？**
预测需要图，定位需要请求级结构；图传播会把根因证据扩散到症状。
我们的融合实验显示，端到端学出来的 $a_i$ 反而拖累 CPCA（λ 越大越好）。
所以采用“学习式预测 + 结构式归因”的两阶段。

**Q5：CPCA 需要训练吗？**
不需要。它只依赖 span 树和正常参考，是确定性的。
这也是它在这份小数据（55/64 个事件）上比学习式 restoration 稳的原因。
