# 3. 方法

> 注：公式编号保留空号（8–9、21、24 随学习式根因头移除），
> 以便与前一版逐条对照；投稿版会重新连续编号。

本章给出 OCSID（Operation-Conditioned Structural Counterfactual Attribution）的完整形式化描述。
第 3.1 节定义问题与符号；第 3.2 节给出框架总览；第 3.3 节描述预测模块；
第 3.4 节给出结构反事实归因（CPCA）；第 3.5 节定义训练目标；
第 3.6 节说明推理与校准；第 3.7 节讨论设计依据与假设；第 3.8 节给出实现细节与复杂度。

---

## 3.1 问题定义

### 3.1.1 系统与观测

考虑由 $N$ 个微服务构成的系统。服务之间通过 RPC 调用，调用关系由有向图表示。
用户请求从某个**入口 operation** 进入系统，经过若干服务后返回。
设 $M$ 为入口 operation 的集合。

在分钟 $t$，模型可观测过去 $T=5$ 个完整分钟的三类数据：

- **服务级面板特征** $\mathbf{X}_t\in\mathbb{R}^{T\times N\times F}$，
  其中 $F=19$，包含资源指标、观测掩码、trace 统计、日志统计与入口统计；
- **服务调用图** $\mathbf{A}_t\in\mathbb{R}^{T\times N\times N}$，
  $\mathbf{A}_t[u,v]>0$ 表示服务 $u$ 在分钟 $t$ 调用了服务 $v$；
- **入口 operation 特征** $\mathbf{O}_t\in\mathbb{R}^{T\times M\times F_o}$，
  其中 $F_o=17$，包含请求数、p50/p95/p99、错误率、相对正常基线的比值与退化分数；
- **operation 存在掩码** $\mathbf{m}_t\in\{0,1\}^{T\times M}$，
  当某 operation 在该分钟的请求数不足时置 0。

记窗口内的已完成请求集合为 $\mathcal{T}_t$。每条请求 $\tau\in\mathcal{T}_t$
对应一棵 span 树，节点为 span，边为父子调用关系。

### 3.1.2 任务

给定 $\mathbf{X}_t,\mathbf{A}_t,\mathbf{O}_t,\mathbf{m}_t$ 与 $\mathcal{T}_t$，输出：

1. 下一分钟系统故障概率 $p_t\in[0,1]$；
2. 根因服务排序 $\hat{\mathbf{R}}_t\in\mathbb{R}^{N}$。

监督信号为 $y_{t+1}\in\{0,1\}$（下一分钟是否处于用户可感知故障状态）与
$r_{t+1}\in\{1,\dots,N\}$（注入服务）。每个正例窗口带有事件编号 $e$，
用于事件级评估（见 3.8 节）。

### 3.1.3 符号

| 符号 | 定义 |
|---|---|
| $N,M,T$ | 服务数、入口 operation 数、历史窗口长度 |
| $F,F_o$ | 面板特征维数、operation 特征维数 |
| $\mathbf{X},\mathbf{A},\mathbf{O},\mathbf{m}$ | 面板、图、operation、掩码 |
| $\tau$ | 一条请求 |
| $s$ | 一个 span |
| $v(s)$ | span $s$ 所属服务 |
| $\mathrm{op}(s)$ | span $s$ 的 operation 名 |
| $C(s)$ | $s$ 的直接子 span 集合 |
| $p(c)$ | span $c$ 的父 span |
| $d_s$ | span $s$ 的时长 |
| $\mathrm{self}_s$ | $s$ 扣除直接子区间并集后的自身时间 |
| $\mathrm{local}_s$ | $\mathrm{self}_s$ 中用于本地计算/IO 的部分 |
| $\mathrm{pre}_s(c)$ | 父 span $s$ 调用子 $c$ 前的等待空档 |
| $\mathrm{post}_s(c)$ | 子 $c$ 返回后父 span 继续前的等待空档 |
| $D(\tau)$ | 请求 $\tau$ 的入口时延 |
| $\mathcal{W}_0$ | 因果预热窗口（case 前 10 分钟） |
| $r_i$ | CPCA 结构责任分 |

---

## 3.2 框架总览

OCSID 由两个阶段构成（图 1）：

- **阶段一（预测，可训练）**：由服务时序编码器、图传播、系统风险头与 operation
  风险头组成，输出下一分钟故障概率 $p_t$；
- **阶段二（归因，零训练）**：CPCA 在请求级 span 树上执行结构反事实重放，
  输出每个服务的可移除入口时延比例 $r_i$，并据此排序。

两个阶段共享同一时间窗口与同一批 trace，但优化目标不同：
前者是监督学习问题，后者是结构反事实估计问题。第 3.7.3 节说明该解耦设计的依据。

```text
阶段一：预测（可训练）
  X, A ──► 服务时序编码 ──► 图传播 ──► p_sys ─┐
  O, m ──► operation 编码 ──► q_o ──► p_op ──┴──► p

阶段二：归因（零训练）
  traces ──► span 树分解 ──► 反事实重放 ──► r_i ──► 根因排序
```

---

## 3.3 预测模块

### 3.3.1 服务时序编码

对每个服务 $i$，将其 $T$ 分钟的面板特征按模态分组（metric、log、trace），
分别经因果时序卷积与时间注意力编码，再融合为局部表示：

$$
\mathbf{h}^{\text{local}}_i
= \mathrm{Fuse}\Big(
\big\{\mathrm{TCN}_m(\mathbf{X}_{i,t-T+1:t}^{(m)})\big\}_{m\in\mathcal{M}}
\Big),
\tag{1}
$$

其中 $\mathcal{M}=\{\text{metric},\text{log},\text{trace}\}$，
$\mathrm{TCN}_m$ 为因果膨胀卷积与注意力读出的组合，
$\mathrm{Fuse}$ 为模态门控加权与拼接。实现中隐层维数 $H=64$，时序层数 2。

按模态分组编码的原因是三类数据的量纲与噪声结构差异显著：
资源指标为有界利用率，日志为计数，trace 为毫秒级时延。
分组编码可避免弱模态被强模态淹没。

### 3.3.2 图传播

以调用图 $\mathbf{A}_t$ 为邻接，进行 $L=2$ 层图注意力传播：

$$
\mathbf{h}^{(\ell+1)}_i
= \mathrm{LN}\Big(
\mathbf{h}^{(\ell)}_i
+ \mathrm{Dropout}\big(\mathrm{GAT}(\mathbf{h}^{(\ell)},\mathbf{A}_t)\big)
\Big),\quad \ell=1,2,
\tag{2}
$$

其中 $\mathbf{h}^{(0)}_i=\mathbf{h}^{\text{local}}_i$，
$\mathrm{GAT}$ 为边权感知的图注意力层。记 $\mathbf{h}^{\text{graph}}_i=\mathbf{h}^{(2)}_i$。

图传播使每个服务获得调用方与被调用方的上下文，用于刻画跨服务影响。
3.7.4 节的消融表明该模块在异构入口 operation 较多的数据集上是必要的。

### 3.3.3 系统风险头

对图传播后的服务表示做均值与最大池化，得到系统表示 $\mathbf{g}$，
再经二分类头输出系统风险：

$$
\mathbf{g}
= \mathrm{MLP}\Big(\Big[\tfrac{1}{N}\sum_{i=1}^{N}\mathbf{h}^{\text{graph}}_i,\
\max_{i}\mathbf{h}^{\text{graph}}_i\Big]\Big),
\qquad
\mathbf{z}^{\text{sys}}=\mathrm{MLP}(\mathbf{g})\in\mathbb{R}^{2},
\tag{3}
$$

$$
p_{\text{sys}}=\mathrm{softmax}(\mathbf{z}^{\text{sys}})_1 .
\tag{4}
$$

均值池化刻画系统整体水位，最大池化刻画最严重的单个服务；
二者拼接可同时覆盖“整体退化”与“单点异常”两类情形。

### 3.3.4 Operation 风险头

对每个入口 operation $o$，用共享 GRU 编码其 $T$ 分钟特征序列，
并输出下一分钟退化概率：

$$
\mathbf{u}_o=\mathrm{GRU}\big(\mathbf{O}_{o,t-T+1:t}\big),
\qquad
q_o=\sigma\big(\mathbf{w}_o^{\top}\mathrm{LN}(\mathbf{u}_o)\big).
\tag{5}
$$

系统级 operation 风险由下式聚合：

$$
p_{\text{op}}=
\begin{cases}
\max_{o:\,m_o=1} q_o, & \text{(max)}\\[2pt]
1-\prod_{o:\,m_o=1}(1-q_o), & \text{(product)}
\end{cases}
\tag{6}
$$

默认采用 max 聚合：系统的故障定义是“任一入口 operation 持续退化”，
因此最危险的入口最能代表系统状态。product 聚合假设 operation 相互独立，
在多个 operation 受同一根因影响的场景下会系统性高估，故仅作为消融。

### 3.3.5 融合

系统风险与 operation 风险在 logit 空间融合：

$$
p=\sigma\Big(
w_1\,\mathrm{logit}(p_{\text{sys}})
+w_2\,\mathrm{logit}(p_{\text{op}})
\Big),
\tag{7}
$$

其中 $w_1,w_2$ 为可学习标量。logit 空间的线性组合等价于证据相加，
避免概率空间在 0 与 1 附近的饱和。

推理阶段，预测器不包含学习式根因头。根因排序完全由 3.4 节的 CPCA 给出；
预测器只输出 $p_t$ 与 $q_o$。这样预测与归因的边界是唯一的：
任何根因分数都来自结构反事实，而不是另一个学习头。

训练阶段可选地使用一个根因辅助头做正则：它用根因标签计算排序损失，
但不参与推理，也不进入归因结果。推理模型以 `aux_root=False` 构建，
没有该头，也不返回任何根因分数。因此“归因头”在推理阶段只有 CPCA；
辅助头只是训练期的正则化器。

---

## 3.4 结构反事实归因

本节定义 CPCA。其核心思想是：将请求的 span 树视为结构因果模型，
把服务 $i$ 的局部成本与进出它的边等待恢复到正常参考，重放该请求，
以入口时延的下降量作为 $i$ 对本次故障的贡献。

### 3.4.1 Span 树分解

对请求 $\tau$ 中的每个 span $s$，在区间并集语义下，

$$
d_s=\mathrm{self}_s
+\Big|\bigcup_{c\in C(s)}[\mathrm{start}_c,\mathrm{end}_c]\Big|.
\tag{10}
$$

$\mathrm{self}_s$ 进一步分解为三类互不重叠的时间：

$$
\mathrm{self}_s
=\mathrm{local}_s
+\sum_{c\in C(s)}\big(\mathrm{pre}_s(c)+\mathrm{post}_s(c)\big).
\tag{11}
$$

其中 $\mathrm{local}_s$ 为本地计算/IO 时间；
$\mathrm{pre}_s(c)$ 为子调用开始前的等待空档（请求路径网络与对端排队）；
$\mathrm{post}_s(c)$ 为子调用结束后的等待空档（返回路径）。

该分解将“服务自身变慢”与“服务在等待下游”区分开，
这是后续归因规则的基础。

### 3.4.2 成本归属

对服务 $i$，定义其在请求 $\tau$ 上的成本：

$$
c_i(\tau)
=\sum_{s:\,v(s)=i}\mathrm{local}_s
+\sum_{c:\,v(c)=i}\big(\mathrm{pre}_{p(c)}(c)+\mathrm{post}_{p(c)}(c)\big).
\tag{12}
$$

即 **本地时间归自身，进出该服务的边等待归被调用方**。
由式 (10)(11) 可得 $\sum_i c_i(\tau)=\mathrm{self}_{\text{root}}(\tau)$，
成本分解不重不漏。

将边等待归给被调用方而非调用方，是本文与既有 exclusive-latency 方法的关键差异。
其依据来自对原始 trace 的直接观测：在 RE2 的 delay 注入中，
`post-gap` 由 0.7 ms 增至 200.8 ms，而 `pre-gap` 基本不变；
在 CPU/disk 注入中，仅服务自身的 `local` 与时长上升。
若将返回路径等待留给调用方，delay 类故障会被系统性地错误归因。

### 3.4.3 入口时延与松弛量

请求的入口时延定义为其根 span 的时长：

$$
D(\tau)=d_{\mathrm{root}(\tau)}.
\tag{13}
$$

对服务 $i$，定义松弛量：

$$
\mathrm{slack}_i(\tau)=D(\tau)-D_{-i}(\tau),
\tag{14}
$$

其中 $D_{-i}(\tau)$ 为将 $i$ 的成本置零后重放得到的入口时延。
$\mathrm{slack}_i$ 表示“$i$ 最多能移除多少入口时延”，
用于判断 $i$ 是否位于关键路径：

- $\mathrm{slack}_i=0$：$i$ 不在关键路径上，恢复它不会缩短入口时延；
- $\mathrm{slack}_i>0$：$i$ 位于关键路径上。

需要区分 $\mathrm{slack}$ 与反事实下降量 $\Delta$：
一个服务可以 $\mathrm{slack}>0$ 但 $\Delta=0$，即它位于关键路径上，
但自身成本已处于正常水平。因此 $\mathrm{slack}$ 仅作为结构诊断，
最终排序使用 3.4.5 节的 $\Delta$。

### 3.4.4 反事实重放

设因果预热窗口 $\mathcal{W}_0$ 为每个 case 的前 10 分钟。对服务 $i$ 与
operation $o$，正常参考取 $\mathcal{W}_0$ 内同 operation 局部时间的中位数：

$$
\hat{\mathrm{local}}_{i,o}
=\operatorname*{median}_{s:\,v(s)=i,\,\mathrm{op}(s)=o,\,\tau\in\mathcal{W}_0}
\mathrm{local}_s .
\tag{15}
$$

边等待的正常参考 $\hat{\mathrm{gap}}_{e}$ 以同样方式按
$(\text{父服务},\text{父 operation},\text{子服务},\text{子 operation})$ 估计。

我们也测试了因果滚动参考：对每个决策分钟 $t$，取最近 $W$ 个
$\text{failure}=0$ 的分钟，向固定预热先验收缩，
$\hat{x}_t=\frac{n}{n+k}\mathrm{med}_{\text{rolling}}+\frac{k}{n+k}\mathrm{med}_{\text{fixed}}$。
在 RE2-TT 上，滚动参考的软加权为 $0.938/0.961$，低于固定参考的 $0.953/0.974$；
原因是事件前的“健康分钟”已包含早期退化，滚动参考会把异常吸收进基线。
因此默认使用固定预热参考，滚动版本作为敏感性分析。

对服务 $i$ 的反事实是将 $i$ 的局部时间与进出它的边等待替换为正常参考：

$$
\widetilde{\mathrm{local}}_s=
\begin{cases}
\hat{\mathrm{local}}_{v(s),\mathrm{op}(s)}, & v(s)=i,\\
\mathrm{local}_s, & \text{otherwise},
\end{cases}
\qquad
\widetilde{\mathrm{gap}}_s(c)=
\begin{cases}
\hat{\mathrm{gap}}_{e(s,c)}, & v(c)=i,\\
\mathrm{gap}_s(c), & \text{otherwise}.
\end{cases}
\tag{16}
$$

自底向上重放 span 树：

$$
\tilde d_s=\widetilde{\mathrm{local}}_s
+\sum_{c\in C(s)}\big(\widetilde{\mathrm{pre}}_s(c)+\tilde d_c+\widetilde{\mathrm{post}}_s(c)\big).
\tag{17}
$$

重放的精确定理性依赖于“同一父节点下的子调用不重叠”。
在 RE2-TT 的抽样审计中，18,099 个多子节点均无区间重叠，
因此式 (17) 的串行重放与区间并集语义等价。

### 3.4.5 责任分

对请求 $\tau$ 与服务 $i$，定义反事实可移除时延与责任分：

$$
\Delta_i(\tau)=D(\tau)-\tilde D_i(\tau),
\qquad
r_i=\frac{\sum_{\tau\in\mathcal{T}_{o^\star}}\Delta_i(\tau)}
{\sum_{\tau\in\mathcal{T}_{o^\star}}D(\tau)},
\tag{18}
$$

其中 $\mathcal{T}_{o^\star}$ 为所选入口 operation 的请求集合（见 3.4.6）。
$r_i\in[0,1]$ 表示当前窗口入口总时延中，可通过恢复 $i$ 移除的比例。
采用比例而非绝对毫秒，使不同正常时延量级的服务可比较。

### 3.4.6 Operation 条件化归因

设 $n_o(t)$ 为 operation $o$ 在窗口 $t$ 的请求数，$\mathrm{p95}_o(t)$ 为其 p95 时延，
$\mathrm{p95}_o^{\text{normal}}$ 为其正常参考。选择最异常的入口 operation：

$$
o^\star_t=
\arg\max_{o:\,n_o(t)\ge n_{\min}}
\frac{\mathrm{p95}_o(t)}{\mathrm{p95}_o^{\text{normal}}},
\tag{19}
$$

其中 $n_{\min}=20$。硬选择版本仅在该 operation 的请求上计算 $r_i$。

为降低对硬选择的敏感性，本文默认采用软加权聚合：

$$
w_o=
\frac{\exp(\mathrm{ratio}_o/\tau)}
{\sum_{o'}\exp(\mathrm{ratio}_{o'}/\tau)},
\qquad
r_i=\sum_o w_o\,r_i^{(o)},
\tag{19a}
$$

其中 $\mathrm{ratio}_o=\mathrm{p95}_o(t)/\mathrm{p95}_o^{\mathrm{normal}}$，
$\tau=1$ 为温度。硬选择是 $\tau\to0$ 的极限；软加权让根因不在最风险 operation
上时仍有分数，并减少对 $n_{\min}$ 的依赖。当前实现用入口 p95 比值作为 $q_o$ 的代理，
将训练好的 $q_o$ 作为权重是下一步工作。

Operation 条件化的动机来自标签定义：在 RE2-TT 中，故障标签按
“同一入口 operation 的 p95 连续两分钟超过其注入前基线”定义。
若将多个 operation 的请求混合，根因所在 operation 的信号会被其他 operation 稀释。
在 RE2-OB 中仅存在一个入口 operation，该步骤退化为系统级归因。

在 RE2-TT 的事件级评估中，硬选择为 $0.952/0.974$（63 个事件），
软加权为 $0.953/0.974$（64 个事件）：软加权多覆盖一个事件，H@1 略升，MRR 持平。
在 RE2-OB 上两者一致（单入口 operation）。

我们进一步用训练好的 $q_o$ 替换 p95 比值作为权重。RE2-TT 上仍为 $0.953/0.974$
（与软加权相同），RE2-OB 上退化为硬选择（单入口 $q_o$ 无法区分 operation）。
因此默认实现使用 p95 比值，$q_o$ 保留在预测侧，不作为 CPCA 的归因权重。

### 3.4.7 覆盖率与回退

定义服务 $i$ 的覆盖率：

$$
c_i=\frac{\sum_{\tau\in\mathcal{T}_{o^\star}}\mathbb{1}[i\in\tau]\,D(\tau)}
{\sum_{\tau\in\mathcal{T}_{o^\star}}D(\tau)},
\tag{20}
$$

其中 $\mathbb{1}[i\in\tau]$ 表示请求 $\tau$ 的路径经过 $i$。
若 $c_i$ 低于阈值 $\kappa$，CPCA 无法可靠估计 $r_i$；此时该服务不参与排序，
记为最后，不再有学习式回退。在本文的实验中，为保证评估的保守性，
未覆盖的根因事件按“排名最后”计入，而非丢弃；主结果报告的是这一保守口径。

### 3.4.8 算法与复杂度

```text
Algorithm 1  CPCA attribution for one window
Input : completed requests T_t, normal reference, services S
Output: ranking over S

1. select entry operation o* by Eq. (19)
2. delta[i] = 0 for all i; total_D = 0
3. for each request tau in T_t with operation o*:
4.     build span tree; compute local, pre, post by Eq. (11)
5.     D = duration(root); total_D += D
6.     for each service i on tau:
7.         replay with i restored by Eq. (16)(17) -> D_cf
8.         delta[i] += max(D - D_cf, 0)
9. r[i] = delta[i] / total_D                  # Eq. (18)
10. return S sorted by r descending
```

**复杂度。** 预测阶段单样本为
$O\big(TN^2+TN\!FH+TM\!F_oH\big)$，主要来自图注意力与 operation GRU。
CPCA 阶段为 $O(K\cdot S\cdot L)$，其中 $K$ 为窗口内采样请求数（上限 100），
$S$ 为单条请求经过的服务数，$L$ 为 span 数；由于 $K,S,L$ 均为小常数，
归因开销远低于一次前向传播，且不需要训练。

---

## 3.5 训练

### 3.5.1 目标函数

预测阶段优化两项损失：

$$
\mathcal{L}
=\mathcal{L}_{\text{sys}}
+\alpha\,\mathcal{L}_{\text{op}},
\tag{22}
$$

其中

$$
\mathcal{L}_{\text{sys}}=\mathrm{BCE}(p_t,y_{t+1}),
\qquad
\mathcal{L}_{\text{op}}=\mathrm{BCE}(p_{\text{op}},y_{t+1}).
\tag{23}
$$

$\mathcal{L}_{\text{sys}}$ 监督最终故障概率；$\mathcal{L}_{\text{op}}$ 监督
operation 聚合风险，确保 operation 头获得有效梯度。实现中 $\alpha=0.2$。

训练阶段可选地加入根因辅助损失
$\mathcal{L}_{\text{aux}}=-\log\mathrm{softmax}(a_t)_{r_{t+1}}$，
只用于正则化共享编码器；推理模型不包含该头。消融显示，完全去掉根因监督会使
RE2-OB 的 AP 从 $0.9675$ 降到 $0.8997$，RE2-TT 从 $0.8530$ 降到 $0.8302$，
因此默认保留训练期辅助损失，但明确它不是归因模块。

### 3.5.2 优化协议

- 五折 deployment-group 留出：第 $k$ 折为测试集，第 $(k+1)\bmod 5$ 折为验证集，其余为训练集；
- 优化器 Adam，学习率 $10^{-3}$，batch size 32，训练 30 个 epoch；
- 以验证集平均精度（AP）选择 checkpoint；
- 所有标准化参数仅在训练折上拟合；
- 阶段二不使用任何训练，正常参考仅取自每个 case 的前 10 分钟（因果预热）。

---

## 3.6 推理与校准

推理流程如下：

1. 阶段一前向计算 $p_t$ 与 $q_o$；
2. 阶段二对同一窗口的 trace 计算 $\bar r_i$，按覆盖率规则得到最终排序；
3. 概率校准：在验证集上选择使 F1 最大的阈值，并拟合温度缩放参数，
   测试集仅评估一次。

温度缩放定义为 $p'=\sigma\big(\mathrm{logit}(p)/T\big)$，
其中温度 $T$ 在验证集上以负对数似然最小化选取。
校准不改变排序，仅修正概率尺度，从而缓解高 AP 下 F1 随阈值剧烈波动的问题。

---

## 3.7 设计讨论

### 3.7.1 为何使用观测 DAG 而非学习因果图

CIRCA 类方法在服务/指标级学习或先验一个因果图，再在图上做干预。
本文不学习因果图，原因有二：

1. **结构方程已知。** 请求的 span 树满足区间守恒（式 (10)），
   父子调用的时间关系是观测到的事实，不需要从数据中推断；
2. **粒度匹配。** 服务级指标无法区分“自身变慢”与“等待下游变慢”，
   而请求级 span 分解可以精确区分二者。

因此 CPCA 的估计量是“在观测 DAG 的结构方程下，恢复服务 $i$ 能移除多少入口时延”，
而不是任意干预下的反事实。该估计量与标签语义（注入服务）对齐。

### 3.7.2 为何边等待归被调用方

若将返回路径等待留给调用方（exclusive-latency 的常见做法），
delay 类故障会被错误地归因给调用方。3.4.2 节的 trace 观测表明，
delay 注入主要表现为 `post-gap` 上升；
将这一项归给被调用方，才能使其与 CPU/disk 的 `local` 上升落在同一服务分数上。
该规则由结构守恒与数据观测共同决定，而非超参数。

### 3.7.3 为何采用两阶段而非端到端

预测与归因的表示需求不同：预测需要图传播以捕获跨服务影响，
而归因需要请求级结构以避免根因证据被邻居稀释。
将二者耦合会导致负迁移：我们曾训练 source/impact 学习式根因头，并与 CPCA 融合；
融合权重越大指标越接近纯 CPCA，说明学习头没有提供互补信息。
因此最终移除了学习式根因头，框架只保留“可训练预测器 + 零训练 CPCA”，
预测与归因的边界是唯一的。

### 3.7.4 假设与适用范围

CPCA 的结构反事实依赖以下假设：

1. **单一服务干预**：每个事件由一个注入服务导致，根因标签即干预目标；
2. **无排队反馈**：恢复 $i$ 后其他服务时延不变，重放保持观测相对时刻；
3. **请求混合稳定**：窗口内 operation 与路径分布不发生剧变；
4. **trace 覆盖充分**：入口 span 与父子关系可恢复；覆盖率不足时该服务排最后。

在上述假设下，CPCA 给出的是保守估计：真实系统中恢复根因会进一步降低下游排队，
因此实际可移除时延通常不小于 $\Delta_i$。

方法的适用边界还包括：

- 当系统仅有一个入口 operation 时，operation 条件化退化为系统级，
  图传播的增益也相应减弱（RE2-OB）；
- 当故障主要表现为错误率而非时延（如部分 loss 场景）时，
  时延分解的信号较弱，CPCA 的相对优势下降。

---

## 3.8 评估协议与实现细节

### 3.8.1 评估口径

本文同时报告窗口级与事件级指标，并以事件级为主：

- **窗口级**：每个正例的 5 分钟滑窗为一个样本，逐窗口排序后对窗口平均；
- **事件级**：按事件编号聚合同一故障事件的窗口分数（取均值），每个事件排序一次，
  再对事件平均。

事件级的样本单位为 $55$（RE2-OB）与 $64$（RE2-TT），近似独立；
窗口级的 $455$ 与 $436$ 个窗口在同一事件内高度相关，存在伪重复。
因此主表使用事件级，窗口级作为附录，用于刻画逐分钟稳定性并与既有窗口级工作对比。
显著性检验按事件聚类。

预测指标为 AP 与 F1（验证集选阈值）；定位指标为 Hit@1、Hit@3 与 MRR。

### 3.8.2 多 horizon 协议

对 `state@h`，把标签从 $t+1$ 移到 $t+h$，输入窗口与模型不变；
CPCA 仍在决策分钟 $t$ 的请求上计算。对 `first-visible@h`，
目标是 $(t,t+h]$ 内首次故障，可用离散时间 hazard
$p_h=1-\prod_{k=1}^{h}(1-\lambda_k)$ 保证单调；本文尚未训练该任务。

在进入 3/5 分钟前，先做 Gate 0：检查决策时刻根因是否已有信号。
RE2-TT 的事件级信号为 $state@3$ 93.5%、$state@5$ 85.7%；
RE2-OB 为 96.4%、85.5%。因此多 horizon 可做，但应以事件级为主。
当前实现为每个 horizon 单独训练一个预测器，尚未使用共享编码器 +
horizon embedding；CPCA 也尚未使用 trailing window。

### 3.8.3 实现细节

| 项 | 取值 |
|---|---|
| 历史窗口 $T$ | 5 分钟 |
| 面板特征 $F$ | 19（metric 4 + observed 4 + trace 3 + log 3 + entry 5） |
| operation 特征 $F_o$ | 17 |
| 服务编码隐层 | 64，时序层数 2 |
| 图传播 | 2 层 GAT |
| operation GRU 隐层 | 32 |
| 聚合方式 | max（product 为消融） |
| 优化器 | Adam，学习率 $10^{-3}$，batch 32 |
| 训练轮数 | 30，验证 AP 选模 |
| 损失权重 | $\alpha=0.2$，$\beta=0.5$ |
| CPCA 每窗口请求上限 | 100 |
| 正常参考 | case 前 10 分钟中位数，按 (service, operation) 与边分别估计 |
| 事件级聚合 | 同事件窗口分数取均值后排序一次 |
