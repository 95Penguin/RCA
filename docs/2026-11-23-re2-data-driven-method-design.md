# RE2 数据审计与数据驱动的方法设计

日期：2026-11-23

## 1. 先说明这份数据实际是什么

RE2-OB 和 RE2-TT 都不是一条连续生产时间序列，而是 90 个独立故障注入 case。每个数据集由 5 个注入服务、6 类故障和 3 次重复组成：

`5 services × 6 faults × 3 repetitions = 90 cases`

故障类型均为 CPU、delay、disk、loss、memory 和 socket。每个 case 约 24 分钟，前段为正常基线，后段包含故障注入。监督标签不是“发生过注入”，而是注入后是否出现可观测的入口业务延迟影响。

| 数据集 | 原始压缩包 | 图节点 | 有业务影响的 case | 正样本事件 | 正样本分钟 / 有效标签分钟 |
|---|---:|---:|---:|---:|---:|
| RE2-OB | 1.11 GB | 11 | 51 / 90 | 55 | 455 / 2054（22.2%） |
| RE2-TT | 2.61 GB | 27 | 53 / 90 | 64 | 436 / 2070（21.1%） |

不同故障类型并不必然产生业务影响：

| 数据集 | CPU | Delay | Disk | Loss | Memory | Socket |
|---|---:|---:|---:|---:|---:|---:|
| OB 受影响 case / 15 | 5 | 13 | 5 | 11 | 11 | 6 |
| TT 受影响 case / 15 | 7 | 10 | 6 | 10 | 11 | 9 |

这意味着任务同时包含两个问题：异常是否真的传到业务入口，以及传到入口后源头在哪里。把所有注入一律当故障会改变任务定义；当前标签避免了这个问题。

## 2. Metric、Log、Trace 原始数据里有什么

### 2.1 Metric

每个 case 同时包含：

- `metrics.csv`：较完整的容器/Kubernetes 指标。OB 有 675 个去重指标列，TT 有 1828 个。
- `simple_metrics.csv`：预聚合资源、workload、error 和 latency 指标。OB 有 86 个去重列，TT 有 383 个。

当前 panel 实际只使用每个图节点的四个一分钟均值：

- `metric_cpu`
- `metric_mem`
- `metric_diskio`
- `metric_socket`

同时保存四个 observed mask。OB 的 CPU、内存和 socket 覆盖率约 99.9%，disk I/O 只有 59.7%。TT 四项覆盖率均接近 100%。

OB 的 11 个指标资源都能映射到图节点。TT 原始 simple metrics 有 68 个资源实体，当前只保留 27 个 Trace 可见应用服务；另外 41 个实体被丢弃，其中包括大量 Mongo/MySQL、若干业务服务和 UI，例如 `ts-auth-mongo`、`ts-order-mongo`、`ts-food-mysql`、`ts-voucher-mysql`、`ts-route-plan-service` 等。

因此，TT 不是“指标不足”，而是当前节点定义主动舍弃了大量资源上下文。

### 2.2 Log

原始 `logs.csv` 包含：

- 纳秒级 timestamp
- container/service name
- 原始 message
- level
- request path/error（部分数据可用）
- OB 还直接附带 `cluster_id` 和 `log_template`

每个 case 另有 `cluster_info.json`，提供模板化日志。跨 case 去重后，OB 至少有 145 种模板，TT 至少有 955 种模板。

当前 panel 只使用：

- `log_count`
- `log_error_count`
- `log_observed`

日志 error 极稀少：OB 中 error/log 约为 0.0262%，TT 约为 0.0048%。因此当前所谓“Log 模态”主要是日志量，而不是错误语义。模板出现、模板消失、模板转移和消息参数异常都没有进入模型。

覆盖方面，OB 每分钟中位数有 9/11 个服务产生日志，TT 为 20/27；存在明显的结构性缺失，零值不能简单解释为“正常”。

### 2.3 Trace

原始 `traces.csv` 包含：

- `traceID`
- `spanID`、`parentSpanID`
- `serviceName`
- `methodName`、`operationName`
- start time、duration
- statusCode

这些字段足以恢复请求级有向 span DAG、入口 operation、每段 inclusive/exclusive latency 和跨服务调用路径。

原始聚合文件中，OB 固定有 17 个 service-operation 序列；TT 跨 case 有 177 个 operation，单个 case 通常出现约 157 个。当前 panel 将它们压缩为每服务每分钟三个量：

- `trace_span_count`
- `trace_error_count`
- `trace_duration_mean_ms`

只为入口额外保留 `entry_span_count`、`entry_p95_ms` 和滚动退化分数。operation 身份、p95/p99、请求路径、父子时延分解和同一 operation 的正常基线都没有保留。

TT 的根 HTTP span `statusCode` 为空，整个 panel 的 `trace_error_count` 实际为 0。OB 的非零 Trace status 也只占所有 span 的约 0.0251%。在这份数据上，Trace 的有效信号几乎完全来自调用量和时延，而不是错误码。

当前图虽然由 parent-child span 统计得到，但最终使用对称邻接，而且整分钟内所有请求混在一起。它回答“哪些服务通常互相调用”，不能回答“哪个受影响 operation 的哪条路径在变慢”。

## 3. 已有实验已经证明了什么

### 3.1 模态消融

正式五折 LTFP 模态实验如下：

| 数据集 | 模态 | AP | F1 | RCA Hit@1 | RCA MRR |
|---|---|---:|---:|---:|---:|
| OB | Metric | 0.694 | 0.585 | 0.611 | 0.753 |
| OB | Log | 0.830 | 0.656 | 0.192 | 0.496 |
| OB | Trace | 0.898 | 0.780 | 0.818 | 0.876 |
| OB | Metric + Trace | **0.928** | 0.812 | **0.851** | **0.897** |
| OB | All | 0.919 | **0.837** | 0.807 | 0.878 |
| TT | Metric | 0.683 | 0.579 | 0.741 | 0.826 |
| TT | Log | 0.493 | 0.460 | 0.231 | 0.466 |
| TT | Trace | 0.800 | 0.762 | 0.748 | 0.823 |
| TT | Metric + Trace | 0.837 | 0.736 | 0.881 | 0.920 |
| TT | All | **0.846** | **0.771** | **0.894** | **0.932** |

数据结论：

1. Trace 是两个数据集预测业务影响的主模态。
2. RCA 最稳定的底座是 Metric + Trace。
3. 当前 Log 对 OB 的预测有用、对 RCA 明显有害；TT 的 Log 单模态接近噪声，只在与其他模态联合时提供很小增益。
4. 因此不应继续使用无条件的三模态等价融合。Log 更适合作为可选辅助证据。

### 3.2 根因与症状的时序关系

使用 case 内注入前正常基线、robust deviation > 3 且持续两分钟的审计结果：

| 数据集 | 根因出现一般异常 | 根因早于或同时于邻居 | 根因相对业务影响的中位提前量 | 仅凭首次异常 Hit@1 / MRR |
|---|---:|---:|---:|---:|
| OB | 49 / 51 | 79.6% | 1 分钟 | 0.353 / 0.560 |
| TT | 53 / 53 | 86.8% | 1 分钟 | 0.245 / 0.486 |

但根因相对邻居的中位领先量为 **0 分钟**：大多数根因和症状在一分钟粒度下同时越过阈值。故障类型对应的资源指标也不总出现：OB 只有 23/51、TT 只有 27/53 个受影响 case 出现 fault-aligned root signal；memory 和 loss 尤其差。

这解释了为什么“改成有向图”或“只按谁先异常排序”很难稳定提升：方向信息存在，但一分钟聚合后多数节点同时变化，而且全局调用图混合了未受影响请求。

### 3.3 所谓下一分钟预测主要是什么

正式 OOF 样本中：

- OB 的 455 个正分钟里，400 个是持续故障，严格的 normal→failure onset 只有 10 个，另有 45 个来自 unknown→failure 标签边界。
- TT 的 436 个正分钟里，372 个是持续故障，严格 onset 只有 20 个，另有 44 个 unknown→failure。

LTFP 对 continuing 的召回约为 OB 83.5%、TT 84.7%，对严格 onset 只有 OB 20%、TT 45%。所以高总体指标主要证明“故障状态延续预测”，不能单独证明强提前预警能力。深度学习 onset 分支只有 10/20 个 OOF 正例，继续加复杂结构很容易过拟合。

## 4. 数据真正支持的方法方向

建议的方法不是再做一次全图 GNN，而是：

## Operation-Conditioned Source–Impact Decomposition（OC-SID）

核心思想：先确定哪个入口 operation 正在形成业务影响，只在该 operation 对应的请求路径上分解“源头局部异常”和“沿路径产生的影响”。

### 4.1 Operation 级业务影响跟踪

保留入口 operation 身份，为每个 operation 构造只使用历史的滚动正常参考：

- request count
- p50 / p95 / p99
- timeout 或缺失完成率
- p95 斜率和加速度
- 相对同一 operation 基线的 deviation

预测头分成两个可解释部分：

- persistence head：当前已经退化时预测下一分钟是否持续；这是数据量充足的主体任务。
- onset hazard：当前尚未退化时，根据 operation 尾延迟趋势和路径异常估计首次退化风险。由于 onset 极少，应使用低容量 change-point/hazard 模型，而不是再训练一个深网络。

最终概率由当前状态路由，但两类结果分别报告，避免总体指标掩盖 onset 能力。

### 4.2 只检索受影响 operation 的 Trace DAG

在时间 t 只能使用已经完成的 span。根据高风险入口 operation，检索过去五分钟属于该 operation 的 traceID，重建请求级有向 DAG；不再把一分钟内所有 operation 混成一张图。

这一步同时解决两个问题：

- 图的方向来自真实 parent→child，不需要人为复制双向边。
- 未受影响 operation 的高频正常调用不会淹没异常路径。

### 4.3 在路径上分解 source 与 impact

对每个 span 计算：

- inclusive latency：当前 span 总 duration。
- exclusive latency：父 span 时间区间减去子 span 区间并集，避免并发子调用被重复相加。
- edge waiting：父子边相对同 operation 正常路径的额外等待。
- operation-conditioned deviation：只和相同 operation 的历史正常 trace 比较。

服务 i 的证据拆成：

- source score：本地 exclusive latency 增量、CPU/memory/disk/socket 偏离、日志模板新颖度。
- impact score：该服务下游 inclusive latency、受影响请求比例、到入口的路径贡献。

根因排序主要依据 source score；impact score 用来证明该异常确实解释入口退化。这样才是有数据支撑的“源—影响分解”，而不是再把两个向量送进一个 MLP。

### 4.4 Metric 和 Log 的角色

- Metric 是 source validator：验证某个路径节点是否存在本地资源异常，而不是负责全局预测。
- TT 被舍弃的 Mongo/MySQL 等 41 个资源作为 context node 接入，但不进入应用服务根因候选集合，避免标签空间改变。
- Log 不做主干。先把模板转成每服务的模板频率、首次出现、消失和分布漂移分数；只有模板变化显著时才作为 source score 的加成。
- 模态门控依据本窗口的观测覆盖和异常强度，不依据数据集名称，也不强迫 Log 始终参与。

## 5. 在实现完整模型前必须先过的三个数据门槛

### Gate A：Trace 路径分解审计

不训练神经网络，仅用正常基线的 operation-conditioned exclusive latency deviation 对根因排序，分别评估：

- 首个可见故障分钟
- 整个事件聚合
- 六种故障类型

必须稳定超过当前简单“服务平均 duration deviation”，并在两个数据集至少都不下降，才实现路径编码器。

### Gate B：日志模板增量审计

比较 `log_count/error_count` 与模板分布漂移的 RCA/预测增量。如果 TT 仍无增量，就在正式方法中删除 Log 主干，只保留为可选解释证据。

### Gate C：TT 资源上下文审计

把 41 个额外资源映射到应用服务，仅计算其训练期正常偏离并作为应用服务上下文。若 TT 的 Metric RCA 排序没有稳定提升，就不构造大型异构图。

这三个 Gate 都是固定特征、低容量、五折验证，不需要先训练完整模型。它们可以防止再次出现“模型做完后才发现数据里没有对应信号”。

## 6. 当前建议

首选先做 Gate A。原因是：Trace 已经是两个数据集最强业务影响模态，原始数据又恰好保留了完整 parent-child 路径和 operation，但当前模型把这些结构压掉了。这是目前唯一同时满足“有未使用数据”“直接对应根因—症状混淆”“能够解释有向图失败”的方向。

如果 Gate A 不通过，就应停止在 RE2 上继续设计复杂 RCA 结构，转而把论文贡献定位为严格任务定义、全面基线和数据边界分析；不能再靠结构搜索制造创新点。
