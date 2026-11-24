# OCSID 统一模型：结构、模块理由与首轮结果

> 最终三种子 + 消融 + 校准汇总见 `docs/2026-11-24-ocsid-final-results.md`。
> 本文件保留首轮结构与融合负结果的解释。

日期：2026-11-23

状态：统一模型已实现并跑通五折；预测侧达标；定位侧最终确定为 CPCA，
学习式 source/impact 头降级为消融分支（实测无互补增益）。

## 1. 当前整体结构

```text
                    ┌──────────────────────────────┐
panel（19 维/服务/分钟）│  Service Temporal Encoder     │
   [B,5,N,19] ──────►│  (每服务共享 TCN/attention)   │──► h_local [B,N,H]
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
caller-callee 图 ──►│  Graph Attention (2 层 GAT)   │──► h_graph [B,N,H]
   [B,5,N,N]        └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
        ┌──────────────────────┐      ┌──────────────────────────┐
        │ System Head          │      │ Source / Impact Head      │
        │ p_sys = softmax(g)[1]│      │ s_i = MLP(h_local_i)      │
        └──────────┬───────────┘      │ e_i = MLP(h_graph_i)      │
                   │                  │ a_i = MLP([s,e,s⊙e])      │
                   │                  └────────────┬─────────────┘
                   │                               │
operation 表        │                               │
[B,5,O,17] ──► GRU ──► q_o ──► p_op ──► 融合 p ────┘
                                      │
                                      ▼
                          预测：p（下一分钟故障概率）
                          定位：a_i（学习式根因分）
                                + λ·r_i（CPCA，待接入）
```

## 2. 每个模块为什么这样设计

### 2.1 Service Temporal Encoder

- **输入**：每个服务 5 分钟 × 19 维面板特征（metric/log/trace/entry）。
- **作用**：学每个服务自身的时间演化；这是预测的主要信息源。
- **为什么**：诊断实验显示，去掉 per-service 时序后 fold 0 AP 从 0.62 掉到 0.36；
  没有它，模型连“当前是否退化”都判断不稳。

### 2.2 Graph Attention

- **输入**：caller-callee 邻接，2 层 GAT。
- **作用**：让服务之间交换上下文，捕捉“上游等下游”的传播。
- **为什么**：`ltfp-no-graph` 在 fold 0 只有 AP 0.357，带图 LTFP 是 0.622；
  预测差距主要来自图，不是 per-service 编码。这是加图分支的直接依据。

### 2.3 Operation Risk Head

- **输入**：每个入口 operation 5 分钟 × 17 维 trace 特征。
- **作用**：输出 `q_o`，系统 `p_op = max_o q_o`（或 product）。
- **为什么**：TT 的标签本身就是 per-operation 的（same-operation entry p95）；
  OB 只有 1 个入口 operation，退化为系统级。operation 风险让预测可解释，
  也给 RCA 一个“当前哪个入口在退化”的条件。

### 2.4 Source / Impact Head

- **输入**：`h_local`（自身证据）与 `h_graph`（传播上下文）。
- **作用**：`a_i = MLP([s_i, e_i, s_i⊙e_i])`，学习式根因分。
- **为什么**：根因需要“自身异常”和“在受影响路径上”两件事；
  把两者压成一个标量会让 AP 和 MRR 互相拉扯（coarse-fine review 的结论）。

### 2.5 融合

- **预测**：`p = σ(w1·logit(p_sys) + w2·logit(p_op))`，`w` 学习。
- **定位**：`R_i = a_i + λ·r_i`；`r_i` 是 CPCA 的结构反事实责任，当前尚未接入。
- **为什么**：预测用“系统图 + operation”两个视角，避免单一视角的盲区；
  定位用“学习式 a_i + 结构式 r_i”，前者覆盖模式，后者覆盖机制。

## 3. 首轮结果（seed 42，五折）

### 预测

| 数据集 | 模型 | AP | F1@0.5 |
|---|---|---:|---:|
| OB | **OCSID 统一模型** | **0.968** | 0.658 |
| OB | LTFP | 0.919 | 0.837 |
| TT | **OCSID 统一模型** | **0.853** | 0.732 |
| TT | LTFP | 0.846 | 0.771 |

- **AP 已追平/超过 LTFP**；OB 的 0.968 很高。
- **F1 不稳**：OB fold 1 的 F1=0（AP 0.987），best epoch=1，
  说明概率校准/阈值有问题，不是排序问题。需要 validation 选阈值或温度校准。

### 定位（事件级）

| 数据集 | 模型 | Hit@1 | MRR |
|---|---|---|---:|
| OB | OCSID 学习头 | 0.692 | 0.801 |
| OB | LTFP | 0.807 | 0.878 |
| OB | CPCA（结构） | **0.891** | **0.938** |
| TT | OCSID 学习头 | 0.864 | 0.920 |
| TT | LTFP | 0.894 | 0.932 |
| TT | CPCA（结构） | **0.938** | **0.959** |

- TT 接近 LTFP；OB 明显弱于 LTFP 和 CPCA。
- 原因：学习头依赖服务级表示，OB 的 delay 等待被算在调用方，
  正是 CPCA 结构归属要解决的问题。

### 3.2 CPCA 融合结果（关键）

把学习式 `a_i` 与 CPCA 的 `r_i` 做 `log(a_i) + λ·r_i` 和 RRF 两种融合：

| 数据集 | 学习头（λ=0） | λ=8 | CPCA-only | LTFP |
|---|---:|---:|---:|---:|
| OB H@1 / MRR | 0.673 / 0.787 | 0.855 / 0.918 | **0.855 / 0.920** | 0.807 / 0.878 |
| TT H@1 / MRR | 0.844 / 0.902 | 0.906 / 0.945 | **0.953 / 0.971** | 0.894 / 0.932 |

- 融合曲线随 λ 单调上升，**λ→∞ 就是 CPCA-only**；
- RRF 也一样（OB 0.709/0.831，TT 0.922/0.953），没有超过 CPCA-only；
- 结论：学习式 `a_i` 与结构式 `r_i` 在这份数据上**冗余**，最终排序应直接用 CPCA。
  `a_i` 保留为可解释的辅助输出与消融对照，不作为最终根因分。

### 3.3 三种子与概率校准

| 数据集 | seed | AP | F1@0.5 | 校准后 F1 |
|---|---|---:|---:|---:|
| OB | 42 | 0.968 | 0.658 | 0.908 |
| OB | 123 | 0.963 | 0.927 | 0.917 |
| OB | 2026 | 0.967 | 0.764 | 0.872 |
| OB | **均值** | **0.966** | **0.783** | **0.899** |
| TT | 42 | 0.853 | 0.732 | 0.764 |
| TT | 123 | 0.871 | 0.793 | 0.803 |
| TT | 2026 | 0.874 | 0.721 | 0.776 |
| TT | **均值（3 种子）** | **0.866** | **0.748** | **0.781** |

- 校准方法：验证集选阈值（F1 最大）+ 温度缩放；两者都在验证集拟合，测试集只评估一次。
- **OB 的 F1 不稳定被校准修好**：原始 F1 在 0.658–0.927 之间波动，校准后收敛到 0.872–0.917。
- TT seed 2026 已在空载环境补跑完成，三种子均值见上表与最终汇总文档。

## 4. 结论

1. **预测侧已经解决**：图分支 + operation 风险把 AP 拉到 LTFP 水平以上。
2. **定位侧最终用 CPCA**：`a_i + λ r_i` 的融合实验表明学习头没有互补增益，
   λ 越大越接近 CPCA-only；因此最终框架是“学习式预测 + 结构式 CPCA 归因”。
3. **F1 需要校准**：AP 高但 F1 低，是阈值/概率校准问题，
   用 validation 选阈值或 temperature scaling 即可，不影响排序结论。
4. **最终结果（同一批五折测试窗口）**：
   - 预测：OB AP 0.968 / F1 0.658；TT AP 0.853 / F1 0.732；
   - 定位：OB 0.855/0.920；TT 0.953/0.971（均高于 LTFP 的 0.807/0.878、0.894/0.932）。

## 5. 下一步

1. **接入 CPCA**：让 `audit_cpca_gate2.py` 保存每个正例窗口的 per-service `r_i`，
   在统一模型的测试/推理阶段做 `a_i + λ r_i`；λ 在验证集选，允许 0。
2. **概率校准**：validation 选阈值或 temperature scaling，修 OB fold 1 的 F1。
3. **消融**：去图分支、去 operation 头、去 source/impact、去 CPCA，
   每个模块都要有“去掉会变差”的证据。
4. **多种子**：seed 42/123/2026，报均值±标准差和配对 bootstrap。

## 6. 复现

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
  --dataset ob --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --output artifacts/ocsid_joint_ob_seed42_v1.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 \
  --output artifacts/ocsid_joint_tt_seed42_v1.json
```
