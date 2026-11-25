# 多 horizon（1/3/5 分钟）结果

日期：2026-11-24。内容：Gate 0（根因信号是否还在）、`state@1/3/5` 预测、
CPCA@1/3/5 事件级归因。全部 seed 42，五折 deployment 划分。

## 1. Gate 0：决策时刻还有根因信号吗

对每个正例窗口，取根因服务，用 case 前 10 分钟做基线；
任一特征 $|z|>3$ 或 `entry_p95` 超过基线 1.5 倍记为有信号。

### RE2-TT

| h | 任务 | 正例窗口 | 事件 | 窗口有信号 | 事件有信号 |
|---:|---|---:|---:|---:|---:|
| 1 | state | 436 | 64 | 0.991 | 1.000 |
| 3 | state | 383 | 62 | 0.909 | 0.935 |
| 5 | state | 385 | 63 | 0.704 | 0.857 |
| 1 | first-visible | 53 | 53 | 0.925 | 0.925 |
| 3 | first-visible | 72 | 53 | 0.944 | 0.925 |
| 5 | first-visible | 80 | 53 | 0.950 | 0.925 |

### RE2-OB

| h | 任务 | 正例窗口 | 事件 | 窗口有信号 | 事件有信号 |
|---:|---|---:|---:|---:|---:|
| 1 | state | 455 | 55 | 0.967 | 1.000 |
| 3 | state | 406 | 55 | 0.865 | 0.964 |
| 5 | state | 413 | 55 | 0.644 | 0.855 |
| 1 | first-visible | 51 | 51 | 0.765 | 0.765 |
| 3 | first-visible | 62 | 51 | 0.806 | 0.765 |
| 5 | first-visible | 70 | 51 | 0.800 | 0.765 |

**结论**：3/5 分钟事件级仍有 86%–96% 的根因信号，多 horizon 可做；
窗口级 5 分钟会掉，因为很多窗口是故障持续/衰减段。

## 2. 预测：state@1/3/5

| 数据集 | h | AP | F1（验证阈值） |
|---|---:|---:|---:|
| OB | 1 | 0.9675 | 0.9081 |
| OB | 3 | 0.9033 | 0.8485 |
| OB | 5 | 0.8579 | 0.6772 |
| TT | 1 | 0.8530 | 0.7636 |
| TT | 3 | 0.8374 | 0.7233 |
| TT | 5 | 0.7993 | 0.6561 |

预测随 horizon 单调下降，符合预期；5 分钟的 F1 掉得较多，
说明“5 分钟后是否仍故障”比“3 分钟后”更难，部署时需要单独校准阈值。

## 3. CPCA：事件级 1/3/5 分钟

| 数据集 | h | 事件数 | H@1 | H@3 | MRR |
|---|---:|---:|---:|---:|---:|
| OB | 1 | 54 | 0.907 | 1.000 | 0.951 |
| OB | 3 | 49 | 0.898 | 1.000 | 0.946 |
| OB | 5 | 46 | **0.935** | 1.000 | **0.964** |
| TT | 1 | 64 | 0.953 | 1.000 | 0.974 |
| TT | 3 | 55 | **0.982** | 1.000 | **0.988** |
| TT | 5 | 49 | 0.959 | 1.000 | 0.976 |

（这里是 p95-soft 结果；hard 版本略低：OB 0.857/0.925/0.870，TT 0.952/0.982/0.958。）

## 4. 解读

1. **CPCA 对 horizon 稳定**：OB 从 0.907 到 0.898/0.935，TT 从 0.953 到 0.982/0.959；
   没有出现“3/5 分钟崩掉”。这与 Gate 0 的事件级信号一致。
2. **不能宣称“越远越准”**：h=3 TT、h=5 OB 的数字略高，但事件集变小（64→55/49），
   且更长 horizon 的正例往往对应更持续、根因更明显的故障，存在选择效应。
   需要按共同事件做配对比较后才能下结论。
3. **预测与归因的 horizon 行为不同**：预测随 h 明显下降（AP 0.9675→0.8579 OB），
   但归因稳定。这说明“根因在这些事件里已经可观测”，难的是“判断 h 分钟后是否仍故障”，
   不是“判断是谁”。
4. **first-visible 尚未训练**：Gate 显示可行，但 hazard 头/首次可见任务还没做；
   当前所有结果都是 `state@h`。

## 5. 实现边界（诚实声明）

- 每个 horizon **单独训练**一个预测器，还没有共享编码器 + horizon embedding；
- CPCA 仍用决策分钟 $t$ 的请求，**没有 trailing window**；
- 事件集随 h 变化，跨 horizon 的差异未做配对 bootstrap；
- `first-visible@h` 只有 Gate，没有模型结果；
- 5 分钟的 F1 较低，部署时需单独选阈值。

## 6. 下一步

1. **共享编码器 + horizon embedding**（或 hazard 头）替代“每个 h 一个模型”；
2. **CPCA trailing window**：$w=\max(1,h)$，降低单分钟方差；
3. **共同事件配对 bootstrap**：确认 h=3/5 的差异是否显著；
4. **first-visible + hazard**：$p_h=1-\prod_{k\le h}(1-\lambda_k)$，保证单调。

## 7. 复现

```bash
# Gate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/gate_horizon_leadtime.py \
  --dataset tt --horizons 1 3 5 --output artifacts/gate_horizon_tt_v1.json

# 预测 state@h
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 --horizon-minutes 3 \
  --output artifacts/ocsid_joint_tt_h3_seed42_v3.json

# CPCA@h
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 --horizon-minutes 3 \
  --output artifacts/cpca_h3_tt_first10_z2_scores_v1.json
```
