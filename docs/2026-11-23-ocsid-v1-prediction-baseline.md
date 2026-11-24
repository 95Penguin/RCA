# OCSID-V1 预测侧进度：operation 风险头与基线差距

日期：2026-11-23

状态：数据层与第一版预测头已跑通；**预测 AP 仍低于 LTFP**，需要补 per-service 上下文。
RCA 侧（CPCA）已由 Gate 2 验证，见 `2026-11-23-cpca-gate2-result.md`。

## 1. 做了什么

1. `scripts/build_ocsid_operation_features.py`：从原始 trace 生成
   “分钟 × 入口 operation”特征表。
   - TT：90 case，31,749 行，17 个 operation；
   - OB：90 case，2,246 行，1 个 operation（`frontendservice/frontend`）。
   - 特征：count、p50/p95/p99/mean/max、span_count、error_count、n_services、
     固定 10 分钟基线 p95/count、滚动 5 分钟基线。
2. `scripts/train_ocsid_predictor.py`：两种预测头。
   - `operation`：每个 operation 独立 5 分钟 GRU → `q_o`，`p = 1−∏(1−q_o)` 或 `max`；
   - `flat`：operation 聚合 + 面板 19 维 max/mean → 系统级 GRU。
3. 协议：deployment 五折（test=fold N，validation=(N+1)%5），
   5 分钟历史，best validation AP，seed 42，30 epochs。

## 2. 结果（RE2-TT，5 折）

| 模型 | AP | F1@0.5 | P | R |
|---|---:|---:|---:|---:|
| **OCSID operation** | **0.777±0.232** | **0.717±0.222** | 0.729 | 0.715 |
| OCSID flat | 0.699±0.190 | 0.672±0.152 | 0.653 | 0.692 |
| persistence（用上一分钟标签） | **0.830±0.063** | **0.888±0.040** | — | — |
| LTFP（正式基线） | **0.846±0.127** | **0.771±0.130** | — | — |

逐折 AP（operation）：`[0.383, 0.813, 0.941, 0.798, 0.953]`。

### 2.1 尝试过的改进（负结果）

给 `q_o` 拼接 per-service 面板上下文（最后分钟 + 5 分钟变化量）、
dropout 0.3、weight decay 1e-3、label smoothing、固定 epoch 选模：

| 配置 | AP | F1 | 逐折 AP |
|---|---:|---:|---|
| 原 operation（entry 上下文） | **0.777±0.232** | **0.717±0.222** | [0.383, 0.813, 0.941, 0.798, 0.953] |
| + 面板上下文 / 正则 / 固定 epoch | 0.718±0.201 | 0.673±0.210 | [0.405, 0.735, 0.824, 0.682, 0.944] |
| + 面板上下文 / 正则 / val AP | 0.697±0.205 | 0.647±0.178 | [0.361, 0.735, 0.800, 0.682, 0.905] |

**结论：没有变好。** fold 0 略升（0.383→0.405），但其余四折全部下降。
说明瓶颈不是“缺系统上下文”或“缺正则”，而是缺 per-service 时序编码。
默认配置已恢复为原 operation（entry 上下文），脚本保留
`--context panel --dropout 0.3 --weight-decay 1e-3 --selection last` 作为可复现的负结果入口。

## 3. 诊断

1. **fold 0 是主要缺口**：operation 模型 0.383，LTFP 0.622，persistence 0.725。
   其他四折 0.80–0.95，已接近 LTFP。
2. **验证选模放大了缺口**：fold 0 的验证折是 fold 1（较容易），
   选出的 epoch 在 fold 0 上不泛化；flat 模型 fold 0 best epoch 28、val AP 0.805、
   test AP 0.419，是典型的选择偏差。
3. **operation 模型缺少 per-service metric/log 上下文**：目前只有 trace 的
   operation 特征 + 5 维 entry 上下文；LTFP 的 0.846 来自 27 个服务 × 19 维时序。
4. **persistence 很高**（AP 0.830）：这份标签主要是状态延续；
   预测侧的真实增量在 onset（只有 10/20 个正例），不应把总体 AP 当作主卖点。

### 3.1 关键诊断：差距来自图，不是 per-service 编码

在 TT fold 0 上补跑仓库自带的 `ltfp-no-graph`（同样的面板特征、per-service 时序、
不做图传播）：

| 模型 | fold 0 AP | F1 |
|---|---:|---:|
| LTFP（带服务图） | **0.622** | 0.562 |
| ltfp-no-graph | 0.357 | 0.407 |
| OCSID operation | 0.383 | 0.352 |

**结论：** OCSID 的 operation 模型已经达到“无图 LTFP”的水平；
与 LTFP 的 0.24 AP 差距主要来自跨服务图传播，而不是缺少 per-service 时序编码。
因此预测侧有两条路：给 OCSID 加一个独立的图分支，或把预测交给检测器、论文聚焦 RCA。

## 4. 下一步（按性价比）

1. **给预测头加独立的图分支**：per-service 时序编码 + caller-callee GAT
   （不加载 LTFP 权重，独立训练）。这是唯一能把 fold 0 从 0.38 拉到 0.6+ 的已知路径。
   代价：预测头会更像 LTFP 的架构，但仍可用“operation 级 RCA”作为论文差异点。
2. **改选模策略**：用两个验证折的平均 AP，或固定 epoch；
   避免 fold 0 被 fold 1 的容易样本带偏。
3. **加正则/早停**：dropout 0.1→0.3，weight decay，label smoothing。
4. **如果预测侧仍差 0.05 AP 以上**：论文改为“检测器给定、RCA 为主”，
   预测头作为系统模块报告，不把 AP 作为主贡献。

## 5. 结论

- RCA 侧：Gate 2 有条件通过（OB 显著，TT 不劣）。
- 预测侧：第一版 `0.777/0.717`，比 LTFP 低约 `0.07 AP / 0.05 F1`，
  **还不够**，但差距集中在 fold 0，且原因明确（缺 per-service 上下文 + 选模偏差）。
- 下一步先补 per-service 上下文与选模，再判断是否继续 OCSID 联合训练。

## 6. 复现

```bash
# 特征表
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/build_ocsid_operation_features.py \
  --dataset tt --output artifacts/re2_tt_ocsid_operation_features_v1.csv

# 预测头（operation）
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_predictor.py \
  --dataset tt --folds 0 1 2 3 4 --epochs 30 --seed 42 --arch operation \
  --output artifacts/ocsid_predictor_tt_operation_seed42_v1.json
```
