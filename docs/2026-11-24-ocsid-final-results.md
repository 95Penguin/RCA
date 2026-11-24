# OCSID 最终实验结果（2026-11-24）

本文件汇总 OCSID 统一模型的预测、定位、消融与校准结果。
所有数字来自 `artifacts/ocsid_*_v3.json`、`artifacts/ocsid_summary_v3.json`
和 `artifacts/ocsid_cpca_fusion_*_v1.json`。

## 1. 协议

- 数据集：RE2-OB（55 事件）、RE2-TT（64 事件），90 case/数据集。
- 切分：deployment-group 五折（test=fold N，validation=(N+1)%5）。
- 预测：5 分钟历史；OCSID 三种子（42/123/2026），报五折均值的种子均值±标准差。
- 定位：事件级 Hit@1/MRR；OCSID 学习头为三种子均值，CPCA 为确定性方法。
- 对照：LTFP 来自正式五折结果（seed 42），persistence 来自持久化审计（窗口级参考）。
- 校准：验证集选阈值（F1 最大）+ 温度缩放，测试集只评估一次。

## 2. 预测主表

| 数据集 | 模型 | AP | F1@0.5 | 校准后 F1 |
|---|---|---:|---:|---:|
| OB | **OCSID（3 seeds）** | **0.966 ± 0.002** | 0.783 ± 0.135 | **0.899 ± 0.024** |
| OB | LTFP | 0.919 ± 0.026 | 0.837 ± 0.083 | — |
| OB | persistence | 0.884 | 0.921 | — |
| TT | **OCSID（3 seeds）** | **0.866 ± 0.011** | 0.748 ± 0.039 | **0.781 ± 0.020** |
| TT | LTFP | 0.846 ± 0.127 | 0.771 ± 0.130 | — |
| TT | persistence | 0.844 | 0.894 | — |

**读法：**

1. OCSID 的 AP 在 OB 上高 0.047、TT 上高 0.020，且种子方差很小（±0.002/±0.011）。
2. 原始 F1@0.5 在 OB 上跨种子波动很大（0.658–0.927），**校准后收敛到 0.872–0.917**。
3. persistence 的 F1 仍然最高（0.921/0.894），但它使用上一分钟的**标签本身**，
   不能作为可部署模型；它的 AP（0.884/0.844）低于 OCSID。
4. LTFP 是单种子结果，OCSID 是三种子；严格配对比较需要重跑 LTFP 的 123/2026。

### 2.1 逐种子明细

| 数据集 | seed | AP | 原始 F1 | 校准 F1 | 阈值 |
|---|---|---:|---:|---:|---:|
| OB | 42 | 0.9675 | 0.6580 | 0.9081 | 0.42 |
| OB | 123 | 0.9632 | 0.9265 | 0.9167 | 0.47 |
| OB | 2026 | 0.9671 | 0.7636 | 0.8721 | 0.48 |
| TT | 42 | 0.8530 | 0.7317 | 0.7636 | 0.42 |
| TT | 123 | 0.8709 | 0.7926 | 0.8030 | 0.38 |
| TT | 2026 | 0.8743 | 0.7205 | 0.7764 | 0.44 |

## 3. 定位主表（事件级）

| 数据集 | 方法 | Hit@1 | MRR |
|---|---|---|---:|
| OB | OCSID 学习头（3 seeds） | 0.738 ± 0.057 | 0.833 ± 0.042 |
| OB | **CPCA-only** | **0.855** | **0.920** |
| OB | LTFP | 0.807 | 0.878 |
| TT | OCSID 学习头（3 seeds） | 0.864 ± 0.002 | 0.922 ± 0.002 |
| TT | **CPCA-only** | **0.953** | **0.971** |
| TT | LTFP | 0.894 | 0.932 |

**读法：**

- 最终定位用 **CPCA**，不是学习头；学习头只作为可解释辅助与消融。
- CPCA 在 OB 上 H@1 高 0.048、MRR 高 0.042；TT 上高 0.059 / 0.039。
- CPCA 覆盖了 90.3%（OB）/ 91.5%（TT）的正例窗口；未覆盖窗口回退到学习头。

## 4. 融合与负结果

`score_i = log(a_i) + λ·r_i`，λ 从 0 到 8：

| 数据集 | λ=0（学习头） | λ=2 | λ=8 | CPCA-only |
|---|---|---|---:|---:|
| OB H@1 / MRR | 0.673 / 0.787 | 0.818 / 0.889 | 0.855 / 0.918 | **0.855 / 0.920** |
| TT H@1 / MRR | 0.844 / 0.902 | 0.859 / 0.914 | 0.906 / 0.945 | **0.953 / 0.971** |

- 融合曲线随 λ 单调上升，**λ→∞ 就是 CPCA-only**；
- RRF 同样没有超过 CPCA-only（OB 0.709/0.831，TT 0.922/0.953）；
- 结论：学习式 `a_i` 与结构式 `r_i` 在这份数据上冗余，最终排序直接用 CPCA。
  这是一个可写进论文的负结果：结构反事实证据已经覆盖学习式服务证据。

## 5. 消融（seed 42）

| 数据集 | 完整模型 AP | no-graph AP | no-operation AP |
|---|---:|---:|---:|
| OB | 0.9675 | 0.9652 | 0.9398 |
| TT | 0.8530 | 0.8239 | 0.8187 |

**读法：**

1. **Operation 风险头在两个数据集上都必要**：OB −0.028、TT −0.034。
2. **图分支只在 TT 上必要**：TT −0.029；OB 基本不变（−0.002）。
3. 这与数据集结构一致：OB 只有 1 个入口 operation（`frontendservice/frontend`），
   图传播没有额外关系可学；TT 有 17 个入口 operation，跨服务关系重要。
4. 该结论也解释了为什么“图是预测差距主因”只在 fold 0/早期实验成立：
   加入 operation 风险头后，OB 的图不再是瓶颈。

## 6. 关键结论

1. **预测侧达标**：OCSID AP 0.966/0.866，均高于 LTFP 0.919/0.846；
   校准后 F1 0.899/0.781，OB 的 F1 不稳定性被修复。
2. **定位侧达标**：CPCA-only 0.855/0.920（OB）、0.953/0.971（TT），
   均高于 LTFP 0.807/0.878、0.894/0.932。
3. **学习式 RCA 头冗余**：融合曲线与 RRF 都证明 `a_i` 没有增量；
   最终框架是“学习式预测 + 结构式 CPCA 归因”。
4. **模块必要性有数据**：operation 头两边都必要；图分支只在 TT 必要；
   这与 OB 单入口 operation 的结构一致，是可解释的边界而不是缺陷。

## 7. 局限（论文 threats to validity）

1. **LTFP 是单种子**，OCSID 是三种子；严格比较需要补 LTFP 的 123/2026。
2. **消融是单种子（42）**：需要在 3 seeds 上重复 no-graph / no-operation。
3. **CPCA 的 normal reference** 用 case 前 10 分钟；这是因果 warm-up，
   但仍是每 case 单独估计，和在线滚动参考有差距。
4. **CPCA 覆盖率** 90.3%/91.5%；未覆盖事件回退到学习头，论文需报告覆盖率。
5. **事件数少**（55/64）：主表应补事件级配对 bootstrap 与每事件 win/loss。
6. **F1 原始值波动大**：主表同时报 raw 与 calibrated，并说明校准在验证集拟合。

## 8. 复现

```bash
# 预测 + 定位（三种子）
for seed in 42 123 2026; do
  for ds in ob tt; do
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
      --dataset $ds --folds 0 1 2 3 4 --epochs 30 --seed $seed \
      --output artifacts/ocsid_joint_${ds}_seed${seed}_v3.json
  done
done

# 消融
for ds in ob tt; do
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
    --dataset $ds --folds 0 1 2 3 4 --epochs 30 --seed 42 --no-graph \
    --output artifacts/ocsid_joint_${ds}_nograph_v3.json
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/train_ocsid_joint.py \
    --dataset $ds --folds 0 1 2 3 4 --epochs 30 --seed 42 --no-operation \
    --output artifacts/ocsid_joint_${ds}_nooperation_v3.json
done

# 汇总
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/summarize_ocsid_runs.py
```

CPCA 分数与融合：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_cpca_gate2.py \
  --dataset tt --cases $(cat /tmp/tt_cases.txt | tr '\n' ' ') \
  --max-requests 100 --normal-mode first10 \
  --output artifacts/cpca_gate2_tt_first10_z2_scores_v1.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/fuse_ocsid_cpca.py \
  --dataset tt --ocsid artifacts/ocsid_joint_tt_seed42_v3.json \
  --cpca artifacts/cpca_gate2_tt_first10_z2_scores_v1.json \
  --output artifacts/ocsid_cpca_fusion_tt_v1.json
```
