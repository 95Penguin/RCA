# LTFP Transition Residual最小实验

结论：**NO-GO for this minimal architecture**。

所有比较使用相同的first-visible 1分钟标签、deployment五折、30 epoch、seed 42和RCA loss。
置信区间对同一物理case的两种模型预测进行配对重采样。

## 五折均值

| 数据集 | LTFP AP | Transition AP | ΔAP | LTFP Hit@1 | Transition Hit@1 | ΔHit@1 |
|---|---:|---:|---:|---:|---:|---:|
| RE2-OB | 0.621 | 0.539 | -0.082 | 0.417 | 0.467 | +0.050 |
| RE2-TT | 0.409 | 0.361 | -0.048 | 0.807 | 0.747 | -0.060 |

## 配对case-bootstrap

| 数据集 | Pooled ΔAP | ΔAP 95% CI | Pooled ΔHit@1 | ΔHit@1 95% CI |
|---|---:|---|---:|---|
| RE2-OB | -0.230 | [-0.384, +0.008] | +0.059 | [-0.048, +0.170] |
| RE2-TT | -0.053 | [-0.151, +0.084] | -0.057 | [-0.167, +0.055] |

## 判定

预设继续门槛是：至少一个数据集平均AP提升0.03，另一个数据集不明显下降，且RCA不退化。
当前最小结构没有达到门槛：两个数据集的平均AP都下降，提升只出现在部分deployment折。
因此不继续在这个结构上增加更多分支；该负结果不否定persistence-bias和first-visible评测故事，
但说明简单的logit分解本身不能作为方法贡献。

