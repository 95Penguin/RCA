# Transition-aware预测RCA可行性审计

结论：**CONDITIONAL GO: 研究问题和first-visible主任务可行；复杂Transition模型暂缓**。

这里把两个问题分开：研究问题是否成立，以及是否已经有足够证据开发复杂Transition模型。
置信区间以物理case为单位、在deployment折内重采样，避免把相邻分钟当成独立样本。

## 核心结果

| 数据集 | Persistence F1 | LTFP状态F1 | First-visible正例case | AP | 正例率 | AP lift | AP 95% CI | AP-正例率 95% CI | RCA Hit@1 |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| RE2-OB | 0.921 | 0.833 | 51 | 0.536 | 0.046 | 11.8× | [0.391, 0.690] | [0.345, 0.643] | 0.431 |
| RE2-TT | 0.894 | 0.785 | 53 | 0.320 | 0.049 | 6.5× | [0.195, 0.467] | [0.151, 0.412] | 0.792 |

## 严格相邻状态探针

| 数据集 | 严格onset正例 | State AP | State+change AP | 判读 |
|---|---:|---:|---:|---|
| RE2-OB | 10 | 0.018 | 0.027 | 变化特征有弱增益 |
| RE2-TT | 20 | 0.094 | 0.107 | 变化特征有弱增益 |

## Go / No-Go判定

### GO：论文问题与first-visible主任务

- 两个数据集的Persistence总体F1都高于LTFP，确认状态持续性会显著影响总体分数。
- 每个数据集都有超过50个独立first-visible正例case，而不是把持续故障分钟重复计数。
- 两个数据集的case-bootstrap `AP-正例率` 95%区间下界均高于0，说明首次故障前存在可泛化排序信号。
- First-visible RCA明显高于随机服务排序，预测与定位可以放在同一故事中。

### HOLD：复杂Transition架构

- 严格相邻状态可观测的onset只有OB 10个、TT 20个，不能单独支撑复杂多分支模型。
- State+change线性探针只带来小幅且数据集不稳定的改善，尚不足以证明需要复杂变化编码器。
- Recovery样本更少，现阶段只能作为辅助指标，不能作为主贡献。

## 下一步最小实验

在first-visible标签上实现一个轻量 `current-state prior + transition residual`，保持LTFP编码器、数据划分和RCA头不变。
只有当它在两个数据集中的至少一个达到 `AP +0.03`，且另一个数据集不明显下降，才继续扩展到1/3/5分钟联合模型。
严格10/20个相邻onset只做敏感性分析，不作为训练主样本集合。

