# 同 cohort 基线重算（修正负 logits 弃权错误）

已有作者代码适配版本；seed42；五折测试裁到当前固定窗口。未重训基线。
CPCA 的全零恢复分按自身定义弃权；基线的合法负 logits 不弃权。
每个可用基线向量加同一个常数，保持窗口排名和事件均值排名不变；缺失输出仍为零证据。

## OB

| 方法 | 事件 H@1 | 事件 MRR | 窗口 H@1 | 窗口 MRR |
|---|---:|---:|---:|---:|
| cpca_uniform | 0.8545 | 0.9198 | 0.7978 | 0.8777 |
| train_event_root_prior | 0.0364 | 0.3580 | 0.0484 | 0.3957 |
| train_event_operation_root_prior | 0.0364 | 0.3580 | 0.0484 | 0.3957 |
| global_z | 0.1818 | 0.4297 | 0.0220 | 0.3028 |
| global_z_stored_native | 0.1818 | 0.4297 | 0.0220 | 0.3028 |
| global_z_rank_all_finite | 0.1818 | 0.4297 | 0.0220 | 0.3035 |
| inclusive_z | 0.2000 | 0.4145 | 0.0462 | 0.3030 |
| inclusive_z_stored_native | 0.2000 | 0.4145 | 0.0462 | 0.3030 |
| inclusive_z_rank_all_finite | 0.2000 | 0.4145 | 0.0462 | 0.3040 |
| exclusive_z | 0.4545 | 0.5790 | 0.3363 | 0.4871 |
| exclusive_z_stored_native | 0.4545 | 0.5790 | 0.3363 | 0.4871 |
| exclusive_z_rank_all_finite | 0.4545 | 0.5790 | 0.3363 | 0.4896 |
| exclusive_p95_z | 0.2727 | 0.4420 | 0.1495 | 0.3518 |
| exclusive_p95_z_stored_native | 0.2727 | 0.4420 | 0.1495 | 0.3518 |
| exclusive_p95_z_rank_all_finite | 0.2727 | 0.4420 | 0.1495 | 0.3554 |
| cpca_local | 0.1273 | 0.4035 | 0.1956 | 0.4354 |
| cpca_edge | 0.8364 | 0.9062 | 0.8044 | 0.8840 |
| cpca_caller | 0.1091 | 0.3884 | 0.0418 | 0.3397 |
| ltfp | 0.7818 | 0.8738 | 0.8110 | 0.8867 |
| art | 0.5818 | 0.7046 | 0.5275 | 0.6434 |
| eadro | 0.6000 | 0.7635 | 0.5253 | 0.7193 |
| instantops | 0.2000 | 0.4835 | 0.2703 | 0.4671 |
| circa | 0.7273 | 0.8083 | 0.8088 | 0.8586 |
| tracerca | 0.4545 | 0.6545 | 0.5319 | 0.7045 |
| microrank | 0.2000 | 0.5138 | 0.2462 | 0.5425 |
| tvdiag | 0.4545 | 0.6286 | 0.4813 | 0.6560 |
| torai | 0.5091 | 0.6755 | 0.5538 | 0.7046 |
| rcd | 0.4545 | 0.6087 | 0.4154 | 0.5097 |
| diagfusion | 0.2727 | 0.5230 | 0.3165 | 0.5566 |
| shapleyiq_default_old_adapter | 0.0727 | 0.3526 | 0.0286 | 0.3091 |
| shapleyiq_warmup10_old_adapter | 0.1091 | 0.3893 | 0.0440 | 0.3347 |
| shapleyiq_default_aligned | 0.0727 | 0.3526 | 0.0286 | 0.3091 |
| shapleyiq_warmup10_aligned | 0.1091 | 0.3893 | 0.0440 | 0.3347 |
| shapleyiq_default_uniform_aligned | 0.0727 | 0.3483 | 0.0286 | 0.3091 |
| shapleyiq_warmup10_uniform_aligned | 0.1091 | 0.3868 | 0.0440 | 0.3347 |

## TT

| 方法 | 事件 H@1 | 事件 MRR | 窗口 H@1 | 窗口 MRR |
|---|---:|---:|---:|---:|
| cpca_uniform | 0.9844 | 0.9922 | 0.9381 | 0.9640 |
| train_event_root_prior | 0.1719 | 0.4576 | 0.1193 | 0.4132 |
| train_event_operation_root_prior | 0.1719 | 0.4576 | 0.1193 | 0.4132 |
| global_z | 0.4688 | 0.6333 | 0.5642 | 0.6871 |
| global_z_stored_native | 0.3594 | 0.5738 | 0.2225 | 0.4216 |
| global_z_rank_all_finite | 0.4688 | 0.6333 | 0.5642 | 0.6882 |
| inclusive_z | 0.6406 | 0.7635 | 0.6307 | 0.7383 |
| inclusive_z_stored_native | 0.4062 | 0.6026 | 0.2339 | 0.4196 |
| inclusive_z_rank_all_finite | 0.6406 | 0.7635 | 0.6307 | 0.7383 |
| exclusive_z | 0.6562 | 0.7504 | 0.6399 | 0.7341 |
| exclusive_z_stored_native | 0.8438 | 0.8893 | 0.6995 | 0.7651 |
| exclusive_z_rank_all_finite | 0.6562 | 0.7504 | 0.6399 | 0.7341 |
| exclusive_p95_z | 0.4688 | 0.6371 | 0.4794 | 0.6254 |
| exclusive_p95_z_stored_native | 0.7031 | 0.7958 | 0.6193 | 0.7084 |
| exclusive_p95_z_rank_all_finite | 0.4688 | 0.6371 | 0.4794 | 0.6254 |
| cpca_local | 0.5469 | 0.6763 | 0.5894 | 0.7042 |
| cpca_edge | 0.8594 | 0.9206 | 0.8486 | 0.8998 |
| cpca_caller | 0.9531 | 0.9680 | 0.9312 | 0.9557 |
| ltfp | 0.8750 | 0.9232 | 0.8693 | 0.9145 |
| art | 0.6406 | 0.7034 | 0.5206 | 0.5992 |
| eadro | 0.6094 | 0.7622 | 0.6078 | 0.7762 |
| instantops | 0.3125 | 0.4050 | 0.2110 | 0.3156 |
| circa | 0.5625 | 0.6713 | 0.4541 | 0.5930 |
| tracerca | 0.1250 | 0.2761 | 0.1284 | 0.2765 |
| microrank | 0.1250 | 0.3401 | 0.1468 | 0.3892 |
| tvdiag | 0.7812 | 0.8504 | 0.6835 | 0.7814 |
| torai | 0.3750 | 0.5226 | 0.3280 | 0.4588 |
| rcd | 0.3125 | 0.4174 | 0.2661 | 0.3271 |
| diagfusion | 0.2344 | 0.5266 | 0.2202 | 0.5104 |
| shapleyiq_default_old_adapter | 0.4219 | 0.6986 | 0.5000 | 0.7319 |
| shapleyiq_warmup10_old_adapter | 0.8125 | 0.9036 | 0.8486 | 0.9188 |
| shapleyiq_default_aligned | 0.4219 | 0.6986 | 0.5000 | 0.7319 |
| shapleyiq_warmup10_aligned | 0.8125 | 0.9036 | 0.8486 | 0.9188 |
| shapleyiq_default_uniform_aligned | 0.5000 | 0.6675 | 0.4794 | 0.6436 |
| shapleyiq_warmup10_uniform_aligned | 0.9375 | 0.9648 | 0.9220 | 0.9534 |

旧 ShapleyIQ 使用原有 per-trace 平均，不能称为 operation 等权；aligned 控制另补 operation 等权。
简单控制按每个 operation 的原始分数显式等权重算；global_z 等仍按其定义将正异常作为证据。
分数语义、训练输入和标签适配不同；本表不等价于各原论文完整系统的同环境复现。
逐折核验、负 logits 计数、事件配对区间见 BASELINES_UNIFIED.json。
