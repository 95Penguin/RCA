# 定型审计：参考稳定性、全量覆盖和完整请求分层

冻结 median/replace/post/uniform。100 次健康分钟块重采样，整块请求及其所有分量共同抽取。
分位范围只描述参考变化，不能称为真实修复效果的置信区间。

## OB

| 请求集合（共享主健康参考） | 事件 H@1 | 事件 MRR | 窗口 MRR | 无证据窗口 |
|---|---:|---:|---:|---:|
| full | 0.8545 | 0.9198 | 0.8777 | 1 |
| complete_refill | 0.8545 | 0.9198 | 0.8779 | 1 |
| complete_shared_cap | 0.8545 | 0.9198 | 0.8779 | 1 |
| no_clock_shared_cap | 0.8545 | 0.9198 | 0.8777 | 1 |

原始冻结分数最大差：0；事件排名完全复现。
100 次参考重采样事件 MRR：{'mean': 0.9198347107438015, 'min': 0.9198347107438017, 'max': 0.9198347107438017, 'reference_resampling_range95': [0.9198347107438017, 0.9198347107438017]}。
事件名次平均一致率：1.0000；名次始终不变事件：55/55。

| 全量覆盖量 | 数值 |
|---|---:|
| 原始 trace ID | 2114622 |
| 原始 span 行 | 34461235 |
| 有识别入口的 trace | 1627778 |
| 接受 trace | 1627704 |
| 完整接受 / 部分接受 | 1624894 / 2810 |
| 接受请求中的 span 保留率 | 99.9830% |
| 正例入口 trace 接受率（不截断到100） | 99.9766% |
| 接受请求中发生时钟修改 | 101 |
| 最大 span 总开始位移（μs） | 130.0 |
| 总开始位移超过1ms的 span | 0 |

拒绝原因：{'rejected_no_recognized_business_entry': 486841, 'rejected_rejected_with_parent_child_boundary_violation': 2, 'rejected_duplicate_span_id': 75}
参考支持：{'cases_with_less_than_5_healthy_minute_blocks': 0, 'min_healthy_minute_blocks': 10, 'max_healthy_minute_blocks': 10, 'cutoff_after_injection_cases': [], 'evaluation_cases': 51, 'min_blocks_in_evaluation_cases': 10}

## TT

| 请求集合（共享主健康参考） | 事件 H@1 | 事件 MRR | 窗口 MRR | 无证据窗口 |
|---|---:|---:|---:|---:|
| full | 0.9844 | 0.9922 | 0.9640 | 0 |
| complete_refill | 0.9375 | 0.9533 | 0.9447 | 0 |
| complete_shared_cap | 0.9375 | 0.9533 | 0.9447 | 0 |
| no_clock_shared_cap | 0.9062 | 0.9479 | 0.8661 | 0 |

原始冻结分数最大差：8.33e-17；事件排名完全复现。
100 次参考重采样事件 MRR：{'mean': 0.992109375, 'min': 0.984375, 'max': 0.9921875, 'reference_resampling_range95': [0.9921875, 0.9921875]}。
事件名次平均一致率：0.9998；名次始终不变事件：63/64。

| 全量覆盖量 | 数值 |
|---|---:|
| 原始 trace ID | 573945 |
| 原始 span 行 | 67345051 |
| 有识别入口的 trace | 563647 |
| 接受 trace | 519550 |
| 完整接受 / 部分接受 | 501924 / 17626 |
| 接受请求中的 span 保留率 | 97.2059% |
| 正例入口 trace 接受率（不截断到100） | 92.0772% |
| 接受请求中发生时钟修改 | 216020 |
| 最大 span 总开始位移（μs） | 1989.0 |
| 总开始位移超过1ms的 span | 8533 |

拒绝原因：{'rejected_no_recognized_business_entry': 10298, 'rejected_rejected_with_parent_child_boundary_violation': 44097}
参考支持：{'cases_with_less_than_5_healthy_minute_blocks': 1, 'min_healthy_minute_blocks': 3, 'max_healthy_minute_blocks': 10, 'cutoff_after_injection_cases': [], 'evaluation_cases': 53, 'min_blocks_in_evaluation_cases': 10}

## 解释限制

- no_clock_shared_cap 仅保留原 cap 中完全不需要时间修改的请求；仍共享主参考且不补样，不等价于重跑完整的零容差模型。
- complete_shared_cap 是原样本内完整请求子集；complete_refill 允许按相同预算补取后续完整请求；两者均保留所有评价窗口，缺证据记零。
- 无识别入口的原始 trace 可能是内部片段或入口埋点缺失，不能一概当业务请求拒绝，也无法仅凭 span CSV 判断物理缺失原因。
- sibling overlap/boundary violation 是观测模式，不能直接区分真实并行、缺失记录和时钟差。
- 参考重采样分位范围不是干预效果置信区间，未参与重新选择主结构。
