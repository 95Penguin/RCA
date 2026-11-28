# ShapleyIQ / Sage 对照实验

## 实验结果

已取得并运行 ShapleyIQ 作者公开的核心实现，将 RE2 span CSV 映射到该算法需要的调用树和时间线。两种数据集都使用相同的 corrected-v2 正例预测 cohort、相同服务全集、每个入口操作/分钟最多 100 条有效请求、entry-component 有效树和 1 ms 时钟校准；评估采用全服务排序、缺失分数补零、并列取最差名次。

| 数据集 | 方法 | Event Hit@1 | Event MRR | Window Hit@1 | Window MRR |
|---|---|---:|---:|---:|---:|
| OB | CPCA 主分数 | 0.855 | 0.920 | 0.798 | 0.878 |
| OB | ShapleyIQ 作者默认（无 normal-duration cache） | 0.073 | 0.353 | 0.029 | 0.309 |
| OB | ShapleyIQ + 前 10 分钟正常耗时均值 | 0.109 | 0.389 | 0.044 | 0.335 |
| TT | CPCA 主分数 | 0.922 | 0.957 | 0.798 | 0.829 |
| TT | ShapleyIQ 作者默认（无 normal-duration cache） | 0.422 | 0.699 | 0.500 | 0.732 |
| TT | ShapleyIQ + 前 10 分钟正常耗时均值 | 0.813 | 0.904 | 0.849 | 0.919 |

正例 cohort 为 OB 455 个窗口/55 个 event、TT 436 个窗口/64 个 event。TT 上 CPCA 的 uniform-operation 对照更强：Event MRR 0.992、Window MRR 0.964；OB 上它与 CPCA 主分数相同。

## 怎么读这些结果

- ShapleyIQ 作者类的默认 `using_cache=False` 会把正常耗时当作 0，因此主要归因的是请求总耗时。表里同时保留了这个原样默认结果。
- 第二行 ShapleyIQ 变体把前 10 分钟有效健康请求的操作均值作为 normal-duration 参考，再调用同一 ShapleyIQ 算法。这是为 RE2 做的离线参考适配，不是作者代码开箱即用的配置。
- warm-up 参考明显改善 TT，但 OB 仍较弱；在 TT 的 window MRR 上它高于当前 CPCA 主分数，event MRR 仍低于 CPCA，且低于 CPCA uniform-operation 对照。跨两个数据集看，CPCA 的 event/window MRR 都更稳定。
- 结果说明 ShapleyIQ 是可运行且有意义的 RCA 对照；不能只报它较弱的默认缓存设置，论文里应注明正常耗时参考和数据集差异。

## 源码与复现文件

- 作者仓库：[lonyle/ShapleyIQ](https://github.com/lonyle/ShapleyIQ)
- 核心算法源码：[shapley_value_rca.py](https://raw.githubusercontent.com/lonyle/ShapleyIQ/main/rca4tracing/rca/shapley_value_rca.py)，本地副本在 `third_party/shapley_value_rca.py`，未改作者算法代码。
- RE2 适配和运行脚本：[run_shapleyiq_re2.py](run_shapleyiq_re2.py)
- 指标收集脚本：[collect_shapleyiq_comparison.py](collect_shapleyiq_comparison.py)
- 逐数据集完整结果：[`shapleyiq/ob/results_fastmerge100.json`](shapleyiq/ob/results_fastmerge100.json)、[`shapleyiq/tt/results_fastmerge100.json`](shapleyiq/tt/results_fastmerge100.json)
- CPCA 对照和源文件 SHA-256：[SHAPLEYIQ_SAGE_COMPARISON.json](SHAPLEYIQ_SAGE_COMPARISON.json)

为了缩短作者实现中时间线合并的运行时间，适配器只优化了候选时间段搜索范围；原有严格重叠条件、候选顺序、同步区间合并和贡献分配规则均保持不变。优化版与作者函数在 3,000 组随机时间线（包含 epoch 微秒量级）上逐项一致；OB 全数据运行与未优化作者函数逐窗口分数也完全一致。

## Sage 状态

目前没有找到可确认由 Sage 作者发布、可直接运行的实现，因此本轮没有把自写近似版冒称为 Sage，也没有产出 Sage 指标。Sage 原论文使用 Causal Bayesian Network 描述 RPC 依赖，并用 graphical VAE 生成反事实；论文：[Sage: Practical & Scalable ML-Driven Performance Debugging in Microservices](https://people.csail.mit.edu/delimitrou/papers/2021.asplos.sage.pdf)。要做 Sage 对照，需要按论文复现并训练这套 CBN/GVAE，而不是把 CPCA 分数替换名字。
