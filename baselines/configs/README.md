# 配置计划

本轮没有可运行配置。后续每个方法的配置记录作者源码 commit、运行环境、数据导出规格、任务适配模式、划分清单、监督标签、调参预算和模型选择规则。

模型超参数先保留作者建议设置，再按训练/验证数据调整。依赖冲突的方法使用独立环境；旧 DGL/torch 或 TORAI 的 patched causal-learn 不直接合并到主工程 `pyproject.toml`。

`registry.json` 中 `pinned_commit: null` 表示尚未固定上游代码，不是可复现运行已准备就绪。
