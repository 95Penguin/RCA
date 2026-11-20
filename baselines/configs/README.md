# 配置计划

当前通过命令行配置，入口见上级 `RUNNING.md` 与 `SPECIALISTS.md`，暂不额外维护一套重复的 YAML。每个运行目录的 `metrics.json` 和 `data_manifest.json` 记录作者 commit、超参数、任务模式和划分指纹。独立环境使用两份 requirements 锁；专用方法的完整锁兼容四框架。

模型超参数先保留作者建议设置，再按训练/验证数据调整。依赖冲突的方法使用独立环境；旧 DGL/torch 或 TORAI 的 patched causal-learn 不直接合并到主工程 `pyproject.toml`。

`registry.json` 中 `pinned_commit: null` 表示尚未固定上游代码，不是可复现运行已准备就绪。
