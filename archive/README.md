# Archive

本目录只保存历史快照和旧实验输出，不参与当前训练、测试或结果汇总。

- `legacy_exports/`：早期交付目录的原样快照，仅用于追溯。
- `legacy_outputs/audits/`：曾散落在项目根目录的审计 JSON。
- `legacy_outputs/micross/`：早期 MicroSS 导出表、schema 和阈值审计。

当前代码入口只在 `scripts/`，当前模型只在 `src/models/`。新的训练产物应写入
`artifacts/`，不要继续向项目根目录写入 CSV 或 JSON。
