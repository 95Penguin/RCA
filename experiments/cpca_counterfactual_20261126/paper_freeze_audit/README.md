# OCSID–CPCA 定型审计与方法初稿

先读 **INNOVATION_DECISION.md**。结论：结构可定型写作，创新属于有限的建模/系统方法贡献，不包装为强因果学习；参考稳定，但 TT 对 partial 和时钟策略有依赖，最近邻合理适配后的差距较小。

## 文档与结果

- `METHOD_DRAFT.md`：当前实现对应的方法章；包括时间桶、账本定义、四个基本性质及解释边界。
- `RELATED_WORK_AUDIT.md`：Sage、ShapleyIQ、CRISP、CauseLens 的证据来源与贡献矩阵；标注全文核查限制。
- `AUDIT_RESULTS.md/json`：100 次健康分钟块重采样、全部 raw trace 覆盖、完整请求/不需时间修正的分层。
- `BASELINES_UNIFIED.md/json`：4 个框架、7 个 specialist、简单异常/频率先验、ShapleyIQ 的统一口径重算。
- `INNOVATION_DECISION.md`：值得保留、必须纠正、不可宣称与定型决定。

## 代码

- `audit_paths.py`：归档进仓库后自动解析根目录；外部运行可设 `NEXTMINUTE_RCA_ROOT`。
- `build_cache.py`：完整原始 trace 审计、正常分钟块参考、capped 请求评分缓存、100 次联合重采样。
- `summarize_audits.py`：原正式分数及请求计数精确核对、事件/窗口指标、覆盖分母和分层。
- `unify_baselines.py`：共同测试集合、显式 unavailable、合法负 logits、逐折标签与源文件核验；先验只拟合该折训练事件。
- `run_shapley_matched.py`：保留作者核心，补对齐输入及 operation 等权控制。
- `shapley_adapter_original.py`：保留之前的适配脚本作溯源，正式复现使用 wrapper。
- `third_party/shapley_value_rca.py`：作者公开核心副本，SHA-256 与此前冻结源一致，未改归因算法。
- `test_freeze_audit.py`：解析器合同、参考块加权中位数、服务内截零顺序、缺证据、负 logits 与 local 性质检查。
- `verify_freeze.py`：180 个案例缓存、评分范围、冻结分数、请求计数和作者核心源哈希核验；生成 `VERIFY_RESULTS.json` 与 `MANIFEST.json`。

大体积缓存、旧输入快照与逐窗结果由本目录 `.gitignore` 排除，但本地归档保留。核心汇总 JSON、代码与文档可纳入版本管理。
原 OCSID、CPCA、训练模型和旧正式结果没有被改写。

## 复现

在本目录用仓库 `.venv/bin/python -B` 运行脚本。原始 archive、服务面板及正式 cohort 位于仓库原有路径。

```bash
python -B build_cache.py --dataset ob --output-dir cache_ob --repetitions 100
python -B build_cache.py --dataset tt --output-dir cache_tt --repetitions 100
python -B summarize_audits.py
python -B run_shapley_matched.py --dataset ob --output shapley_matched_ob.json
python -B run_shapley_matched.py --dataset tt --output shapley_matched_tt.json
python -B unify_baselines.py
python -B -m unittest -v test_freeze_audit.py
python -B verify_freeze.py
```

为重建而非续跑，给缓存和 ShapleyIQ 输出指定新目录/文件。不要删除旧结果来伪装第一次试验。
bootstrap 范围是参考选择敏感性，不是因果修复置信区间。事件配对区间来自反复用于开发的数据，不是无选择偏差验证。
原数据集中未识别入口的 trace 不等价于真实业务请求被拒绝；不可据此推导无条件生产覆盖率。
