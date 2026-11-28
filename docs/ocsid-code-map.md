# OCSID / CPCA 代码地图

当前预测实现为 corrected-v2、归因为 corrected-v3-paths：已修复 Operation 轴排列、train-only 标准化、二值标记与串行守恒；
历史结果保留，不能与新实现混用。[当前四模块方法](2026-11-25-method-v4.md)固定当前叙述，
[原始审查](2026-11-25-code-review.md)用于追溯问题。

## 当前结构

```text
src/models/ocsid.py          预测模型；可选训练辅助根因头
src/models/ltfp.py           复用时序/图编码骨干
src/ocsid/
  __init__.py               公共 API；模型按需导入
  __main__.py               包命令分发
  paths.py                  checkout 数据根路径
  config.py                 当前uniform归因profile（审计与推理共用）
  aggregation.py            业务均匀/请求数加权/hard/soft共享汇聚
  evaluation.py             正式同cohort事件/窗口指标
  features.py               19/17 维特征及 operation 表读取
  operation_builder.py      原始 trace → operation 特征表
  training.py               窗口构建、标准化、五折训练、阈值/温度拟合
  cpca.py                   守恒树分解、反事实重放、线性责任分
  clocks.py                 可选有界时钟对齐（严格模式默认关闭）
  inference.py              无标签统一推理：风险触发 + CPCA 两种输出
  audit.py                  RE2 正常参考、hard/soft operation、对照与审计
  fusion.py                 历史学习头融合消融（需要旧 node_scores）
scripts/                    保留原四个薄入口，旧命令和部分 helper 导入兼容
```

## 入口

从仓库根目录运行：

```bash
PYTHONPATH=src .venv/bin/python -m ocsid --help
PYTHONPATH=src .venv/bin/python -m ocsid features --help
PYTHONPATH=src .venv/bin/python -m ocsid train --help
PYTHONPATH=src .venv/bin/python -m ocsid attribute --help
```

旧入口仍可用：`build_ocsid_operation_features.py`、`train_ocsid_joint.py`、
`audit_cpca_gate2.py`、`fuse_ocsid_cpca.py`。未移动其他 baseline、历史实验、原始数据、
结果 JSON，也未注册到不支持 operation 张量的通用 factory。

## 默认方法与历史消融

- 预测：服务时序/图 + operation 风险；默认训练含辅助根因监督。
- 辅助头只在训练态计算；eval 不输出辅助根因分数。checkpoint 仍可含辅助参数，
  不宣称已物理删除所有辅助权重。
- 定位：CPCA；hard/soft、q_o、rolling 参考由 audit 独立评估。
- `cpca_scores` 历史字段固定保存 hard 分数；新增 `scores_by_method` 保存各方法分数，
  防止把 hard 输出误当 soft。
- fusion 只适用于含 `node_scores` 的旧结果；当前 predictor 产物不含该字段。
- `train_ocsid_predictor.py` 是早期对照，仍有独立历史特征定义，不是正式入口。
- `summarize_ocsid_runs.py` 读取旧 v3/fusion 产物，属于历史汇总，非新统一评估器。

## 检查

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest tests.test_ocsid tests.test_ocsid_package tests.test_ocsid_contracts tests.test_ocsid_v2 -v
```

原有四个失败契约现已修复并去掉 expectedFailure；当前方法相关测试包括守恒、部分树、低流量和统一评价契约。
训练报告标记 corrected-v2，审计标记 corrected-v2-entry；显式时钟对齐时带 clock 标识。
新实验输出独立保存，正式报告包含 checkpoint、scaler、服务/operation 顺序及验证阈值。
当前profile采用入口子树、min1、replace/uniform及1ms量化界；底层树函数仍默认strict。
最终事件指标使用 `python -m ocsid evaluate`，不要引用审计内部的历史覆盖均值指标。
