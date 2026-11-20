# 适配器计划

已实现 `data.py` 与 `frameworks.py`，支持四个框架；`raw.py` 读取严格历史原始观测，`events.py` 拟合训练事件向量，`statistical_rca.py` 与 `learned_specialists.py` 复用十个专用算法。详见上级 `RUNNING.md` 和 `SPECIALISTS.md`。每个适配器负责：

1. 从共同 case/window split manifest 导出该作者实现要求的数据。
2. 调用原实现，处理命名空间或独立环境，不把所有模型强行搬进现有训练循环。
3. 显式配置原生预测、历史检测分数预测或未来分类头适配模式。
4. 将输出映射到公共时间键和服务候选，保留原始分数与原始输出。

如必须修改作者模型，单独记录补丁和原因，给予新的 implementation_id；不要悄悄替换原核心算法。具体时间和评价规则见上级 `INTEGRATION.md`。
