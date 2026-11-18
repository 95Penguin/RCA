# State–Relation 第二阶段：事件复查与拓扑对照

## 结论

本轮没有获得真实调用拓扑稳定优于打乱拓扑的证据。上一轮 OB fold 3 的增益不能作为“Trace 建模传播并改善 RCA”的证明；真实图确实会影响这个固定模型的输出，但用错配图重新训练也能获得相近收益。

应暂停增加有向图、时滞注意力和跨粒度反馈，保留当前实现作为对照。这个结果不等于 Trace 没有价值，也不能否定服务级和系统级协同思想。

## 实验范围

只检查上一轮出现明显收益/退化的 OB folds 3、4。属于事后机制诊断，不是预注册的全数据集显著性检验。

- 冻结相同的 seed 42 LTFP checkpoint，系统预测路径不变。
- 重复真实图模型的事件级预测，逐事件比较 root rank 和首个可用标注窗口 root rank。
- 对已训练真实图模型，仅在推理时替换新增分支的图：自连接，以及 10 种固定节点置换图。
- 从头训练新增分支：训练 seed 42，真实图与 3 种错配图（graph seeds 2026、2027、2028）比较。
- 训练 seed 43 再比较真实图和 graph seed 2026 错配图，基线仍固定 seed 42。
- 全部重训 30 轮，相同学习率、minibatch 顺序、参数量和验证集事件 MRR 选择规则。没有用测试指标重新选 checkpoint。

“错配图”指 `A[p,p]`：只把邻接矩阵的服务索引置换，节点特征、服务标签及冻结基线均不变。它保留每张图的整体拓扑形状、权重集合与度分布，但破坏真实服务对应关系，并不代表所有随机图类型。所用三种置换与真实图的非对角边 Jaccard 约为 0.059–0.200，确实大幅改变了服务连接。

## 重新训练的事件级 MRR

| 模型 | sidecar seed | fold 3 | fold 4 |
| --- | ---: | ---: | ---: |
| 冻结 LTFP | — | 0.687500 | 0.850000 |
| Local 对照（上一轮） | 42 | 0.694444 | 0.850000 |
| 真实 Trace 图 | 42 | 0.857143 | 0.791667 |
| 错配图 2026 | 42 | 0.861111 | 0.858333 |
| 错配图 2027 | 42 | 0.851852 | 0.850000 |
| 错配图 2028 | 42 | 0.768519 | 0.850000 |
| 真实 Trace 图 | 43 | 0.777778 | 0.850000 |
| 错配图 2026 | 43 | 0.777778 | 0.850000 |

fold 3 的两个错配图能重现主要提升；换训练 seed 后，真实与错配图相同。fold 4 真实图 seed 42 的退化在错配图与另一个训练 seed 下均未重现。当前结构对训练轨迹和选模敏感，不能把收益归功于正确拓扑。

## 固定模型的推理替换

| seed 42 真实图训练模型 | fold 3 MRR | fold 4 MRR |
| --- | ---: | ---: |
| 原图推理 | 0.857143 | 0.791667 |
| 自连接图推理 | 0.783333 | 0.853333 |
| 10 个置换图推理均值 | 0.780714 | 0.806917 |

这说明固定模型使用了邻接信息，但替换输入属于分布扰动，不能单独用这个实验判定真实拓扑带来可泛化收益。必须结合上面的重训控制：依赖图输入，与依赖正确的服务关系，是两件事。

## 哪些事件发生变化

### fold 3：收益主要集中在两个事件

- `RE2-OB_recommendationservice_loss_1_event_47`：根因第 2 → 第 1。3 个错配图重训均能实现这个修正。
- `RE2-OB_recommendationservice_loss_1_event_48`：第 2 → 第 1。其中 2 个错配图也能实现。
- `RE2-OB_emailservice_delay_3_event_25`：第 8 → 第 7，仍然定位失败。

此折共 6 个事件，仅两个 recommendationservice 事件各从第 2 升到第 1，就贡献约 `(0.5+0.5)/6=0.1667` 的 MRR 改善，占主要收益。训练集中 email 根因事件为 0，验证有 1 个，测试也有 1 个，因此其困难还与服务覆盖有关。

### fold 4：一个 email 事件解释主要退化

- `RE2-OB_emailservice_mem_2_event_26`：第 1 → 第 3；前两名变为 currencyservice、productcatalogservice。相同模型用自连接图推理恢复第 1；3 个错配图重训均保持第 1。
- `RE2-OB_checkoutservice_loss_3_event_6`：第 6 → 第 4，小幅改善。

共 10 个事件，email 退化贡献 `(1/3-1)/10=-0.066667`，checkout 改善贡献 `(1/4-1/6)/10=0.008333`，相加正好为五折报告中该折的 `-0.058333`。没有证据证明前两名服务就是传播症状，只能说关系分支错误强化了它们。

此折验证根因事件只有 checkout（5 个）和 currency（13 个）；测试以 productcatalog（6 个）为主，并有 email、recommendation、checkout。验证集最优不一定代表这些测试服务最优，但不能用测试服务表现回调 checkpoint。

## 对提前定位的提醒

取每个事件**首个可用且有根因标签的窗口**进行一次排名，得到以下 MRR：

| 方法 | fold 3 | fold 4 |
| --- | ---: | ---: |
| LTFP | 0.437500 | 0.629091 |
| Local | 0.472222 | 0.645758 |
| State–Relation | 0.361111 | 0.703333 |

fold 3 的全事件聚合提升伴随首个窗口下降；fold 4 则相反。平均一个事件中的多个窗口会掩盖不同阶段的行为。这个首窗口指标仍不自动等于故障前预测：它取决于当前标签和可用窗口范围，不能直接替代严格的 onset / first-visible 协议。

## 下一步方法选择

优先验证“显式 metric 偏差在进入 TCN / LayerNorm 后是否被压缩，低维排序校准能否保留其增量”。建议做训练正常原型的直接旁路、使用原 RCA margin 与偏差的轻量修正器，并保留同容量的无偏差对照；所有参数与停止规则只按训练/验证数据制定。它先作为局部证据利用方式的诊断控制，而不是立即宣称论文创新。

若上述控制有效，再考虑服务局部证据如何提高系统预测 AP；若无收益，不能继续以添加门控或动态图挽救当前结构。当前结果既没有验证 AP 改善，也没有证实跨粒度协同已经建立。

## 代码与结果

- `scripts/run_state_relation_pilot.py`：增加 `--variants`、`--graph-control permuted`、`--graph-seed`；置换只作用于新增 RCA 分支。
- `scripts/audit_state_relation_cases.py`：重放 checkpoint、逐事件排名、首窗口排名、图替换以及训练/验证/测试根因覆盖统计。
- [完整数据与训练轨迹](../artifacts/state_relation_graph_audit_2026-11-18.json)。
- 57 项测试通过，新增测试确认图置换只影响 sidecar，系统预测不变，自连接对照不受置换影响。

复现示例：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B scripts/run_state_relation_pilot.py \
  --datasets ob --folds 3 4 --variants state-relation --seed 42 \
  --graph-control permuted --graph-seed 2026 --epochs 30 --device cpu \
  --output-dir /private/tmp/state_relation_new_permutation

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B scripts/audit_state_relation_cases.py \
  --checkpoint-dir /private/tmp/state_relation_cached_fivefold_20261118 \
  --control-dirs /private/tmp/state_relation_permuted_2026 \
  --output /private/tmp/state_relation_new_audit.json
```
