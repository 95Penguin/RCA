# 未来遥测辅助预测最小实验

结论：**NO-GO for the current forecast-auxiliary design**。

固定LTFP编码器协议、first-visible 1分钟标签、deployment五折、30 epoch、seed 42，
加入下一分钟服务遥测预测和0.2权重的masked MSE。

| 数据集 | LTFP AP | Forecast AP | ΔAP | LTFP Hit@1 | Forecast Hit@1 | ΔHit@1 | Forecast RMSE | Persistence RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RE2-OB | 0.621 | 0.598 | -0.023 | 0.417 | 0.523 | +0.106 | 2.539 | 2.438 |
| RE2-TT | 0.409 | 0.459 | +0.051 | 0.807 | 0.752 | -0.055 | 0.976 | 0.459 |

## 配对case-bootstrap

- RE2-OB：pooled ΔAP=-0.038，95% CI [-0.118, +0.045]；ΔHit@1=+0.137。
- RE2-TT：pooled ΔAP=-0.015，95% CI [-0.129, +0.102]；ΔHit@1=-0.057。

## 判定边界

只有当至少一个数据集五折平均AP提升0.03、另一个下降不超过0.01且RCA不退化，
才进入条件正常反事实rollout。遥测预测还必须与persistence forecast比较，
否则只能称为辅助正则化，不能声称已经学到更准确的未来系统状态。

