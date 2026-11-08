# 当前帧几何监督：W1/W3 四卡训练协议

两组从随机初始化开始，使用已有当前帧几何监督，不继承窗口选择阶段的权重。当前项为固定零 beta 下预测姿态与 GT 姿态的 SMPL22 世界位置平方误差，按 0.1 m 归一化；正式评分仍对官方原始关节位置。

| 设置 | 本轮数值 |
|---|---|
| W1 GPU | 0/1/2/3 |
| W3 GPU | 4/5/6/7 |
| 每组优化更新 | 12000 |
| global / local / micro batch | 64 / 16 / 8 |
| 每组累计 online queries | 768000 |
| 初始化种子 | 62 |
| 当前帧监督权重 / 几何尺度 | 1.0 / 0.1 m |
| 源 / 求解 | Gaussian，sigma=1，一步 |
| 后续反馈 | none |
| 历史 / roll-in | 20 / 4帧 |
| 学习率 / weight decay | 3e-5 / 0.01 |
| 梯度裁剪 / EMA | 1.0 / 0.999 |
| 验证 | 每1000步，同一128片段/128 takes，seed1062 |
| 最终验证 | 加载minival最优EMA，统一4379片段的首次输出，前20帧不计分 |

本轮仍使用 UniEgoMotion 更新次数参考学习率时钟，12000步内不触发衰减；不称为350 epochs。原来的同窗口 Flow-only 结果保留作为参照，不能把本轮权重覆盖到原实验目录。

## 四卡与原双卡采样的对齐

`--sample-reference-world-size 2` 保留原双卡的逻辑抽样计划。每个逻辑 rank 使用 seed+1+logical_rank，逐8样本microbatch按原顺序抽取query、tau和端点mask；四个物理rank分别保留其连续16样本，其余计划仅消耗CPU随机数，不加载身体或图像数据。全局noise offset仍为0至63。

该机制保持原双卡每步查询、tau/端点和stateless噪声计划；不保证不同卡数下的dropout、浮点归约或优化轨迹逐张量一致。各组数据仍为官方take-balanced采样，不引入GT身体历史或GT体型。需要恢复本轮时，保留原world size、此采样参数和冻结源码。

启动前已检查：独立旧双卡采样循环与新四卡计划连续三步的全局query/tau及生成器状态完全一致；真实两组各四GPU三步中，Flow/current损失、梯度和同步hash均正常。每组初始G哈希与原基线相同：`455e73dfb7181c99fff554ed9b3e974bb897292ab69880932e9b6f42789d8fb6`。

## 运行与监控

运行产物位于 `exp/current-geometry-20261108/`，完整参数保存在 `launch-plan.json`，冻结源码位于 `source/`，两组进度分别在 `W1/progress.json` 和 `W3/progress.json`。原始预检报告位于 `exp/current-geometry-preflight-20261108/reference-sampling/preflight-result.json`。

```bash
watch -n 5 /root/miniconda3/envs/egorecover/bin/python /root/workspace/songxuhua/exp/current-geometry-20261108/monitor.py
```

监督器保存进程状态、错误和最终全验证对照表。完整 `resume.pt` 保存优化器、调度器和每个物理rank的RNG；`best.pt` 用于评估。运行状态以产物目录的 `status.json` 为准，不以本文件是否存在作为已经完成训练的证据。
