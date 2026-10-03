# P 统一体型 FK 监督实验

日期：2026-09-29。状态：代码、12项测试和真实GPU验证通过；用户已要求启动正式训练。

## 实验问题与对照

验证：使预测姿态与GT姿态使用相同启动体型计算FK，能否改善运动预测，并减少原FK监督中姿态补偿体型差异的压力。

四组都使用常速度＋残差、GT历史、geometry=1、原FK损失=0，不加显式速度。网络仍为20帧输入、243D输出、256宽度、4层8头Transformer，输出层零初始化。

| 组别 | 新增统一体型FK权重 | GPU |
|---|---:|---:|
| control | 0 | 4 |
| pose01 | 0.1 | 5 |
| pose03 | 0.3 | 6 |
| pose10 | 1.0 | 7 |

每组seeds62/63/64，共12次训练，各卡依次执行同一权重的三个种子。每个seed的初始权重、目标帧采样和预算配对。对照组重新训练，并核对与已完成CV/FK=0实验的复现情况。

## 唯一训练改动

```text
L = 原表示损失 + 原dense位置损失 + lambda * L_same_shape
L_same_shape = mean(||FK(预测姿态, 预测根位置, beta_boot)
                       - FK(GT姿态, GT根位置, beta_boot)||²) / (0.1m)²
```

两边都使用模型启动体型beta_boot。目标侧无梯度，GT体型通道不参与该FK目标，原始GT关节位置不会被覆盖。训练不调用G，推理网络和输入不增加新组件。

实现：[新损失](../../egorecover/prior_same_shape.py)、[训练](../../run/train_prior_same_shape.py)、[队列与汇总](../../run/prior_same_shape_experiment.py)。旧实验代码与产物不覆盖。

## 固定协议

- 复用已审计48 train / 12 dev序列；12 holdout不参与本轮选模。
- 10Hz、20帧历史、预测未来100ms；训练目标t=40…199。
- 每组2400步，batch32，AdamW lr3e-4/weight_decay0.01，梯度裁剪1.0，dropout0.1。
- 每100步全dev评价，包含step0；按原始GT上的世界Body MPJPE选择checkpoint，相同分数保留更早checkpoint。
- 正式评价始终保留原始GT的Body MPJPE，另外报告PA-MPJPE、统一体型MPJPE、根位置、全局/局部旋转及dense位置误差。
- 纯P自反馈100–1000ms仅作辅助诊断，不作为本轮单独淘汰条件。
- 各组完成后报告三种子均值、标准差、逐seed差值。先对同一take的三个seed差值取平均，再按take配对bootstrap10000次，统计seed62。
- 候选按三种子平均正式MPJPE从非零权重中选择，并明确是否超过权重0和纯CV。只有新监督对应指标下降、正式MPJPE未改善时，不能宣称整体精度提升。
- dev用于checkpoint与损失权重选择；配对区间未做多重比较校正。holdout、G历史适配和接入G收益需后续独立验证。

## 验证与资源

12项测试已通过，覆盖权重0与旧损失严格相等、同姿态在相同体型下零误差、GT体型不影响新目标、目标侧无梯度、正式指标不被更改、精确断点恢复、权重身份和配对统计。

[真实GPU检查](../../verification/prior_same_shape_gpu_smoke.json)：batch32下进行新损失反传与正式指标计算，峰值PyTorch reserved约0.314GiB（不含CUDA上下文和其他进程）；确认新损失对旋转输出有非零梯度。

共享GPU4–7，启动要求连续两次空闲显存至少12GiB，给CUDA上下文和其他任务波动留余量。显存不足的卡等待，其他卡正常推进；不停止其他项目进程。

## 记录与监控

输出：`exp/egorecover_prior_same_shape/v1/`。保存plan.json、queue.json、controller.log、jobs和train/<case>_s<seed>/中的checkpoint、report.json、selection.json及dev预测。全部完成后自动生成RESULTS.md、summary.json和指标图。

[LOG.md](../../LOG.md)记录开始/异常摘要及比较结论，并链接实际结果文件；不记录每次checkpoint选点。

```bash
cd /gaozt-test1/sxh/EgoRecover
python -m json.tool exp/egorecover_prior_same_shape/v1/queue.json
tail -n 20 -F exp/egorecover_prior_same_shape/v1/controller.log
tail -n 5 -F exp/egorecover_prior_same_shape/v1/jobs/*_s62.1.log
```
