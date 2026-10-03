# P D组：扩大训练数据与预算

日期：2026-09-29。状态：已完成，Body 34.099±0.025mm、PA 10.552mm、统一体型14.054mm；[最终结果](../../exp/egorecover_prior_data_budget/v1/RESULTS.md)。用户决定只做D组，并允许制作更多take。当前阶段见[队列](../../exp/egorecover_prior_data_budget/v1/queue.json)。

冻结清单实际规模：192个train take，562个不重叠20秒片段，187.33分钟，为原16分钟的11.71倍。每take的片段数量：39个take有1段、32个有2段、25个有3段、96个有4段。全部数据审计和三个种子训练均已完成，正式审计见数据report。

## 配置

- 常速度＋残差，geometry=1，原FK与统一体型FK均为0，不加显式速度，不改P架构。
- 从原48个train take扩至192个，保留原48个片段；新增144个take按任务类别分层选择，只允许官方train，排除现有dev/holdout。
- 每take最多4个不重叠200帧片段（10Hz，20秒）；按绝对时间检查跨源片段重叠。长度不足时保留实际片段数，不复制充数。冻结清单后不按实验效果调整数据。
- 先等概率抽take，再抽片段和目标帧40–199；模型仍只看目标之前20帧GT。此处40帧取样边界沿用旧训练协议，不启用Two-Forward。
- 原take沿用原E7启动体型/地面高度；新take只用一个片段前20帧传感器，经冻结E7 EMA生成启动体型和地面高度，在该take内共享。训练目标为真实下一帧；未用G预测历史训练P。
- 每个新增片段通过SMPL标注一致性审计；原12个dev的数据、顺序、体型必须与原缓存逐张量一致；12个holdout及官方val/test不进入训练。
- 三个种子62/63/64，各9600步；batch32，AdamW lr3e-4/wd0.01，梯度裁剪1，dropout0.1，恒定学习率。
- GPU4–7：先准备缓存，再按可用显存调度三个种子；单张卡同时最多一个本队列任务。每100步评估固定dev，以原始GT Body MPJPE选模。

## 评价与结论边界

比较旧48take/2400步CV/FK=0与D组三种子Body、PA、统一体型、根位置和旋转误差，报告逐seed差值与按take配对区间。初始化必须与旧同seed逐张量一致。

额外报告2400/4800/9600固定预算点和最终模型，避免只看更多选点后的最优值。纯P一秒自反馈仅作辅助诊断。训练loss可在完整曲线查看，本轮不额外安排train诊断评估。

本轮同时增加take数、片段数与更新次数，只检验组合收益，无法分别归因。开发集参与多轮设计选择，最终收益仍需独立holdout及接入G验证。35/15mm仅为项目工程目标。

## 文件与运行

- 入口：[队列](../../run/prior_expanded_experiment.py)、[数据准备](../../run/prepare_prior_expanded.py)、[训练](../../run/train_prior_expanded.py)。
- [计划](../../exp/egorecover_prior_data_budget/v1/plan.json)、[冻结片段清单](../../exp/egorecover_prior_data_budget/v1/data/manifest.json)、[数据审计报告](../../exp/egorecover_prior_data_budget/v1/data/report.json)。
- 可恢复的逐take缓存位于 `exp/egorecover_prior_data_budget/v1/data_shards/`，每个缓存有哈希与绑定清单；异常不静默跳过。
- [结果](../../exp/egorecover_prior_data_budget/v1/RESULTS.md)、[完整统计](../../exp/egorecover_prior_data_budget/v1/summary.json)在三个种子完成后自动生成，同时把结论写入[LOG](../../LOG.md)。不向LOG写逐checkpoint选点记录。

```bash
cd /gaozt-test1/sxh/EgoRecover
/root/miniconda3/envs/egorecover/bin/python -u -m run.prior_expanded_experiment --output exp/egorecover_prior_data_budget/v1
```

重启控制器会接管仍在运行的子进程；任务失败后检查日志，使用同一命令加 `--retry-failed` 从已审计数据分片或训练checkpoint恢复。

## 实现验证

新增6项检查通过：跨源片段时间去重、官方train与保留集隔离、take均衡采样、固定dev防篡改、断点恢复参数逐位一致、固定预算及配对统计汇总。原常速度残差7项测试也通过。实际数据审计进度以队列和数据报告为准。
