# 独立 P 改进实验：运行说明

当前状态（2026-09-29）：用户明确使用 GPU0–3 启动 P 训练。60 条身体序列准备与审计已完成，四组训练在 0–3 共享运行；G 消融评估保持停止。

更新：2026-09-29。状态：用户于 2026-09-29 12:07 要求停止 G 消融评估、立即启动 P 实验，当前已解除对应等待。方法见 [设计文档](p-motionstreamer.md)，事实记录见 [LOG.md](../../LOG.md)。

## 任务与启动条件

调度入口：`run/prior_two_forward_experiment.py`。输出：`exp/egorecover_prior_two_forward/v1/`。

默认模式等待下面两套队列的总状态和全部任务均为 `complete`：

- `exp/egorecover_stages_1_4/continuation_v1/queue.json`
- `exp/egorecover_loss_ablation/g00_g11_v1/queue.json`

本次用户明确取消 G 消融评估，使用 `--skip-loss-ablation-wait`，仅保留已完成的四阶段依赖。G 队列标记 `stopped_by_user`，不会被伪记为完成。默认模式下，依赖缺失、失败或仍在运行都会继续等待；不等待 LingBot。本轮只使用 GPU0–3，各卡可用显存至少 32 GiB，连续两次检查通过才启动。

| 第一轮任务（seed62） | GPU | 历史 | geometry / fk |
|---|---:|---|---|
| A_gt_dense | 0 | GT | 1 / 0 |
| B_two_forward_dense | 1 | 当前 P 的渐进 Two-Forward 混合 | 1 / 0 |
| C_gt_fk | 2 | GT | 1 / 1 |
| D_two_forward_fk | 3 | 当前 P 的渐进 Two-Forward 混合 | 1 / 1 |

先准备一次经过 SMPL-X GT 审计的 48 train / 12 dev 身体序列缓存，不缓存 holdout。四组相同初始化、目标帧 t=40…199、batch32、2400 更新；每 100 步按完整 GT dev 下一帧 FK 选模，step0 也参与。P 的结构和 20 帧部署接口不变，独立训练不调用 G。

完成第一轮后，从 B/C/D 中按主指标选出最好的候选，再用 seed63、64 分别重跑 A 和候选。即使候选首轮不优于 A，也完成复核并如实报告差值，不自动把它接入 G。

## 启动与监控

以下命令从仓库根目录执行。调度器有进程锁，已在运行时无需重复启动。

```bash
cd /gaozt-test1/sxh/EgoRecover
conda activate egorecover
python -u -m run.prior_two_forward_experiment \
  --output exp/egorecover_prior_two_forward/v1 --skip-loss-ablation-wait --gpus 0 1 2 3
```

查看状态和总日志：

```bash
cd /gaozt-test1/sxh/EgoRecover
python -m json.tool exp/egorecover_prior_two_forward/v1/queue.json
tail -n 40 -F exp/egorecover_prior_two_forward/v1/controller.log
```

任务启动后查看四组训练日志：

```bash
cd /gaozt-test1/sxh/EgoRecover
find exp/egorecover_prior_two_forward/v1/jobs -name '*.log' -print
# 根据上一条输出选择文件；第一次尝试的 A 组为：
tail -n 20 -F exp/egorecover_prior_two_forward/v1/jobs/A_gt_dense_s62.1.log
```

查看各训练任务最新进度：

```bash
cd /gaozt-test1/sxh/EgoRecover
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('exp/egorecover_prior_two_forward/v1/train').glob('*/progress.json')):
    print(path.parent.name, json.loads(path.read_text()))
PY
```

失败后先排查对应 `jobs/*.log`，修复后用原调度命令加 `--retry-failed`。训练每次选点评估后保存模型、优化器与全部随机状态；有 `resume.pt` 时继续，有不完整输出且无恢复点时先归档。数据、关键代码或训练协议改变会拒绝恢复，应新建实验版本。

## 结果与判读

- `data/sequences.pt`、`data/report.json`：共享 GT 序列、来源哈希、逐 take 资产审计。
- `train/<配置>_s<seed>/`：`initial.pt`、`prior.pt`、`resume.pt`、`selection.json`、`progress.json`、`report.json`。
- `dev_*.pt`：逐帧单步结果、自反馈预测骨架及 GT，可重新汇总/画图。
- `screening.json`：第一轮及复核候选；`summary.json`、`RESULTS.md`、`figures/`：最终汇总和图表。
- `LOG.md`：保留简短开始/异常记录，重点记录实验结论和可点击结果链接；不记录每次选点。完整选点曲线保存在 `selection.json`，完成时汇总下一帧、自反馈、消融贡献和种子复核结果。

主指标为 20 帧 GT 历史后的 100ms 世界 SMPL22 FK MPJPE，按 take 宏平均。另报 t≥40 匹配子集、根部、去根平移、旋转和 dense 误差。保持、常速度、原 P 均按相同 dev 输入评估；原 P 只作历史参照。

短时自反馈固定每个 dev take 的 t=20、40、…、180 共 9 个起点，合计 108 个起点；输入 20 帧 GT，随后 10 帧只反馈自己的预测，报告 100/200/400/800/1000ms。固定 t=80 的所有 12 个 dev take 用于可视化，避免只展示有利片段。

配对区间以 take 为单位计算，seed62 用于候选筛选，seed63/64 用于复核；dev 被用于选模，不能当独立测试。记录有监督样本数、额外候选预测数、前向调用次数及排除评估的同步训练墙钟耗时；共享 GPU 上耗时受其他任务影响。Two-Forward 计算开销更高，目前没有新的精度结论。
