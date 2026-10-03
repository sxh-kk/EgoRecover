# G 损失消融：geometry/fk = 0/0 与 1/1

## 固定协议

- 仅改变 G 的损失权重；所有任务复用同一个冻结 P（72-take/2400，P 原训练权重为 geometry=1、fk=0）。这不是 P 与 G 同时改损失的消融。
- G 从 exp/e7/last.ckpt 的 EMA 独立初始化，未从已有微调 G 续训。
- 72-take 固定划分：48 train、12 dev、12 holdout；holdout 不参与。
- 每个 G 2400 步，batch=32，AdamW lr=8.5e-5、weight_decay=0.01；sigma=1，Euler10。
- 训练 seed=62，dev 选点 seed=1062，最终评估 seed=62；每 200 步以 clean dev 闭环 FK 选点。
- GT 身体历史训练，无 replay；四动作训练；在线评估固定 a11。
- 公共 E7 模型启动缓存、20 帧启动、18 秒预测后缀、固定 beta_boot、planar reference。

| GPU | G 损失 geometry/fk | source |
|---|---|---|
| 4 | 0/0 | Gaussian |
| 5 | 0/0 | History |
| 6 | 1/1 | Gaussian |
| 7 | 1/1 | History |

## 评估与报告

每个训练任务完成后在相同 GPU 自动评估 12 takes × 3 条件，共 36 条轨迹。主指标为 clean 世界 SMPL22 MPJPE 的 take 宏平均；辅助指标为 clean/freeze/drift 三条件宏平均。计算 1/1 减 0/0，以及两者分别减已有 1/0 的按 take 配对差值和 bootstrap 95% 区间。单训练 seed、开发集比较，不作为独立测试结论。

训练每 20 步记录 representation、weighted_dense、weighted_fk 与总损失。完成后生成 summary.json、results.md，并追加 LOG.md；失败记录日志，不将部分结果视为完成。

## 运行

独立控制器 run.loss_ablation；queue.json 保存状态，jobs/*.status.json 保存训练子进程和退出码。只使用 GPU4–7。与已有 LingBot 和四阶段队列共享 GPU，启动前连续两次检查至少 32 GiB 剩余显存；该阈值不是运行期硬限额。原四阶段队列继续运行。

监控：

```bash
tail -n 15 -F /gaozt-test1/sxh/EgoRecover/exp/egorecover_loss_ablation/g00_g11_v1/jobs/train_*.1.log
```
