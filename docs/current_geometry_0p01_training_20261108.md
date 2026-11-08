# 当前帧几何权重0.01：W1/W3重新随机初始化训练

按用户要求停止lambda=1的两组训练，并完整保留旧目录 `exp/current-geometry-20261108/` 的日志、配置、源码和已保存检查点。W1/W3最后记录步数为2757/2845，最新完整 `resume.pt` 与 `best.pt` 均为2000步；文件哈希与停止信息保存在原目录 `stop-summary.json`，不能把最后日志步数当成可恢复的检查点步数。

新实验目录为 `exp/current-geometry-0p01-20261108/`。与旧运行逐字段比较，只把 `current_weight` 从1改为0.01；源码包逐文件哈希保持相同。输出和配置路径改用独立目录，原检查点不作为初始化来源。

| 设置 | 数值 |
|---|---|
| W1 / W3物理GPU | 0/1/2/3、4/5/6/7，各4卡 |
| 初始化 | 随机，seed62，显式from-scratch |
| 当前帧几何权重 / scale_m | 0.01 / 0.1 m |
| global / local / micro batch | 64 / 16 / 8 |
| 训练更新 / 每组queries | 12000 / 768000 |
| 学习率 / weight decay | 3e-5 / 0.01 |
| grad clip / EMA | 1.0 / 0.999 |
| 源 / sigma / 求解 | Gaussian / 1.0 / 一步 |
| 历史 / roll-in | 20 / 4帧 |
| feedback / 回写 | none / revise_recent=true |
| sampler参照 | 双逻辑rank，保持原query/tau/stateless-noise计划 |
| 周期验证 | 每1000步，同一128片段/128 takes，seed1062 |
| 最终验证 | 加载minival最优EMA，统一4379片段，前20帧不评分 |

没有新增warm-up、改变P、扩大batch或调整其他训练设置。记录完整参数的 `launch-plan.json` 与冻结 `source/` 位于新目录。新训练采用的候选权重来自 [梯度审计](geometry_gradient_audit_20261108.md)，尚不代表已经验证有效。

```bash
watch -n 5 /root/miniconda3/envs/egorecover/bin/python /root/workspace/songxuhua/exp/current-geometry-0p01-20261108/monitor.py
```

对比lambda=1时，必须使用相同步数、相同128片段验证；旧组提前停止，没有完整4379片段最终结果。对比Flow-only旧基线时，也须保留同评价mask与统计加权，不将四卡和双卡的dropout/浮点归约视为逐张量相同。
