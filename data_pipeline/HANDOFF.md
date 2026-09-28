# 给模型实现对话的接口交接

完整验证集已构建并通过两轮完整审计（2026-09-23 北京时间 15:14）。专属完成信号为 `data/EE4D_MISMATCH_FULL_READY.json`，详细交接见完整集目录内的 `HANDOFF.md`。原 `data/EE4D_MISMATCH_READY.json` 也已纳入完整集。

- 完整集：`/home/ld666/projects/EgoRecover/data/ee4d_mismatch_val_full_v1`，560 takes、5,236 个原始序列、36,652 个变体。每条保留原序列长度，范围 21–5,462 帧。模型如果要求固定 200 帧，须按合法时间顺序分块或填充。
- 完整集的 `engineering.jsonl` 与 `audit/validation.json` 已通过源文件 SHA256 和逐变体回放。24 项自动测试通过；完成信号包含全部路径。

等待专属信号：

```bash
python3 -u /home/ld666/projects/EgoRecover/data_pipeline/wait_for_dataset.py --signal /home/ld666/projects/EgoRecover/data/EE4D_MISMATCH_FULL_READY.json
```

下面两套原有小样本集仍保留用于快速检查：

- 主小样本集：`/home/ld666/projects/EgoRecover/data/ee4d_mismatch_pilot_v0`，12 takes × 7 variants = 84 条，每条 200 帧。
- 扩展检查集：`/home/ld666/projects/EgoRecover/data/ee4d_mismatch_smoke_v0`，2 takes × 18 variants = 36 条，每条 200 帧。
- 原始数据根目录：`/home/ld666/projects/EgoRecover/data/ee4d_motion_uniegomotion`。
- 22 项自动测试通过；两套数据的源文件 SHA256、标签摘要、每个变体的数组和算子回放全部通过审计。详细证据见 `BUILD_REPORT.md` 与各数据集的 `audit/validation.json`。

```bash
python3 -u /home/ld666/projects/EgoRecover/data_pipeline/wait_for_dataset.py --timeout 7200
```

收到“数据集构建完成”且命令退出码为 0 后，读取输出中的 dataset path 和 `audit/validation.json`。数据管线独立在 `data_pipeline/`，不依赖 `UEM-E7/` 中正在实现的模型。该对话可继续修改模型，本数据构建工作不会修改这些模型文件。

入口：`data_pipeline.ee4d_mismatch.dataset.MismatchDataset`。详细用法见同目录 `README.md`。

- `observations(index_or_variant_id, as_of=t, history_frames=n)` 返回截止 t 的视觉特征 `[n,1024]`、绝对轨迹 `[n,9]`、真实模态可用性和原始 30 FPS frame id；开头不足 n 帧时返回已有历史。
- 轨迹 9D 排列为 rotation6d（旋转矩阵前两行）+ xyz 米；运动 10 FPS、DINO 5 FPS，已经完成绝对时间映射。
- `supervision(...)` 单独读取不变的真实身体标签。仅训练/离线指标可用；不得当作 P 历史或 Q 在线输入。
- `arrays(...)` 包含故障类型关联的源索引、故障 mask 等审计数据，不应把这个字典直接传给策略。
- 冻结/延迟/漂移仍可用；只有实际缺失才 availability=false。缺失轨迹不是全零 6D 旋转，而是单位旋转+零位置。
- 18D E7 条件转换、P 的预测历史、Q 的动作收益生成和 G 的连续 rollout 是模型层工作，本目录不生成这些内容。
- 所有样本为官方 val 的工程检查样本；不能用它们宣称独立测试性能。`split_manifest.json` 记录抽取 take。
