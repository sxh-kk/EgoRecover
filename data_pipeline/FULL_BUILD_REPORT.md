# 完整验证集错配数据构建结果

2026-09-23，北京时间 **15:14:18** 完成。产物：[ee4d_mismatch_val_full_v1](/home/ld666/projects/EgoRecover/data/ee4d_mismatch_val_full_v1)。源为已经下载和校验的官方 EE4D-Motion 处理版 `ee_val.pt`、`egoview_dinov2_val.pt` 等文件。

| 项目 | 结果 |
| --- | ---: |
| 官方验证集有标签 take | 560 / 560 |
| 原始运动序列 | 5,236 / 5,236 |
| 源序列帧数合计 | 932,525 |
| 每序列配对变体 | 7 |
| 错配数据记录 | 36,652 |
| 变体帧数合计 | 6,527,675 |
| 产物磁盘占用 | 约 428 MB |
| 自动测试 | 24 passed |
| 两轮逐记录审计 | 均通过 |
| 源文件 SHA256 与下载时记录 | 一致 |

每个原始序列完整保留为一个 episode，所有变体共享原序列的物理身体标签，视觉特征引用原始 DINO 缓存，没有在新数据里复制。清洁、1 秒/3 秒视觉冻结、0.2 秒/0.4 秒视觉延迟、0.01/0.03 m/s 水平漂移各有 5,236 条。

3,953 个序列短于 200 帧，仍全部纳入。它们使用可复查的自适应启动段和实际可容纳的事件长度：1,794 个序列的标称 30 帧事件被截短，3,490 个序列不足以保留 80 帧恢复期。实际长度、起点、终点和标称时长写在 manifest；审计将每条数据重放到相同结果。因此这些短序列不应直接当作固定 20 秒、3 秒故障和 8 秒恢复的实验片段。

主要证据：[审计报告](/home/ld666/projects/EgoRecover/data/ee4d_mismatch_val_full_v1/audit/validation.json)、[格式定义](/home/ld666/projects/EgoRecover/data/ee4d_mismatch_val_full_v1/spec.json)、[构建日志](/home/ld666/projects/EgoRecover/data/ee4d_mismatch_val_full_build.log)、[测试结果](/home/ld666/projects/EgoRecover/data_pipeline/test_results_full.xml)。审计逐条核对 NPZ 哈希、原样标签摘要、启动段、时间索引、无未来源索引、故障精确回放、配对关系、take 覆盖及在线读取字段；最终轮另核对全部官方源文件的 SHA256。

这是官方 val 的完整错配版本，可用于数据管线和模型工程验证。使用其 take 调参后，最终独立评估需要另留数据。模型接入使用 `MismatchDataset`；完整接口和可变长度注意事项见 [交接文档](/home/ld666/projects/EgoRecover/data/ee4d_mismatch_val_full_v1/HANDOFF.md)。
