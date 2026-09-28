# EgoRecover 工程实验产物

本分支保存新接口、测试、实验报告和逐帧**模型预测**，方便核对已经完成的修正。完整阶段记录见 [LOG.md](LOG.md)，接口与复跑说明见 [EGORECOVER.md](EGORECOVER.md)。数值来自官方验证数据的工程划分及随机 E7 初始化器；包括 dense22 与真实 SMPL-X FK 两类指标，仍不是正式独立测试结果。

新增 [SMPL-X 预测轨迹预检](verification/smplx_prediction_preflight.json)：12 条、2160 帧的预测状态恢复与原 dense22 报告一致。模型到位后完成真实 55 关节 GT 审计及全部 12 条人体指标，见 [SMPL-X 汇总](verification/smplx_geometry_summary.json)、[take0 完整报告](exp/egorecover_closed_loop_v3_take0/smplx_geometry.json)、[take1 完整报告](exp/egorecover_closed_loop_v3_take1/smplx_geometry.json)。模型文件本身不上传；复跑命令见 [评估说明](EGORECOVER.md#离线-smpl-x-几何评估)。

## 已上传的报告

- [修正汇总 JSON](verification/egorecover_repair_summary.json)、[PNG](verification/egorecover_repair_summary.png)、[PDF](verification/egorecover_repair_summary.pdf)：参考系修正、物理选模、预测历史适配及 20 条闭环回放的对照。
- [原因排查 JSON](verification/egorecover_cause_analysis.json)、[PNG](verification/egorecover_cause_analysis.png)、[PDF](verification/egorecover_cause_analysis.pdf)：冻结权重的参考系干预、单步误差分解和闭环归因。
- [工程实验汇总](verification/egorecover_engineering_summary.json)、[测试结果](verification/egorecover_tests.xml)、[全量数据适配检查](verification/mismatch_adapter_planar.json)。
- `exp/` 内原有 27 个 JSON 报告及两份 SMPL-X 完整报告按原路径保留，包括 v0–v3 训练、预测历史缓存、原因排查、闭环回放和身体保持/常速度基线。每个闭环目录中的 `.pt` 是模型生成的逐帧预测输出；原标签只在预测结束后用于误差统计。

## 本分支中的二进制范围

20 个闭环 `.pt` 预测结果合计约 8 MB，可用项目环境的 `torch.load(path, map_location="cpu", weights_only=True)` 阅读。`exp/` 在原仓库的 `.gitignore` 中；这些小型报告和预测文件是显式加入版本控制的。原始 EE4D-Motion、生成的错配数据集、预测历史训练缓存和模型 checkpoint 保留在本地。模型 G checkpoint 单文件约 369 MB，未纳入普通 Git 提交；报告中保留了它们的 SHA256、配置、训练划分和选中步数。克隆本分支即可阅读报告与代码，复跑模型仍需独立准备数据与权重。

报告中的 `/home/ld666/projects/EgoRecover/...` 是原实验机器的绝对路径，用于记录数据来源，不是仓库内可直接复用的路径。`run.report_repairs` 依赖本地完整 `exp/` 与数据环境，已生成的汇总文件可直接查看。
