# 代码来源与精简记录

- 上游仓库：https://github.com/sxh-kk/UEM-update
- 提取提交：`156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08`
- 原始实验配置：`ablation/configs/e7_x0_global_w8_u84k.yaml`
- 原始方法：UniEgoMotion，ICCV 2025，Chaitanya Patel 等。
- 官方完整身体网络参考：`chaitanya100100/UniEgoMotion`，提交 `580c92c6d70a91672c4106bab30ca82cdd80f379`。

网络精简为完整身体 Transformer decoder 与 E7 连续时间嵌入。为兼容上游 E7/EMA 权重，保留参数名称和顺序、未使用的 text 兼容参数及其冻结设置。工作树不再包含 MoE、稀疏关节、TaskFiLM、双输出头和 Diffusion 模型。

保留 E7 的加权 x0 目标、随机条件屏蔽、Euler 采样、Beta 时间分布、训练配置和 EMA。新增推理时可选历史约束与显式初始噪声参数，开关默认关闭。约束的具体含义见主 README；本次未证明它能改善真实运动恢复精度。

数据转换、几何、指标及可视化工具继承原作者代码。`module/ema.py` 等文件的第三方许可头保留；本次整理未为上游代码、SMPL-X 或数据资产授予新的许可。

原 UEM-update 工作区保留上游 Git 历史。独立 `sxh-kk/EgoRecover` 仓库以当前代码快照建立首次提交，不携带旧历史中的数据或实验轨迹；上游版本可通过上述仓库与提交追溯。

后续新增 EgoRecover 的历史条件 G、P/Q 接口、坐标与缓存、错配数据管线、闭环评估、四阶段调度及损失消融。当前实验和限制以 `LOG.md` 为准；原始 E7 路径说明保存在 `docs/reference/e7.md`。
