# E7 所需数据

E7 使用原 UniEgoMotion 的 processed EE4D-Motion，无需重新运行 raw EgoExo4D 视频预处理。

- [官方 processed 数据](https://downloads.cs.stanford.edu/simurgh/chpatel/ee4d_motion_uniegomotion.zip)
- [Hugging Face 镜像](https://huggingface.co/datasets/chaitanya100100/uniegomotion/tree/main)

设 `UEM_DATA_DIR` 为解压后包含 `uniegomotion/` 的目录，需要：

```text
uniegomotion/
  ee_train.pt
  ee_val.pt
  egoview_dinov2_train.pt
  egoview_dinov2_val.pt
  v4_beta_ee_train_stats.pt
  ee_val_gt_for_evaluation.pkl
```

归一化统计必须与 E7 checkpoint 对应。身体模型由 `SMPLX_MODEL_PATH` 指向的目录提供，其中必须有 `SMPLX_NEUTRAL.npz`；默认路径为 `body_models/smplx/`。

每个处理后序列以 `<take_name>___<start_frame>___<end_frame>` 命名；边界来自 30 fps，运动处理为 10 fps。数据加载器使用 80 帧窗口、20 帧步长，尾部补零并提供有效帧 mask。DINOv2 原始缓存为 5 fps，加载时对齐为 10 fps。

标准评估每 10 个数据窗口抽取一个，E7 默认只取前 256 个。未来的连续历史实验需另写 take 顺序遍历和新增帧评估；划分训练/校准/测试应按原始 take，而不是随机拆相邻窗口。

参考运动来自拟合 SMPL-X，不是无误差的传感器真值。原有数据解码 helper 为可视化保留了无预测时的参考值回填，不应直接作为在线策略的 prediction-only 接口。
