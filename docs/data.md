# 数据、资产与坐标协议

统一入口：`egorecover restore`。资产清单随包放在 `egorecover/resources/assets.json.gz`，仅保留当前方案需要的原 E7 EMA、校准 P 和 paired cache；压缩索引由清理前冻结发布清单筛选，原文件内容、SHA256 和 HF revision 均未改变。

## 恢复

```bash
egorecover restore --root assets --preset training --dry-run
egorecover restore --root assets --preset training
egorecover restore --root assets --preset training --verify-only
```

| preset | 内容 |
|---|---|
| `metadata` | 仅 paired manifest，用于划分及空间统计 |
| `preflight` | manifest、校准 P、train/dev 各一个 episode 与完成标记；选择传输量较小的归档，不下载 test payload |
| `models` | E7 checkpoint 与匹配当前 manifest 的校准 P |
| `data` | paired manifest、全部 episode 与完成标记 |
| `training` | models + data，共 9,633 个文件，约 8.018 GiB；压缩/去重传输约 4.817 GiB |
| `smoke` | manifest、校准 P、原发布最小检查所需 episode；用于加载核验，不足以进行完整训练 |

`--dry-run` 不联网，也不创建目标目录。实际恢复在 HF 固定 revision 下读取原始路径，按新路径保存；分片先验证后拼接，归档只读取指定成员。每个目标核对大小和 SHA256，下载失败时不发布半成品，已有不一致文件拒绝覆盖。

空间报告分别列出解包后的逻辑大小、传输量和保留下载缓存的预估占用。完整 training 的逻辑大小约 8.018 GiB，保留下载与归档缓存时预计需要约 16.31 GiB。真实 manifest 含 train 4,369 / dev 224 / test 222 个 episode，每个 200 帧；train/dev/test 的 take 互斥。当前研究阶段只使用 EE4D-Motion，不恢复 Nymeria。

```text
assets/
  pretrained/e7.ckpt
  pretrained/prior.pt
  datasets/paired/
    data/manifest.json
    data/episodes/<episode_id>.pt
    data/episodes/<episode_id>.json
```

模型库 revision：`8b8190b6964f7c257346cbd65d31010a48007f28`；数据集 revision：`fdbff9f6a9856ff5b13e9818edf8677e395c6ee2`。外部资产仍受各自条款约束。SMPL-X 不在恢复清单内，需提供授权的 `SMPLX_NEUTRAL.npz`；默认目录为运行工作目录的 `body_models/smplx/`，或用 `SMPLX_MODEL_PATH` 指向其所在目录。

当前方法消费已处理的 EE4D-Motion 身体和 DINOv2 特征缓存，不包含从原始视频重新提取特征或构造历史故障数据的工具；这些旧工具可在归档分支恢复。

## 缓存格式

`--data-root` 指向 `assets/datasets/paired`。manifest 包含 `records`，每条记录至少提供：

- `episode_id`、`base_take_name`、`group`（train/dev/test）。take 不得跨划分。
- `base_seq_name`，格式为 `take___first30___last30`。
- `episode_start_motion_idx`、`num_frames`。
- `floor_height`，明确记录从原始世界 z 减去的高度，缺失/无效时拒绝加载。

每个 `.pt` 包含以下张量及身份信息：

| 字段 | 含义/形状 |
|---|---|
| `record` | 必须与 manifest 对应记录一致 |
| `stats` | 固定训练集的 motion_mean/std（243）、traj_mean/std（18） |
| `truth` | 离线监督 BodyState：joints `[T,22,4,4]`、reference `[T,4,4]`、auxiliary `[T,36]` |
| `teacher` | 同样结构的因果模型预测；只由它提供启动/训练初始历史 |
| `beta_boot` | 模型启动阶段估计的固定体型 `[10]` |
| `head` | 地面归零世界坐标的头部 rotation6d+xyz `[T,9]` |
| `images` | DINOv2 特征 `[T,1024]` |
| `head_valid` / `image_valid` | 可选布尔有效性 `[T]`；原发布缓存为全可用 |
| `frame_id_30fps` | 可选，若提供则与记录推导出的原始帧网格逐项核对 |

同名 `.json` 标记包含 `completed=true`、manifest SHA256 与 episode SHA256。加载器先验证身份、哈希、时间和所有状态/可用传感器载荷，再构造样本。一个数据集内以及 train/dev 间的统计必须一致，不重新拟合 dev/test 统计。

原帧网格为 30Hz，每 3 帧一个运动样本，因此 FPS=10。时间为原始帧号除以 30 的 float64 推导值，并非实测传感器时钟，不自动进行重采样。

## 在线边界与导出

常规在线重建只接收 `WindowCache` 和 `ObservationWindow`，不接收 truth、未来观测、故障标签或数据集全包。当前时刻每次只读取过去 20 帧加当前一帧的原始传感器范围；GT 仅供损失和重建结束后的评价。多步配对参考分支仅在起始时刻写入一次当前身体标签。

内部缓存位于地面归零世界坐标，窗口编码使用同一分支的公共参考。`--save-predictions` 输出：

- `joints_floor_world`：与缓存相同的地面归零世界坐标。
- `joints_world`：加回记录的地面高度后的原始世界坐标。
- `original_world_from_floor_world`：显式还原变换。
- `identity`：checkpoint、协议、数据、代码和评估选项。

正式几何解码双方使用同一启动体型。主 `body_mm` 对数据集原始身体 GT，`same_shape_mm` 对以启动体型重新解码的真值；两套指标分别报告，不把目标口径差异解释为纯体型误差。
