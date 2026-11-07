# 数据、资产与坐标协议

当前主实验使用作者官方 EE4D-Motion processed 发布包和官方 train/val 划分，G 随机初始化，P 使用解析常速度。Nymeria 留到后期。旧项目自定义 200 帧缓存及其下载缓存已删除，当前主路径不依赖 E7 或学习型 P checkpoint。

## 官方文件与入口

使用 [作者发布的 `ee4d_motion_uniegomotion.zip`](https://github.com/chaitanya100100/UniEgoMotion/blob/main/DATASET.md)，下载和解压命令见 [README](../README.md#正式训练)。`--data-root` 必须指向同时包含以下文件的目录，默认是 `datasets/ee4d_official/ee4d_motion_uniegomotion`：

```text
ee4d_motion_uniegomotion/
  takes.json
  annotations/splits.json
  uniegomotion/
    ee_train.pt
    ee_val.pt
    egoview_dinov2_train.pt
    egoview_dinov2_val.pt
    v4_beta_ee_train_stats.pt
    ee_val_gt_for_evaluation.pkl
```

约 10 GB 的 `ee4d_motion.zip` 仅提供逐 take 原始动作标注，不能代替上述 processed 包。官方文件由 `OfficialWindowData` 直接读取，并按请求切片加载；不重新生成项目旧 paired cache，不重新提取 DINO。SMPL-X 需使用用户授权取得的 `SMPLX_NEUTRAL.npz`，默认位于 `body_models/smplx/`，也可设置 `SMPLX_MODEL_PATH` 为其所在目录。

```bash
conda run -n egorecover --no-capture-output python -m egorecover prepare \
  --data-format official \
  --data-root datasets/ee4d_official/ee4d_motion_uniegomotion \
  --initialization scratch --prior-mode constant_velocity \
  --device cuda --output exp/official-preparation
```

`prepare` 检查本地文件、官方 take 划分与时间网格、统计、坐标往返和启动先验的 SMPL 几何一致性，生成配置、预算、依赖命令和 `preparation.json`；不自动下载数据或启动训练。大文件身份目前记录大小和修改时间，小型划分文件与统计记录 SHA256；这不是对所有官方大文件逐字节完成哈希核验的声明。

## 划分、片段与评分

官方 processed 数据含 2,032 个训练 take 和 560 个验证 take。加载器以 `takes.json` 和 `annotations/splits.json` 验证每个源序列的官方归属，不另拆项目 dev/test。CLI 的 `dev` 与 `val` 都指官方 val，官方模式拒绝 `test`，不能将开发集分数称为独立测试结果。

默认验证按官方源序列顺序，先以 20 帧步长枚举候选，再全局每 10 个候选取一个；若提供官方 `ee_val_gt_for_evaluation.pkl`，核对其有序键和片段长度。当前发布包实际产生 4,405 个参考候选，其中 26 个末尾片段只有 21 帧，不足以支持 20 帧启动及后续成对训练接口，故跳过；最终 4,379 个可用片段。大部分片段长 80 帧，末尾可以更短，跳过数量写入 inventory。

训练从官方 train 中按 take 均衡选择源序列与独立目标时刻，读取所需的身体历史、roll-in 和下一帧区间；不沿用旧项目“每 take 最多 4 个 200 帧片段”的筛选。训练统计固定使用 `v4_beta_ee_train_stats.pt`，不在 val 上重新拟合。

每个评价片段的前 20 帧只建立身体先验和上下文，不计分；随后模型逐帧输出并沿自身缓存连续推进。在线主结果使用每帧首次输出，内部对历史的后续修正不替换已经评分的结果。因此本工程保留官方数据与参考候选来源，但有明确的在线启动和评分边界，不能直接复制论文整段 80 帧结果来计算改进率。外部方法必须在相同片段、有效帧、输入和启动条件下重新评价。

架构验证可先使用 `evaluate --mode rollout --steps 1 --skip-paired`：保留连续重建指标与 CSV，只跳过额外成对诊断。反馈诊断要求锚点之后仍有完整的所选 1/2/4/10 帧，短尾片段与锚点应另外选择。

## 头部启动先验与身体范围

`HeadAlignedBootstrap` 只接收当时可见的头部位姿和授权 SMPL 中性骨架，不读取 GT 身体、GT 体型或 E7 输出。它使用固定 `beta=0` 的平均体型，采用头部局部 +X 在地面上的投影朝向，只取 yaw；通过 `R_heading @ Rx(+90°)` 将 SMPL 的 Y-up 中性骨架转换到世界 Z-up，并保持该 heading 轴一致。头部 pitch/roll 不会倾斜整个人体。

身体平移将 SMPL 头关节 15 对齐观测设备原点。这是缺少精确设备到头部外参时的近似先验，不是相机外参标定，也不代表从头轨迹已经恢复了真实姿态。每一帧的先验只用该帧头部观测构造；前 20 帧不是 GT 身体历史。兼容 `Episode.teacher` 字段在官方模式中存放这一中性先验，名字不表示仍有旧教师模型。常规运行在此基础上持续写入 G 自身输出。

当前只重建并评分 SMPL22 身体骨架，其中含左右手腕 20/21，不新增手指关节输出或 MPJPE-H。243 维表示保留原有辅助通道以兼容接口；官方手部 PCA 与 GT 体型仍属于离线标签载荷，不能进入启动体型或在线传感器条件。主几何解码始终固定使用 `beta=0`，不从当前或后续 GT 取体型。

## 坐标、时间与 DINO 对齐

源序列按 `take___first30___last30` 命名。运动来自原始 30 Hz 帧网格，每 3 帧一个样本，即 10 Hz；时间戳为对应原帧号除以 30 的 float64 推导值，不是另行测得的传感器时钟。

头部位置与 GT 身体位置共同减去官方序列提供的固定 `floor_height`，内部缓存统一在地面归零世界坐标保存。标签的局部旋转转成 22 个关节的世界旋转；原始手部 PCA 和体型标签保留，接触标签按既有速度/高度规则生成。零值地面按未校准哨兵处理，不用 GT 脚高度回填或重新估计；这类片段的地面相关指标排除并记录覆盖情况。已知地面的研究设定不等于完成了部署地面估计。

DINO 缓存为 5 Hz，每帧 5 个 1,024 维 token。当前取 token 0，转换为 float32；运动时刻对应的特征索引为 `floor(frame_id_30fps / 6)`。这相当于将最近已到达的 5 Hz 特征保持到 10 Hz 运动时刻，不插值、不读取下一个视觉特征。在线重建每步只读取过去 20 帧加当前帧的观测范围；GT 只供离线损失与评价。

## 预测导出与指标口径

`evaluate` 输出 `report.json`、`metrics.csv`；`--save-predictions` 另外保存：

- `joints_floor_world`：地面归零世界坐标的首次身体输出。
- `joints_world`：加回官方固定地面高度后的原始世界坐标。
- `body_rotations_world`、`timestamps`：世界旋转和对应评分时刻。
- `original_world_from_floor_world`：显式坐标还原变换。
- `identity`：checkpoint、数据格式、统计、启动、求解和评价设置。

主 `body_mm` / `mpjpe_world_mm` 对比官方拟合身体伪真值；`same_shape_mm` 对比用相同零体型重新解码的目标，两套分开报告，不把它们的差值直接解释为纯体型误差。世界误差、逐帧 PA 和整段相似对齐误差也分别命名。各字段、单位和论文差异见 [EE4D 对比协议](comparison.md)。

## 旧缓存兼容接口

`PreparedWindowData`、`--data-format paired` 及 `restore` presets 仅用于恢复历史流程。该格式要求项目自定义 manifest、episode `.pt/.json`、旧教师预测与绑定 manifest 的 P；其自定义 train/dev/test 不属于官方标准划分。当前官方训练不使用这些恢复入口，也不需要下载旧 paired cache、E7 或历史 P。旧格式细节与工具可在 [清理前归档](https://github.com/sxh-kk/EgoRecover/tree/archive/before-cleanup-20261106) 查询。

本文说明准备与评估接口，不表示正式训练或外部方法对比已经完成。
