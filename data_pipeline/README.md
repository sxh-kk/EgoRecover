# EE4D-Motion 错配数据处理

本目录独立于 `UEM-E7/`：下载官方处理数据，生成可复现的错配输入，读取连续观测，审计标签与时间索引。构建不需要 E7 checkpoint、SMPL-X 模型文件或 EgoExo4D 原始视频。仅修改绝对设备观测，原始身体标签保持不变。

## 目录与依赖

```text
data_pipeline/
  download_validation.py      # HTTP Range 下载、断点续传、CRC32/SHA256 校验
  wait_for_dataset.py         # 供另一工作对话等待完成信号
  ee4d_mismatch/
    source.py                 # 官方数据读取、抽样、标签引用
    corruptions.py            # 确定性错配算子
    build.py                  # 生成配对 manifest 和轻量 NPZ
    dataset.py                # 分离观测、监督和离线审计字段
    audit.py                  # 完整性、回放、时间与划分审计
  tests/
data/
  ee4d_motion_uniegomotion/    # 官方处理文件；生成器只读
  ee4d_mismatch_smoke_v0/     # 已构建：2 takes × 18 variants
  ee4d_mismatch_pilot_v0/     # 已构建：12 takes × 7 variants
  EE4D_MISMATCH_READY.json    # 下载、构建、测试、完整审计全部成功后才产生
```

在项目根目录运行。依赖 Python ≥3.10、NumPy、PyTorch；测试另需 pytest。当前环境已具备依赖，不需要安装 UEM 的训练依赖。

实际构建结果与验证时间见 `BUILD_REPORT.md`。`run_pilot.py` 可等待下载后连续构建、完整审计这两套数据；已有同配置产物会重新审计。它还使用 Matplotlib 绘制输入变化图，不会自行发布跨对话完成信号。

## 下载与构建

```bash
python3 -u data_pipeline/download_validation.py
python3 -m data_pipeline.ee4d_mismatch.build \
  --data-root data/ee4d_motion_uniegomotion \
  --output data/ee4d_mismatch_pilot_v0 \
  --num-takes 12 --frames 200 --profile pilot
python3 -m data_pipeline.ee4d_mismatch.audit \
  data/ee4d_mismatch_pilot_v0 \
  --report data/ee4d_mismatch_pilot_v0/audit/validation.json
```

下载器固定到 Stanford 的 `ee4d_motion_uniegomotion.zip` 当前已核对版本：原归档 35,067,467,786 bytes，通过 Range 只取 5 个成员，总压缩量 7,891,857,404 bytes，解压后 8,614,730,734 bytes。包括 `ee_val.pt`、`egoview_dinov2_val.pt`、`takes.json`、`annotations/splits.json`、`v4_beta_ee_train_stats.pt`。不下载原始视频、训练样本或模型权重。

下载状态在 `data/.ee4d_validation_download/status.json`，包含阶段、字节数、近期速率、预计结束时间和文件校验记录。同一数据目录只运行一个下载进程。下载中断后重新运行可续传；已有完整文件先通过长度与 ZIP CRC 校验再复用，不覆盖损坏文件。

生成目录必须不存在，修改配置时使用新版本目录。所有变体先在临时目录构建，通过审计后整体重命名。`--profile extended --num-takes 2` 可以覆盖全部算子。生成过程不复制大型 DINO 缓存，只保存原文件引用、轨迹和索引。

### 完整验证集

`--all-sequences` 已覆盖官方 `ee_val.pt` 中每一个原始序列，不再每个 take 抽取一条 200 帧片段。验证集为 560 个 take、5,236 个序列、932,525 个源序列帧；配对 7 种基础变体后实际生成 36,652 条记录。实际运行入口：

```bash
python3 -u -m data_pipeline.run_full
# 或仅构建，不发布完成信号：
python3 -m data_pipeline.ee4d_mismatch.build \
  --data-root data/ee4d_motion_uniegomotion \
  --output data/ee4d_mismatch_val_full_v1 --all-sequences
```

长序列按原长度保留，每个原始序列只生成一个故障事件；不复制视觉特征。长度不足 200 帧的 3,953 个序列仍全部进入数据集。为保证短序列也有可观察的错配，启动段缩为 `min(20, max(2, length//3))`，保留至多 10 帧故障前正常观测，30 帧事件按剩余长度截短；实际起点、终点、截短前标称时长、可达到的恢复期均在 manifest 中。实际有 1,794 个序列的 30 帧故障被截短，3,490 个序列无法保留完整 80 帧恢复期。`freeze_1s` 尽量保留 10 帧，在当前最短 21 帧序列里仍可完整生成。

完整集专属信号是 `data/EE4D_MISMATCH_FULL_READY.json`；运行状态在 `data/.ee4d_full_build/status.json`。旧的 `data/EE4D_MISMATCH_READY.json` 也已增加新产物。完整集消耗官方 val 中全部原始序列；用该集调参后，应准备其他独立数据做最终评估。详细结果见 `FULL_BUILD_REPORT.md`。

## Pilot 的具体定义

每个原始 take 抽取一个连续 200 帧片段，运动采样 10 FPS，采样跨度 19.9 s。按 `parent_task_name` 分层、take 内确定性抽样；不足长度或视觉缓存不对应的序列会记入 inventory。

前 20 帧为干净启动段，之后至少 10 帧正常输入才触发故障。故障起点随机，30 帧故障结束后至少保留 80 帧正常恢复观察。这里的启动段只定义数据范围；模型 rollout 不得因此使用 GT 作为预测历史。

| 变体 | 故障定义 |
| --- | --- |
| clean | 原始处理数据 |
| freeze_1s / freeze_3s | 视觉冻结 10 / 30 帧，锁存故障发生前最后一个可用观测 |
| delay_0p2s / delay_0p4s | 视觉输入延后 2 / 4 个运动时间步，持续 30 帧 |
| drift_0p01mps / drift_0p03mps | 头部绝对位置施加水平恒速偏移，持续 30 帧 |

漂移量为 `(t-onset)/10 × velocity`，首个故障采样点偏移为零，末点累计 2.9 秒；结束点立即恢复原始输入。误差率是设定值，不将离散窗口最后一点误记为完整 3 秒位移。

扩展集另有 0.3 s 视觉延迟、视觉缺失、轨迹缺失、轨迹延迟、yaw 漂移、轨迹抖动、单帧脉冲、阶跃、冻结叠加漂移、延迟后冻结、持续冻结，共 18 种。持续冻结保留干净启动段，事件一直持续至片段末尾。组合操作按 manifest 顺序作用于观测流，冻结可以锁存前序延迟后的视觉帧。

## 时间、坐标与信息边界

- `frame_id_30fps = sequence_start_frame30 + 3 × sequence_motion_index`；DINO 索引是 `frame_id_30fps // 3 // 2`，保持官方 5 FPS 缓存的绝对相位，不对每个小窗口重新对齐。
- 0.3 s 延迟先在 10 FPS 的观测流移动 3 步，再查 5 FPS 缓存；由于量化，它与原始视觉源的差值可能交替为 0.2 / 0.4 s。manifest 记录的是输入流延迟。
- `aria_traj_obs` 为 `[T,9]`，前 6 维是旋转矩阵前两行，后 3 维是世界坐标 xyz（米）。yaw 误差在世界坐标左乘旋转，位置不会随 yaw 被绕原点旋转。
- 原始特征若为 `[N,5,1024]`，只读取 token 0；返回 `[T,1024]`。缺失视觉为有限的全零向量、availability=false；缺失轨迹为单位旋转加零位置、availability=false。
- 缺失索引为 `-1`，读取器先按 availability 筛选，绝不会访问缓存最后一帧。冻结、延迟和漂移仍然 availability=true。
- `corruption_mask_gt` 的两列是 `[traj, img]`，表示设定的故障区间，不能作为 Q 的输入。首帧漂移为零、冻结偶遇重复帧时，故障标记也不会自动消失。
- `img_source_idx`、`traj_source_idx`、错配类型、参数、故障时间、GT 等仅用于离线构建和审计。模型接口只返回当前时刻及过去的观测和真实可用性。
- 身体 SMPL-X 参数、betas、kp3d、floor、body_root_offset 来自原片段，按物理时刻读取，所有变体共享相同标签摘要。Q 动作收益、P 预测、Flow 中间状态均不属于数据集标签。

## 给模型代码的读取接口

```python
from data_pipeline.ee4d_mismatch.dataset import MismatchDataset

dataset = MismatchDataset("data/ee4d_mismatch_pilot_v0")
observations = dataset.observations(0, as_of=35, history_frames=20)
# img_feats: [20,1024]; aria_traj_obs: [20,9]
# img_available/traj_available: [20] bool; frame_id_30fps: [20] int64
labels = dataset.supervision(0)     # 整个片段的物理监督；不得直接送入在线策略
audit_arrays = dataset.arrays(0)    # 离线用；含源索引与故障标签
```

从 `UEM-E7/` 内运行时，将项目根目录加入 Python 导入路径，如 `PYTHONPATH=/home/ld666/projects/EgoRecover`。`data_root=` 可覆盖 manifest 中记录的原始数据根目录，便于迁移；原始数据内容必须保持一致。

这不是 E7 的 18D 轨迹适配器。后续模型代码应从 9D 绝对轨迹生成所需条件，合法选择窗口参考、避免以被污染的头部参考重编码身体 GT。读取器不会创建 GT 启动历史或训练动作收益标签。

## 划分与审计

当前下载的是官方 val，因此产物统一标记为 `engineering`，用于验证数据管线与接口。如果使用这些 take 调参，最终结果必须排除它们；名单保存在 `split_manifest.json`。这批小样本不能证明方法精度提升。

正式开发集从官方 train 构建：`--base-split train --purpose development`，先按 take 划分 train/dev，再生成所有变体，禁止同一 take 跨集合。需要另行准备官方 train 文件；当前 7.9 GB 下载未包含它们。

`audit/validation.json` 包含哈希、完整算子回放、启动段、物理标签、配对关系、take 隔离、源索引、缺失值以及读取接口的验证结果。构建时校验全部变体但不重复计算大型源文件哈希；独立 audit 默认额外重新核对所有源文件的 SHA256。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest data_pipeline/tests -q -p no:cacheprovider
python3 -u data_pipeline/wait_for_dataset.py --timeout 7200
```

另一个对话必须先执行等待脚本或主动读取完成信号；文件本身不会唤醒一个已经结束的对话。脚本仅在信号存在、标记 complete 且所有产物通过完整审计后输出“数据集构建完成”并以 0 退出。超时或校验失败不会伪造完成消息。
