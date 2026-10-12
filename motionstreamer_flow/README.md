# MotionStreamer 风格的因果 Flow 模型

这是一份独立的 P+G 替代实现：**逐帧因果 TAE → 历史 Transformer → 观测融合 → x0 参数化 Flow 头 → Two-Forward 训练**。部署时由一个模型逐帧生成身体，不需要先运行 P 再运行 G。

代码放在 `motionstreamer_flow/`，不导入现有 `model/`、`mydiffusion/`、`egorecover/` 或 `run/`。可以只复制这个目录到另一台 GPU 机器，在其父目录执行命令。已有缓存只在离线转换时读取；模型、数据和日志写入指定的新目录。

这是根据项目任务改写的架构，**不是 MotionStreamer 官方复现**。新增模型尚未在真实数据上训练，也没有可以与原 P+G 比较的正式准确率。

## 1. 数据如何经过模型

```text
预测启动身体 0…19 + 固定 beta_boot
              │
              └─ 因果 TAE encoder → 启动姿态 latent

生成第 j 帧（j ≥ 20）：
过去 20 帧的 [姿态 latent, 根部位姿]
              └─ 历史 Transformer → 当前查询 token
过去及当前 21 帧图像特征、头部观测 + 可用性/动作开关
              └─ 观测 cross-attention 融合 → 条件 c_j
噪声 ε + 条件 c_j → x0 Flow 头，Euler 10 步
              └─ 当前 [姿态 latent, 根部位姿]
当前及过去姿态 latent → 因果 TAE decoder → 局部关节旋转、手部 PCA、接触
局部旋转 + 根部 + 启动体型的骨架 → FK → 22 个身体关节
              └─ 保存生成的 latent 和根部，供下一帧使用
```

身体历史到 `j-1`，观测到 `j`，均不读取未来帧。输出 latent 直接进入下一步历史，不把输出身体反复编码。训练和推理的历史窗口完全相同；Transformer 按查询取固定 20 帧窗口，避免多层局部注意力扩大实际历史范围。在线每帧只计算一次历史/观测条件，10 步采样复用这个条件。

TAE 使用 stride=1 的左填充卷积，默认 dilation 为 `[1,2,4]`，encoder/decoder 各有 19 帧感受野；没有时间降采样、双向注意力或跨时间归一化。训练重建时使用变分采样，Flow 标签与启动历史统一使用 FP32 posterior mean，减少目标噪声。Flow 阶段固定 TAE 权重；FK 损失仍可通过 decoder 向 Flow latent 传梯度。

### 身体表示

| 部分 | 维度 | 内容 |
|---|---:|---|
| TAE 输入姿态 | 152 | 21 个非根关节局部旋转 6D（126）＋左右手 PCA 各 12（24）＋足部接触（2） |
| 姿态 latent | 16 | TAE posterior mean；维度可配置 |
| 根部 | 9 | 根部旋转 6D（前 6）＋骨盆位置 XYZ（后 3） |
| Flow 状态 | 25 | 标准化姿态 latent 16＋标准化根部 9 |
| 固定体型 | 10 | `beta_boot`，作为条件；不逐帧生成体型 |
| 头部观测 | 18 | 固定启动锚点中的绝对位姿 9＋相对上一帧的运动 9 |

6D 旋转取矩阵前两**行**，与本项目/PyTorch3D 的约定一致。长度单位为米，时间默认 10 Hz。世界 Z 为竖直方向；平面锚点来自预测启动身体的第一帧 heading/XY，不使用 GT 参考系。根部保留显式 9D，不压入 TAE。

FK 使用离线导出的、由 `beta_boot` 决定的 SMPL-X 前 22 个关节骨架偏移。核心运行时不需要 SMPL-X 资产。导出器逐 episode 对真实 SMPL-X Layer 检查 FK，误差超过 `5e-5 m` 会终止。网格和完整手指关节不在本包输出范围，手部 PCA 参数保留在姿态中。

### 观测与未来的 Q

| action | 图像 | 头部 |
|---|---|---|
| `a11` | 开 | 开 |
| `a10` | 开 | 关 |
| `a01` | 关 | 开 |
| `a00` | 关 | 关 |

动作对当前查询的**整个观测窗口**生效，并与实际可用性取交集。缺失数据在投影前屏蔽；两个模态均缺失时保留 null token，避免空 attention。训练默认对两个模态各以 0.1 概率关闭，每条训练序列使用固定动作组合。不强行把生成身体对齐到可能错误的头部观测。本包提供动作接口，**尚未实现 Q 的训练或策略选择**。

## 2. x0 Flow 与 Two-Forward

令 `s0` 是真实身体通过固定 TAE 得到的 latent 与真实根部状态，训练构造

```text
sτ = (1-τ) s0 + τ ε，ε ~ N(0,I)
τ ~ Beta(1.5,1)，τ ≥ 0.001
Fθ(sτ, τ, c) → s0 的预测值
vθ(sτ,τ,c) = (sτ - Fθ(sτ,τ,c)) / τ
```

推理从 `τ=1` 的纯噪声积分到 `τ=0`，默认 Euler 10 步，不在 0 时调用速度场。训练直接监督 x0；这相当于对速度误差乘 `τ²` 权重，不能当作未加权 velocity matching。此设置继承项目的 x0 Flow 使用方式，保留 MotionStreamer 的历史条件与训练策略。

Two-Forward 每次更新分两遍：

1. 固定 TAE 生成真实标签，构建身体历史。训练窗口起点为 episode 起点时，前 20 帧来自预测启动身体；随机内部窗口的前缀来自 GT，仅用于训练。
2. 第一遍给**当前真实标签加噪**，输入模型做单次 x0 预测。第一遍历史不包含当前身体标签；输出停止梯度。它不是从纯噪声跑完整 10 步，也不是完整闭环 rollout。
3. 按 `γ(s)=0.5×[1-cos(πs/S)]`，在非启动部分选 `floor(γ×候选帧数)` 个共享时间位置，用第一遍预测替换身体历史。启动前缀和无效帧不替换，报告实际替换比例。
4. 第二遍重新计算历史条件，重新采样独立的 `τ,ε`，仍监督原始真实当前身体。物理损失的 decoder 使用**同一份混合过去＋第二遍当前预测**。

默认 Flow 损失为标准化 latent MSE＋`8×`标准化根部 MSE＋固定体型 FK 的关节坐标 MSE。TAE 使用标准化姿态重建 MSE＋`1e-4×KL`＋固定体型 FK 重建 MSE。FK 在 FP32 中计算。权重是实验初始配置，不能据此推断已达到有效精度。

训练中的 GT（ground truth，真实标签）加噪只属于监督过程；在线 API 没有当前 GT 输入。逐渐替换历史有助于缓解训练/推理差异，但不保证完全覆盖自由生成误差。短闭环 rollout 训练留作后续消融，当前代码没有实现这一项。

## 3. GPU 环境与默认配置

在代码父目录执行，例如 `/workspace/EgoRecover`。使用新的虚拟环境，不更改原项目环境：

```bash
python3 -m venv /workspace/msflow_venv
/workspace/msflow_venv/bin/pip install --upgrade pip
# 示例 CUDA wheel；按服务器驱动和 PyTorch 支持选择对应的官方 index。
/workspace/msflow_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
/workspace/msflow_venv/bin/pip install -r motionstreamer_flow/requirements.txt
/workspace/msflow_venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
```

默认单 GPU、CUDA、BF16。默认 Flow 约 1.957 亿参数，TAE 约 528 万参数。建议先在 24 GB 或更大显存 GPU 上检查实际峰值；当前 CPU 环境没有测过显存和吞吐。支持历史层 gradient checkpointing 和梯度累积。显存紧张时把 `batch_size=2, accumulation=1` 改为 `batch_size=1, accumulation=2`，保留每次更新 64 个 Flow 监督帧的预算。

不支持 BF16 的 GPU 可复制配置改为 `precision="fp16"`（使用 GradScaler）或 `"fp32"`。当前是单 GPU 训练入口，**没有 DDP/torchrun 支持**。默认请求 CUDA 时若不可用会报错；不会静默切到 CPU。

| 默认 Flow 设置 | 值 |
|---|---|
| 历史 Transformer | 12 层，宽 768，12 heads，RoPE/RMSNorm/QK normalization |
| 观测融合 | 2 层 cross-attention |
| Flow 头 | 9 个 AdaLN 残差块，宽 1792 |
| 训练窗口 | 20 帧前缀＋32 帧监督目标 |
| batch / accumulation | 2 / 1，64 个目标帧/更新 |
| 更新次数 | 24000，即 1,536,000 个 Flow 目标帧（完整有效窗口） |
| 采样 | Euler 10 步 |

TAE 为单独阶段，默认 12000 次更新，batch=8，窗口 52 帧全部参与重建。其训练计算需单独报告，不包含在 Flow 目标帧预算中。相同目标帧数不代表与旧 P+G 相同 FLOPs。

## 4. 转换现有缓存

源目录必须已经包含真实 `.pt` episode 及完成标记，只有 manifest 不能训练。转换需合法 SMPL-X neutral 资产；无需改旧缓存生产脚本。

```bash
/workspace/msflow_venv/bin/python -m motionstreamer_flow.export_egorecover \
  --source exp/egorecover_g_pretraining/v1/data \
  --output /workspace/msflow_runs/data_v1 \
  --config motionstreamer_flow/configs/gpu_flow.json \
  --smplx-model-path /path/to/smplx \
  --device cuda
```

`/path/to/smplx` 下应有 `SMPLX_NEUTRAL.npz`。转换检查源 manifest hash、完成标记及 episode hash，复制原 train/dev/test 的 take 划分，保存新缓存与资产来源记录。输出目录需为空，且不能包含源目录或位于源目录中。

源缓存已减去协议声明的 floor；转换器沿用该 floor，不重新从全序列找地面。它忽略 truth 的 reference，使用预测启动 reference，固定 `beta_boot`。GT 关节仅作为离线评价目标保存；FK 训练目标在同一个启动体型上重建，避免把体型误差混入旋转重建损失。正式比较前应确认原缓存 floor 与启动模型来源符合既定协议。

### 独立数据格式

无需原缓存时，也可自行生产 `manifest.json` 和纯 tensor `.pt` 文件：

```json
{
  "version": 1,
  "fps": 10,
  "backend": "smplx",
  "records": [
    {"id": "episode_1", "take": "take_1", "split": "train",
     "path": "episodes/episode_1.pt", "sha256": "文件的完整 SHA256"}
  ]
}
```

实际训练需 train/dev，正式交付需独立 test。相同 take 不得跨 split。每个 `.pt` 的字段如下，T 必须大于启动长度：

| 字段 | shape | 含义 |
|---|---|---|
| `pose`, `root` | `[T,152]`, `[T,9]` | 未标准化真实标签 |
| `joints` | `[T,22,3]` | 锚点坐标中的真实关节，米，离线评价用 |
| `images` | `[T,1024]` | 图像特征；本包不含图像特征提取网络 |
| `head` | `[T,18]` | 头部绝对/后向运动特征 |
| `valid` | `[T]` bool | 身体标签是否有效 |
| `image_valid`, `head_valid` | `[T]` bool | 观测是否实际可用 |
| `startup_pose`, `startup_root` | `[20,152]`, `[20,9]` | 上游启动预测，不得用 GT 启动正式评估 |
| `beta`, `offsets` | `[10]`, `[22,3]` | 固定启动体型与父子骨架偏移；root offset 为零 |
| `anchor` | `[4,4]` | world-from-fixed-startup-anchor |
| `startup_source` | string | 正式数据为 `predicted`，合成测试为 `synthetic_predicted` |

其他图像维度、历史长度、启动长度、latent 维度需要对应配置。身体缺失帧不参与训练/评价，启动身体必须完整。图像/头部不可用位置允许 NaN；可用位置必须有限。标准化统计只在 train 拟合，包括 TAE 冻结后的 latent 统计，不访问 dev/test。

## 5. 分阶段训练与恢复

```bash
# Stage 1：TAE，选择 dev 重建最好的 checkpoint。
/workspace/msflow_venv/bin/python -m motionstreamer_flow.train \
  --stage tae --config motionstreamer_flow/configs/gpu_tae.json \
  --manifest /workspace/msflow_runs/data_v1/manifest.json \
  --output /workspace/msflow_runs/seed62/tae --device cuda

# Stage 2：固定 TAE，训练历史/融合/Flow；按 dev 闭环 Body 误差选择。
/workspace/msflow_venv/bin/python -m motionstreamer_flow.train \
  --stage flow --config motionstreamer_flow/configs/gpu_flow.json \
  --manifest /workspace/msflow_runs/data_v1/manifest.json \
  --tae /workspace/msflow_runs/seed62/tae/best.pt \
  --output /workspace/msflow_runs/seed62/flow --device cuda

# 恢复时 config、数据、TAE、逻辑设备编号、dev-limit 必须一致。
/workspace/msflow_venv/bin/python -m motionstreamer_flow.train \
  --stage flow --config motionstreamer_flow/configs/gpu_flow.json \
  --manifest /workspace/msflow_runs/data_v1/manifest.json \
  --tae /workspace/msflow_runs/seed62/tae/best.pt \
  --output /workspace/msflow_runs/seed62/flow --device cuda \
  --resume /workspace/msflow_runs/seed62/flow/latest.pt
```

非空输出目录必须显式 resume，避免覆盖其他实验。checkpoint 包含模型、标准化、optimizer、GradScaler、随机数状态、数据与 TAE hash、配置和 dev 结果。`latest.pt` 用于恢复；`best.pt` 用于评价。默认每 500 步及最终步保存，突然中断会丢失距上次保存的更新。`--stop-after` 是绝对更新编号，可安全做短运行/恢复验证，不改变总步数或 Two-Forward 的调度长度。`--dev-limit` 仅供调试，正式选模应使用完整 dev。

## 6. 闭环预测与比较

```bash
# 与原协议对齐：先从预测启动历史闭环生成，再只统计 40…199（stop 右开）。
/workspace/msflow_venv/bin/python -m motionstreamer_flow.infer \
  --checkpoint /workspace/msflow_runs/seed62/flow/best.pt \
  --manifest /workspace/msflow_runs/data_v1/manifest.json \
  --split test --action a11 --seed 62 --start 40 --stop 200 \
  --output /workspace/msflow_runs/seed62/test_a11.json --device cuda

# 保存某个 episode 的逐帧预测，输出从第 20 帧开始。
/workspace/msflow_venv/bin/python -m motionstreamer_flow.infer \
  --checkpoint /workspace/msflow_runs/seed62/flow/best.pt \
  --manifest /workspace/msflow_runs/data_v1/manifest.json \
  --split test --episode EPISODE_ID --action a11 --seed 62 \
  --output /workspace/msflow_runs/seed62/prediction.pt --device cuda
```

逐帧 API 为 `StreamingSession`：初始化仅接收预测启动姿态/根部、启动观测、体型、骨架和锚点；`step(image, head, image_valid, head_valid, action=..., noise=...)` 生成一帧。输入已是特征，不是原始视频。四个动作共用同一个模型；相同 seed 使用相同逐帧噪声，便于配对比较。`online_inputs` 白名单阻止 GT 姿态、根部与关节流入 rollout。

评价报告 Body/Root/Head/PA Body（mm）与速度误差（mm/s），按有效帧加权，速度按有效相邻帧对加权。坐标使用固定锚点，刚体变换不改变这些距离指标。推理从 startup 开始闭环，延后评价起点不会改为 GT 历史。

建议比较：原 P+G、现有观测融合 G、新模型；保持 take 划分、20 帧预测启动、`beta_boot`、floor、图像特征、10 Hz 和评价区间相同。分别跑 seed 62/63/64 与四种动作。新增 TAE 训练成本、不同参数量和 Two-Forward 额外前向须单独记录。消融至少包括无 Two-Forward（新配置中设 `two_forward=false`）、仅图像/仅头部/无观测及 latent 16/32；本包没有替你运行这些正式实验。latent 32 需要重新训练相应 TAE。

## 7. 验证与当前交付边界

```bash
# GPU 小模型全流程；明确是合成数据，不产生真实准确率结论。
/workspace/msflow_venv/bin/python -m motionstreamer_flow.demo \
  --output /workspace/msflow_runs/synthetic_smoke --device cuda

# 关键属性测试；CUDA 可用时额外运行 BF16 前向/反传测试。
/workspace/msflow_venv/bin/python -m pytest motionstreamer_flow/tests -q

# 仅没有 GPU 的开发机上显式指定 CPU 功能验证。
python -m motionstreamer_flow.demo --output /tmp/msflow_smoke --device cpu
```

测试覆盖因果性、真实窗口与 streaming 一致、缺失/动作屏蔽、QK scale 梯度、x0 Euler、FK 梯度、Two-Forward 停梯度和固定启动前缀、冻结 decoder 的梯度通路、GT 与在线输入隔离、失败帧不提交历史、两阶段恢复一致、缓存 hash/split 检查及脱离旧仓库运行。

本次开发环境为 `torch 2.10.0+cpu`，无 CUDA 和真实 SMPL-X 资产。已完成 CPU 合成全流程与属性测试；GPU BF16 测试因缺少硬件跳过。真实缓存导出、真实资产 FK parity、GPU 显存/速度和真实精度仍需在 GPU 环境验证。合成指标只证明训练/推理链路可运行。

交付验证记录见 `verification.json`：新增测试 26 通过、1 项 CUDA 测试跳过；原有相关回归测试 46 通过。原有代码和训练入口没有被本实现修改。

文件入口：`model.py` 模型，`objective.py` 两阶段损失与 Two-Forward，`streaming.py` 在线生成，`train.py` GPU 训练，`infer.py` 预测/评价，`export_egorecover.py` 只读转换，`data.py` 独立数据与标准化，`geometry.py` 旋转/FK，`demo.py` 合成验证。

## 8. 参考与改写位置

- [MotionStreamer 论文](https://arxiv.org/abs/2503.15451) 与 [官方代码，固定 commit 8aace3f](https://github.com/zju3dv/MotionStreamer/tree/8aace3f6f5f564ae1518aa77d925ae1380070c80)：参考因果 TAE、历史 Transformer、条件生成头和 Two-Forward。这里重新实现，加入身体/根部分离、第一人称观测融合、动作屏蔽和固定启动体型。
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)：连续路径/速度场的基础。这里采用 x0 预测与相应速度场，以项目的线性噪声路径适配官方扩散头。
- [EgoAllo](https://arxiv.org/abs/2410.03665)：头部观测与身体运动分开建模的相关背景。这里使用固定合法启动锚点和后向相邻头部运动，不复用其完整表示或硬对齐。
