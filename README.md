# EgoRecover

按 [plan.docx](plan.docx) 实现的第一视角在线全身运动重建：20 帧世界坐标身体历史、滚动窗口联合修正、预测中心 Flow 和完整后续响应的成对反馈监督。

当前仓库只维护这一套实现。清理前的代码、实验脚本和结果保留在 [归档分支](https://github.com/sxh-kk/EgoRecover/tree/archive/before-cleanup-20261106)。恢复方法及本次删除范围见 [重构记录](docs/refactor.md)。

## 安装

使用 Conda 管理环境；当前环境配置为 Python 3.12 / PyTorch 2.10 / CUDA 12.6：

```bash
conda env create -f environment.yml
conda activate egorecover
```

运行入口统一为 `egorecover`，也可使用 `python -m egorecover`。核心运行依赖只有 PyTorch；SMPL-X 和资产下载依赖按需安装。

## 最小验证

```bash
egorecover train --config configs/paired.json --synthetic --compact \
  --output exp/smoke --train-steps 2 --batch-size 1 --eval-every 1

egorecover evaluate --checkpoint exp/smoke/best.pt --synthetic \
  --output exp/smoke-evaluation --steps 1 2 4 10 --save-predictions

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

合成模式显式标记为诊断，用小网络和 dense 关节位置验证流程与梯度，不代表真实人体精度。

## EE4D 对比指标

当前只做 SMPL22 身体重建，包含左右手腕，不增加手指输出或 MPJPE-H。`evaluate` 自动生成 `report.json` 和可用于表格对比的 `metrics.csv`：世界坐标 MPJPE、逐帧 PA-MPJPE、手腕/头部误差、旋转矩阵误差、根轨迹、上游脚滑指标及单帧延迟。整段相似对齐误差另列，避免混淆不同论文的 MPJPE。CSV 分别输出 take 等权和 episode 等权统计，在线主结果使用 `rollout` 行；`matched` 是内部诊断。

TMR Semantic/FID 尚未接入；jerk 以明确的诊断名称输出，尚不作为 EgoForce 原论文 PJ/AUJ。字段定义、单位、对齐方式和排除项写入评估身份；详细口径见 [对比协议](docs/comparison.md)。

## 正式训练

当前只使用 EE4D-Motion，Nymeria 留待后续扩展。2026-11-07 起，按用户要求改用作者官方发布数据；项目自己的 20 秒片段缓存及其下载缓存已经删除，不再自动恢复。G 使用随机初始化，不加载 E7。

使用官方 `ee4d_motion_uniegomotion.zip`（约 35 GB），包含过滤后的 SMPL-X 动作、DINOv2 特征、训练统计与官方划分。另一份 `ee4d_motion.zip`（约 10 GB）是逐 take 原始动作标注，不能代替完整训练包。参见 [作者下载说明](https://github.com/chaitanya100100/UniEgoMotion/blob/main/DATASET.md)。尚未取得官方包时，可以从项目根目录手动下载、解压：

```bash
mkdir -p datasets/ee4d_official
curl -L --fail --retry 5 -C - -o datasets/ee4d_official/ee4d_motion_uniegomotion.zip \
  https://downloads.cs.stanford.edu/simurgh/chpatel/ee4d_motion_uniegomotion.zip
unzip datasets/ee4d_official/ee4d_motion_uniegomotion.zip -d datasets/ee4d_official
```

主站较慢时，可从 [作者 Hugging Face 镜像](https://huggingface.co/datasets/chaitanya100100/uniegomotion/tree/main) 下载同名文件。下载和解压需要分别占用空间。

官方读取接口已接入，`--data-root` 指向包含 `uniegomotion/`、`takes.json` 和 `annotations/` 的解压目录。授权 SMPL-X 模型仍放在 `body_models/smplx/SMPLX_NEUTRAL.npz`。先生成配置并做本地训练前检查；`prepare` 不下载数据、不启动实验：

```bash
conda run -n egorecover --no-capture-output python -m egorecover prepare \
  --data-format official \
  --data-root datasets/ee4d_official/ee4d_motion_uniegomotion \
  --initialization scratch --prior-mode constant_velocity \
  --output exp/official-preparation --device cuda
```

报告位于 `exp/official-preparation/preparation.json`，各窗口配置位于其 `configs/architecture/`。当前启动历史是只由当时头轨迹构造的直立中性 SMPL22 先验：使用头部 yaw，固定 `beta=0`，将设备原点近似为头关节 15，不使用 GT 身体或 E7。P 采用解析常速度，当前阶段不需要 E7 或学习型 P checkpoint。图像仍读取官方预计算 DINO 特征；5 Hz 特征向 10 Hz 身体时刻作因果保持，不读取下一帧特征。

训练优化参数对齐 UniEgoMotion：AdamW 学习率 `3e-5`、weight decay `0.01`、betas `(0.9, 0.999)`、eps `1e-8`，L2 梯度裁剪 `1.0`，每次优化更新后 EMA `0.999`，有效 batch `64`，seed `62`。完整网络 dropout 为 `0.1`；合成诊断用的 compact 网络保持 dropout `0`。配置及实现来源固定为上游提交 `580c92c6d70a91672c4106bab30ca82cdd80f379`：[配置](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/config/uem.yaml)、[优化器与调度](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/module/uem_module.py)、[训练入口](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/run/train_uem.py)、[网络](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/model/uniegomotion.py)。

下面以 W=3 架构组为例，使用随机 G、一步求解和共同有效 batch；`2400` 次优化更新是 **pilot 预算**，不是官方的 350 epochs，也不表示已收敛。实验顺序为窗口 → 起点 → 成对监督 → 组合。W=1/3/5/10 使用各自配置和同一预算，先在官方 val 上选择窗口：

```bash
conda run -n egorecover --no-capture-output python -m egorecover train \
  --config exp/official-preparation/configs/architecture/W3.json \
  --data-format official \
  --data-root datasets/ee4d_official/ee4d_motion_uniegomotion \
  --from-scratch --prior-mode constant_velocity --device cuda --seed 62 \
  --learning-rate 3e-5 --weight-decay 0.01 --grad-clip-norm 1.0 \
  --lr-schedule uniegomotion --lr-decay-gamma 0.1 \
  --batch-size 64 --microbatch-size 8 --train-steps 2400 --eval-every 200 \
  --output exp/official-architecture/W3/seed62

conda run -n egorecover --no-capture-output python -m egorecover evaluate \
  --checkpoint exp/official-architecture/W3/seed62/best.pt \
  --data-format official \
  --data-root datasets/ee4d_official/ee4d_motion_uniegomotion \
  --prior-mode constant_velocity --split val --mode rollout --steps 1 \
  --skip-paired --device cuda --output exp/official-architecture/W3/seed62/evaluation
```

`--batch-size` 是每次优化的有效样本数，`--microbatch-size` 通过梯度累积控制单次并行量，默认分别为 `64/8`。`--resume` 保留同一实验的优化器与学习率时钟；后续反馈组使用 `--initialize` 从共同预热模型开始新实验，重置优化器和学习率时钟，两者不能混用。`--skip-paired` 只跳过成对反馈诊断，保留连续重建指标和 CSV；需要反馈诊断时另行选择有足够后续帧的片段与锚点。

默认 `--lr-schedule uniegomotion` 将官方 epoch 调度换算为 optimizer 更新次数：取官方训练候选数 `N_ref`，在 `ceil(N_ref / effective_batch) × 300` 次更新后把学习率乘以 `--lr-decay-gamma`，默认 `0.1`。当前 `N_ref=143560`、batch=64 时，衰减点为 `673200`，第 `673201` 次更新开始使用 `3e-6`。可用 `--lr-decay-step` 显式覆盖，或用 `--lr-schedule constant` 固定学习率；合成诊断没有官方候选参考，未显式设置衰减点时不自动衰减。

是否使用完整参考预算需单独决定。官方 350 epochs 对应 `2244 × 350 = 785400` 次更新；这里每个样本是 online query，不能把这些更新称为本项目的 350 epochs。示例 `2400 × 64 = 153600` 个 query 只用于预算试跑，学习率不会为了短预算而把官方衰减点压缩到其末段。确认学习曲线与计算成本后，再给各对照组设置共同的正式 query/update 预算；文档及 `prepare` 均不会自动启动大预算训练。详细换算见 [训练参数与预算](docs/method.md#训练参数与预算)。

官方包只有 train/val；`dev` 是 val 的别名，不能把它称为独立 test。默认验证选择当前发布包中的 4,405 个官方参考候选，跳过 26 个只有 21 帧的短尾，实际评价 4,379 个片段。每段前 20 帧只建立启动历史，不评分，因此这些在线结果不能直接与论文整段 80 帧数值相减。当前只输出并评价 SMPL22 身体，不增加手指重建；更多数据与评分边界见 [数据协议](docs/data.md)。旧 `restore` 和 `--data-format paired` 仅保留兼容用途，不是当前主路径。

## 代码结构

```text
egorecover/
  models/          窗口重建器、冻结历史预测器、E7 注意力组件
  codec.py         243维表示及世界/窗口坐标变换
  state.py         身体缓存与有界观测包
  system.py        历史→P→中心/源→Flow→缓存更新
  flow.py          x0训练路径及可微Euler求解
  losses.py        当前监督、成对/直接后续监督、固定度量对照
  data.py          固定划分、缓存校验和训练样本
  official.py      官方 EE4D train/val、特征与在线采样
  bootstrap.py     无 E7/GT 的头部对齐中性启动先验
  timing.py        源帧网格与采样时间核验
  training.py      训练、开发集选模、恢复
  evaluation.py    连续运行与配对多步指标
  metrics.py       SMPL22 精度、时序、轨迹与脚滑指标
  evaluate.py      评估任务组织及预测导出
  checkpoint.py    E7/P导入与新版checkpoint身份
  assets.py        固定版本资产恢复
  preparation.py   训练前审计、预算与任务生成
  official_preparation.py 官方数据检查及随机初始化任务生成
  cli.py           唯一命令行入口
configs/           基础和成对反馈配置
tests/             当前方法、几何、时间、恢复和入口测试
docs/              方法、数据协议、重构与复核说明
```

方法细节见 [方法与实验设计](docs/method.md)，资产和输入格式见 [数据协议](docs/data.md)，外部方法与指标口径见 [EE4D 对比协议](docs/comparison.md)，实现边界与验证见 [复核记录](docs/review.md)。

完整 G 默认保持 768 维、12 层主干；当前官方数据实验随机初始化 G，不加载 E7。E7/P 导入仅保留为兼容接口。新原生 checkpoint 为 `egorecover_window_v2`，不直接续训清理前的 v1 诊断 checkpoint。

当前已验证合成流程、梯度边界、精确恢复、权重迁移及中性启动先验的几何一致性。官方数据检查记录由 `prepare` 生成；本文命令是实验入口，不表示正式训练已完成，尚无对外精度或 GPU 延迟收益结论。

代码来源及保留的第三方声明见 [SOURCE.md](SOURCE.md)。
