# EgoRecover

基于 UniEgoMotion / E7 的因果人体运动恢复实验代码。模型使用过去 20 帧身体状态和当前视觉、轨迹观测，逐帧生成并提交身体预测，研究观测故障、预测历史适配与损失设计对闭环精度的影响。

文档入口：[分类索引](docs/README.md) · [迁移与复现](docs/release/migration.md) · [实验结论](LOG.md)

## 代码内容

- 保留原始 dense E7 骨干和 Euler10 Flow 采样入口。
- 当前帧生成器 G、历史运动先验 P、Gaussian / History 两种采样源。
- 因果历史缓存、模型生成的公共启动状态、固定启动体型的 SMPL22 FK。
- 表示 MSE、世界 dense 关节位置损失及可选 SMPL FK 位置损失。
- EE4D-Motion 错配数据构建、固定 take 划分、配对评估与可恢复任务队列。
- 四阶段实验、预测历史 replay、四动作诊断及 G 损失消融。

当前P采用常速度残差、geometry=1/FK=0，不加显式速度；D组192 take、562片段、9600步×三seed的下一帧Body MPJPE为34.099mm，统一体型为14.054mm。接入G时使用适配后的校准P，不能用原D_s62 checkpoint直接替换。

G最新R0–R3观测长度×设备参考对照已完成三seed。21帧/预测参考R0最好：Body 131.913±2.465mm，80帧和设备参考未带来全身精度收益；共同口径原E7为115.018mm。结果来自重复使用的222take因果基准、已知地面、固定启动体型、开发集选中的best、Euler10和单draw1062，不能直接与论文整窗指标比较。新S0–S3几何残差×实际R0历史方案仅设计，尚未实现/训练。

原生E7的整段重建与逐帧闭环评估是不同协议。早期 `frozen_e7` 标签还可能指E7权重迁入历史接口的冻结对照，应按具体实验文档区分。完整数值、结论和限制见 [LOG.md](LOG.md)。

## 环境与外部资产

```bash
conda env create -f environment.yml
conda activate egorecover
```

`environment.yml` 使用 Python 3.11 和 PyTorch 2.10 / CUDA 12.8。其他 CUDA 环境需安装兼容的 PyTorch，再安装 `requirements.txt`、`requirements-dev.txt`；可视化另需 `requirements-vis.txt`。

需要自行准备：

1. EE4D-Motion 处理数据、DINOv2 特征及固定训练统计，见 [DATASET.md](docs/data/dataset.md)。
2. 获得授权的 SMPL-X 模型资产，放在 `body_models/smplx/`，或配置 `SMPLX_MODEL_PATH`。评估入口会检查模型资产能否重建数据集 GT。
3. 与 E7 Flow 架构匹配的 checkpoint；当前实验使用 `exp/e7/last.ckpt` 的 EMA。官方 diffusion 权重不能直接代替 E7 权重。
4. 对应的数据 READY 文件和启动缓存；生成及审计流程见 [数据管线文档](data_pipeline/README.md) 和 [开发接口](docs/design/interfaces.md)。

仓库包含当前代码、配置、文档、轻量验证和原路径下的实验结论/统计。大文件分别放在公有HF仓库：[模型与当前续训状态](https://huggingface.co/sxhkk/EgoRecover)、[处理数据、冻结缓存及最终预测](https://huggingface.co/datasets/sxhkk/EgoRecover-data)。SMPL-X资产未上传，需自行授权下载。

## 迁移与复现

发布状态见[发布状态](docs/release/publication.json)。当前最小 `g-smoke` 已上传，其余分组正在传输，完整分组请等状态标记完成后恢复。

从新机器克隆main并恢复相应数据，精确HF revision和每个文件SHA256见 [artifact索引](docs/release/artifacts.json)：

```bash
git clone https://github.com/sxh-kk/EgoRecover.git
cd EgoRecover
conda env create -f environment.yml
conda activate egorecover
pip install -r requirements-release.txt

# 最小检查先恢复一个固定test episode与R0 seed62；另需授权SMPL-X
python tools/restore_release.py --preset g-smoke
python -m run.reproduce_release --model G --group R0 --seed 62 --smoke --device cuda

# 先查看下载量；不会下载或启动训练
python tools/restore_release.py --preset g-evaluation --dry-run
# 恢复24个G对照best、校准P和222take评估缓存，建立共享目录软链接
python tools/restore_release.py --preset g-evaluation
# 将自行授权下载的SMPLX_NEUTRAL.npz放到body_models/smplx/，或设置SMPLX_MODEL_PATH
python -m run.reproduce_release --model G --group R0 --seed 62 --device cuda

# 独立P：恢复原/扩量缓存和各轮P权重，再评价D组
python tools/restore_release.py --preset p-evaluation
python -m run.reproduce_release --model P --seed 62 --device cuda
```

`core`仅下载当前R0三seed、配套校准P和统计量约1.096GiB，不含完整输入数据；`data`下载处理数据与已选缓存/预测；`training`增加原E7、当前续训状态和训练缓存；`all`恢复全部发布大文件。`--verify-only`检查已恢复文件大小和哈希。复现命令输出到新 `exp/reproduction/`，保留原实验报告。完整安装、历史入口及路径适配说明见 [迁移文档](docs/release/migration.md)。

## 实验入口

从仓库根目录运行：

```bash
# P/G 训练及闭环选模
python -m run.engineering_pilot --help

# 多模型共同 dev 配对评估
python -m run.evaluate_paired_development --help

# 阶段 2–4 调度（阶段 1 及已有预算产物见 LOG.md）
python -m run.complete_stages --help

# geometry/fk = 0/0 与 1/1 的 G 损失消融
python -m run.loss_ablation --help

# 历史独立P Two-Forward / FK四组入口；仅查看帮助，不自动启动训练
python -m run.prior_two_forward_experiment --help

# 已完成的G预训练与观测参考四组入口
python -m run.g_pretraining_experiment --help
python -m run.observation_reference_experiment --help
```

损失消融固定同一个已训练 P，四个 G 从相同 E7 EMA 初始化；默认 GPU4/5 对应 0/0 的 Gaussian/History，GPU6/7 对应 1/1。默认每个 G 2400 步，完成后自动共同评估。该调度入口依赖 `run/loss_ablation.py` 中指定的已完成 72-take/2400 P 和基线评估，请先准备依赖产物。完整实验配置和监控方式见 [LOG.md](LOG.md)。

早期P Two-Forward实验采用GT预热、在线生成候选历史、渐进替换及第二次前向反传，方法和完成结果见 [历史方案](docs/experiments/p-motionstreamer.md)。后续采用常速度残差/FK=0，并做了显式速度、统一体型监督、扩量和实际历史适应性实验；这些阶段的协议和结论见分类索引。

旧队列、GPU分配和等待逻辑对应历史实验状态，不代表克隆仓库后应立即启动的任务。使用新配置、新输出目录重跑；原始冻结配置和身份保持不变，异机精确续训尚未验证。旧100%固定G历史适配代码保留供历史复现。[Two-Forward运行说明](docs/experiments/p-two-forward-runbook.md)记录的是当时状态，最新事实以LOG为准。

原始 E7 的训练、采样与评估方式见 [E7.md](docs/reference/e7.md)。几何指标、历史来源、坐标约定和权重迁移接口见 [EGORECOVER.md](docs/design/interfaces.md)。设计文档见 [BLUE_PRINT.md](docs/design/blueprint.md)；历史论文草稿和报告不代表已完成的独立测试结论。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests data_pipeline/tests -q
```

测试覆盖因果读取、坐标编码、FK 梯度、数据划分、缓存一致性、评估恢复、GPU 调度和固定 P 的损失消融。原始 E7 路径的上游数值一致性记录见 [verification/upstream_e7_parity.json](verification/upstream_e7_parity.json)。该验证使用合成输入，不能替代真实数据精度评估。

## 来源

原方法为 [UniEgoMotion](https://github.com/chaitanya100100/UniEgoMotion)，Chaitanya Patel 等，ICCV 2025；E7 扩展来自 [sxh-kk/UEM-update](https://github.com/sxh-kk/UEM-update)。提取提交与来源记录见 [SOURCE.md](SOURCE.md)。独立 EgoRecover 仓库从当前代码快照开始，源项目历史仍可通过来源链接追溯。

保留第三方文件中的原许可说明。模型与数据资产需遵守各自的获取及使用条款；本仓库不重新授权这些资产。
