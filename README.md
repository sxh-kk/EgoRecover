# EgoRecover

基于 UniEgoMotion / E7 的因果人体运动恢复实验代码。模型使用过去 20 帧身体状态和当前视觉、轨迹观测，逐帧生成并提交身体预测，研究观测故障、预测历史适配与损失设计对闭环精度的影响。

## 代码内容

- 保留原始 dense E7 骨干和 Euler10 Flow 采样入口。
- 当前帧生成器 G、历史运动先验 P、Gaussian / History 两种采样源。
- 因果历史缓存、模型生成的公共启动状态、固定启动体型的 SMPL22 FK。
- 表示 MSE、世界 dense 关节位置损失及可选 SMPL FK 位置损失。
- EE4D-Motion 错配数据构建、固定 take 划分、配对评估与可恢复任务队列。
- 四阶段实验、预测历史 replay、四动作诊断及 G 损失消融。

当前结果属于开发集实验。原生 E7 的整段重建与改造后的逐帧闭环评估是不同协议；`frozen_e7` 结果标签指 E7 权重迁入历史接口后的冻结对照。最新进展、已知限制和数值见 [LOG.md](LOG.md)。

## 环境与外部资产

```bash
conda env create -f environment.yml
conda activate egorecover
```

`environment.yml` 使用 Python 3.11 和 PyTorch 2.10 / CUDA 12.8。其他 CUDA 环境需安装兼容的 PyTorch，再安装 `requirements.txt`、`requirements-dev.txt`；可视化另需 `requirements-vis.txt`。

需要自行准备：

1. EE4D-Motion 处理数据、DINOv2 特征及固定训练统计，见 [DATASET.md](DATASET.md)。
2. 获得授权的 SMPL-X 模型资产，放在 `body_models/smplx/`，或配置 `SMPLX_MODEL_PATH`。评估入口会检查模型资产能否重建数据集 GT。
3. 与 E7 Flow 架构匹配的 checkpoint；当前实验使用 `exp/e7/last.ckpt` 的 EMA。官方 diffusion 权重不能直接代替 E7 权重。
4. 对应的数据 READY 文件和启动缓存；生成及审计流程见 [数据管线文档](data_pipeline/README.md) 和 [开发接口](EGORECOVER.md)。

仓库包含代码、配置、文档和轻量验证报告。数据、checkpoint、SMPL-X 资产、运行缓存及 `exp/` 产物需在本地准备。

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
```

损失消融固定同一个已训练 P，四个 G 从相同 E7 EMA 初始化；默认 GPU4/5 对应 0/0 的 Gaussian/History，GPU6/7 对应 1/1。默认每个 G 2400 步，完成后自动共同评估。该调度入口依赖 `run/loss_ablation.py` 中指定的已完成 72-take/2400 P 和基线评估，请先准备依赖产物。完整实验配置和监控方式见 [LOG.md](LOG.md)。

原始 E7 的训练、采样与评估方式见 [E7.md](E7.md)。几何指标、历史来源、坐标约定和权重迁移接口见 [EGORECOVER.md](EGORECOVER.md)。设计文档见 [BLUE_PRINT.md](BLUE_PRINT.md)；历史论文草稿和报告不代表已完成的独立测试结论。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests data_pipeline/tests -q
```

测试覆盖因果读取、坐标编码、FK 梯度、数据划分、缓存一致性、评估恢复、GPU 调度和固定 P 的损失消融。原始 E7 路径的上游数值一致性记录见 [verification/upstream_e7_parity.json](verification/upstream_e7_parity.json)。该验证使用合成输入，不能替代真实数据精度评估。

## 来源

原方法为 [UniEgoMotion](https://github.com/chaitanya100100/UniEgoMotion)，Chaitanya Patel 等，ICCV 2025；E7 扩展来自 [sxh-kk/UEM-update](https://github.com/sxh-kk/UEM-update)。提取提交与来源记录见 [SOURCE.md](SOURCE.md)。独立 EgoRecover 仓库从当前代码快照开始，源项目历史仍可通过来源链接追溯。

保留第三方文件中的原许可说明。模型与数据资产需遵守各自的获取及使用条款；本仓库不重新授权这些资产。
