# EgoRecover 开发接口与工程验证

实现依据 EgoRecover 设计文档。逐阶段实验与问题见 [`LOG.md`](LOG.md)。原 E7 网络、注意力块、Flow 求解器与训练入口保持原样；新增代码从独立入口使用。

当前已实现历史条件 G、当前帧 Flow、P/Q 网络、四动作、E7/EMA 初始化、几何 codec、物理缓存、在线回放、批量每样本启动体型的可微 SMPL22 FK 损失、公共启动缓存、预测历史混合训练入口和经真实模型验证的离线 SMPL-X 几何评估。正式独立训练评估、SMPL22 FK 收益标签与 Q 训练仍需训练好的 E7 启动权重及正式训练集。工程 checkpoint 不是正式模型。

## Conda 环境

只使用 WSL `zyc111` 中的 `egorecover`：

```bash
conda activate egorecover
cd /home/ld666/projects/EgoRecover/UEM-E7
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

当前实际环境：Python 3.11，PyTorch 2.10.0+cu128，Lightning 2.4.0，NumPy 1.26.4；RTX PRO 6000 Blackwell。

从仓库根目录重建环境：`conda env create -f environment.yml`。完整安装版本记录在 `verification/egorecover_pip_freeze.txt`。CUDA wheel 对应 [PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/)。`.venv` 是此前临时环境，后续命令不使用它。

环境局部设置 `PYTHONNOUSERSITE=1`、空 `PYTHONPATH`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，避免系统 ROS Python 3.10 的路径/pytest 插件污染。系统 ROS 未修改。`conda run` 提示覆盖 PYTHONPATH 属于预期隔离行为。

## G 的张量接口

```python
from config.defaults import get_cfg_defaults
from model.history_uniegomotion import HistoryUniEgoMotion
from egorecover.conditioning import build_conditioning
from egorecover.history_flow import HistoryFlow

model = HistoryUniEgoMotion(get_cfg_defaults(), history_length=20).to(device)
flow = HistoryFlow(source_mode="history", sigma=0.3)
y = build_conditioning(
    history_motion=history,         # [B,20,243]，当前公共参考下编码/归一化
    history_valid=history_valid,    # [B,20] bool
    prior_mu=mu,                   # [B,1,243]，冻结 P 或物理先验
    traj=trajectory,               # [B,1,18]，当前实际观测
    img_embs=features,              # [B,1,1024]
    img_available=img_available,   # [B,1] bool，实际可用性
    traj_available=traj_available, # [B,1] bool
    action="a11",
)
model.train()
losses = flow.training_losses(model, target_current, y, epsilon=epsilon)
losses["loss"].mean().backward()
model.eval()
prediction = flow.sample(model, y, epsilon=epsilon)  # [B,1,243]，默认 Euler10
```

`epsilon` 始终是标准噪声，不能传已加 μ/乘 σ 的 source；可以改传显式 `torch.Generator`。配对比较必须共享 epsilon。两种源都接收相同 H/μ；Gaussian=`σ·ε`，History=`μ+σ·ε`。

Flow 状态只有当前帧。G 内部拼接历史、保留 E7 全局时间 token，增加零初始化的状态、帧时间和 μ 加法支路，再只返回当前帧。Flow 的 loss mask 为 `[B,1]`；注意力 mask 内部另建。新增参数 779,520。

所有浮点输入应与 history 的 dtype/device 一致；mask 接受 bool 或严格 0/1。缺失、padding 和被主动屏蔽的 NaN/Inf 在投影前清除；未屏蔽的非有限值报错。G 拒绝 GT、故障日志、过去原始观测、repaint、CFG 等额外字段。

四动作顺序固定为 a11/a10/a01/a00，第一位视觉，第二位轨迹。最终屏蔽=动作屏蔽 OR 实际缺失。`action_equivalence()` 为等价动作选择最早代表，保留 a11 的零收益基准。

## 权重迁移

```python
from egorecover.checkpoint import load_e7_weights
report = load_e7_weights(model, "/linux/path/to/e7.ckpt", weight_source="ema")
```

支持原始 UniEgoMotion state_dict 或 Lightning 的 `model.*` state_dict。原模型 strict 加载后，才按原可训练参数顺序应用已知 v1 EMA，再映射新 G。缺少旧参数、未知 EMA 格式或长度/形状错误都会拒绝。新支路归零；旧 optimizer/EMA 不恢复。继续训练新 G 时使用新 G 自己的 strict state_dict。

没有真实 E7 checkpoint 时只能运行随机初始化工程实验。官方 diffusion checkpoint 不能替代本分支的 E7 权重。

## 坐标、历史与 P/Q

- `MotionCodec` 使用官方固定训练统计。cache 保存 dense 世界关节变换、内部 reference、手/接触/beta；换窗重新编码。当前目标的增量为 `inv(previous_predicted_reference) @ target_reference`。
- 默认 `reference_mode="planar"`：启动、目标重编码、当前解码和缓存 reference 始终为 heading + 水平 x/y。身体关节和相机仍是三维姿态。`legacy_se3` 仅用于重现旧的自由三维参考行为；Flow 中间噪声不做平面投影。
- 当前轨迹的 18D 后半段也连接此前提交的 reference。对齐的干净历史下与 E7 的相邻 reference 一致；预测漂移下形成需要训练适配的新输入分布。
- `HistoryBuffer.from_bootstrap()` 只接模型生成的 20 帧启动结果；`decode_candidate()` 不写缓存；`commit()` 只接一帧预测，严格检查连续时间。β_boot 在启动后固定保存，正式 FK 解码需显式使用它；当前 dense 解码不做 SMPL FK。
- `FixedShapeFK(smpl, beta_boot)` 接收单个 `[10]` 或批量 `[B,10]` 启动体型，在预测骨盆与旋转上可微地运行 SMPL-X。可输出 55 关节与 mesh 顶点；已在真实模型和两条开发视频的保存预测上运行。正式 FK 损失要求每行体型来自模型生成的干净前缀，拒绝旧 GT 启动缓存；FK 输出不替换 dense 历史状态。
- `HistoryPrior` 只读身体历史，在 codec 的物理保持先验上预测修正；`freeze()` 同时关闭梯度和 dropout，并抵抗父 module 的 `.train()`。
- 常速度基线在世界坐标外推位置，在 SO(3) 外推身体/参考旋转；手 PCA、contact、beta 保持末帧。
- `UtilityPredictor` 读取身体历史、μ、截至当前的观测和可选合法兼容性；在 G 前输出三个相对收益。`generate_utility_labels()` 仅离线调用，传入 a11 可用性条件和共享噪声；真实收益的 error_fn 应为公共坐标 SMPL22 FK 误差。
- 在线 `rollout.py` 不导入监督编码模块，读当前/过去观测，先决策再采样，仅提交一个输出。离线 `annotations.py`/`engineering.py` 可以读取标签与 clean counterpart，两条路径不能混用。
- 地面采用显式启动估计：前 20 帧实际头高的中位数减训练统计的平均头高。它不读取 GT floor，但实测并不总准确。离线 contact 标签可单独用注释地面，不影响生成器坐标或输入。
- codec 只解码合法 6D rotation；float32 norm 溢出时用 float64 重算同一投影，零轴/共线等真实退化仍报错。没有 GT fallback、任意裁剪或静默身体重置。

## 数据交接与验证命令

已取得 `data/EE4D_MISMATCH_READY.json` 完成信号。`egorecover.data.open_dataset()` 复核状态、审计、spec/manifest 哈希后调用独立 `MismatchDataset`；不会写入数据产物。交接信号现含两个 `pilot` 条目，本工程按固定 split manifest 的 spec SHA256 精确选择原 pilot 数据，不会因新增完整验证集而悄悄切换；要使用新 spec 须显式传 `spec_sha256` 并建立相应新划分。

```bash
python -m pytest tests -q --junitxml=verification/egorecover_tests.xml
python -m run.smoke_egorecover --device cuda --output verification/egorecover_interface_smoke.json
python -m run.check_mismatch_adapter --output verification/mismatch_adapter.json
```

当前 66 项测试通过，包含完整 12 层 G 的反传、四动作隔离、迁移、几何往返、连续 200 次提交的参考平面不变、身体三维姿态保留、FK 姿态梯度与有限差分、每样本体型隔离、公共启动复用、缓存划分防泄漏、P 冻结、Q 接口及严格因果读取。全部 120 个变体、24,000 个变体帧的坐标适配在新默认 codec 下也通过，见 `verification/mismatch_adapter_planar.json`；这些不是重建精度测试。

## 可复跑的小样本工程实验

仅用于已声明的官方-val 工程 take；`config/egorecover_pilot_split_v1.json` 固定 8 train / 2 dev / 2 holdout，不再随训练 seed 改组；holdout 不用于选择。全部变体随原 take。`--training-seed` 与 `--sampling-seed` 独立。GT 历史实验的结果不能标成闭环效果。

```bash
python -u -m run.engineering_pilot --output exp/my_engineering_run \
  --prior-steps 200 --flow-steps 1000 --batch-size 32 --sigma 0.3
python -m run.compare_engineering_baselines --experiment exp/my_engineering_run
```

默认随机初始化；有真实 E7 权重时额外指定 `--checkpoint ... --weight-source model|ema`。每个 output 必须是新目录，保留旧实验。保存 P、两种 G、配置/划分/哈希/曲线、四动作 dense 指标诊断。`utility_diagnostic_*.pt` 明确不是正式 Q 训练集。

当前默认损失为原加权表示 MSE + dense 世界关节平方距离 / `(0.1 m)^2`，通过 `--geometry-weight` 控制。P 按完整 clean-dev 位置误差选模，未优于物理保持时保留第 0 步零修正；G 每 200 步按相同开发帧/噪声的 Euler10 位置误差选模。参考约束、loss 和选模方式均写入报告。复现旧路径需显式使用 `--reference-mode legacy_se3 --geometry-weight 0 --prior-selection representation --flow-selection final`，并匹配旧 σ、预算和 contact 来源。

预测历史适配（仅工程 train take）：

```bash
python -u -m run.collect_predicted_histories --experiment exp/egorecover_engineering_v1 \
  --output exp/my_predicted_train --allow-gt-bootstrap
python -u -m run.engineering_pilot --output exp/my_mixed_history_run \
  --prior-steps 200 --flow-steps 1000 --sigma 0.3 \
  --history-cache exp/my_predicted_train/frames.pt --replay-probability 0.5
```

旧工程缓存由冻结的两个 G 策略生成并合并。显式 `--allow-gt-bootstrap` 只用于重现 GT 启动诊断，正式 FK 损失会拒绝该缓存。随后身体历史全部来自模型提交；目标和当前轨迹都连接同一个此前预测 reference。P/G 混合采样整行字段，Gaussian/History 训练共享缓存和抽样种子。加载器检查 train take、参考模式、统计 SHA256、启动来源和有限性。

训练好的 E7 Flow checkpoint 到位后，使用下面的共同启动/FK 对照路径；官方 diffusion checkpoint 不能代替 E7 Flow。所有 output 均须为新路径，C0/C1 使用同一划分、启动缓存、训练/采样 seed、步数、σ、NFE、动作和开发闭环选模，只改变 `--fk-weight`。此处命令是待执行协议，**目前没有有效 E7 checkpoint，也没有 C0/C1 数值**。

```bash
python -m run.cache_e7_bootstraps --checkpoint /linux/path/to/trained_e7.ckpt \
  --weight-source model --output exp/common_e7_bootstraps.pt
python -u -m run.engineering_pilot --output exp/c0 --checkpoint /linux/path/to/trained_e7.ckpt \
  --bootstrap-cache exp/common_e7_bootstraps.pt --prior-steps 200 --flow-steps 1000 \
  --sigma 0.3 --fk-weight 0 --flow-selection closed_loop_fk
python -u -m run.engineering_pilot --output exp/c1 --checkpoint /linux/path/to/trained_e7.ckpt \
  --bootstrap-cache exp/common_e7_bootstraps.pt --prior-steps 200 --flow-steps 1000 \
  --sigma 0.3 --fk-weight 0.1 --flow-selection closed_loop_fk
python -u -m run.collect_predicted_histories --experiment exp/c1 \
  --bootstrap-cache exp/common_e7_bootstraps.pt --output exp/model_started_train
```

`closed_loop_fk` 在每个评估点以相同的干净开发 episode、E7 启动缓存和采样种子做连续预测历史回放，并用世界 SMPL22 MPJPE 选择 G checkpoint；单步 GT-history 仍只是诊断。P 当前仍在 G 训练前进行单步开发选择，不应由此声称 P 的闭环收益。需要预测历史混训时，重新指定 `--history-cache exp/model_started_train/frames.pt`，并绑定同一个启动缓存。正式训练须另建 train/val/test 及身份级协议；当前固定 split 只用于官方 val 工程验证。

使用实际初始化权重进行预测历史回放：

```bash
python -u -m run.check_closed_loop --experiment exp/my_engineering_run \
  --output exp/my_closed_loop --bootstrap-checkpoint /linux/path/to/e7.ckpt \
  --bootstrap-cache exp/common_e7_bootstraps.pt
```

如果只有随机初始化器，必须显式指定 `--allow-random-initializer` 且不传启动缓存，结果只叫稳定性诊断。可通过 `--source-modes history --variants clean` 定位故障，`--take-index 1` 检查第二个开发 take，`--reference-mode legacy_se3` 复现旧解码。公共缓存保存完整启动运动、reference、世界关节、β 与地面估计；回放重解码并逐项核验。推理结束后才读取标签算 dense 世界位置误差；无逐帧对齐。返回值区分 raw G `normalized_motion` 与 `committed_motion`，记录 reference_mode。

保存闭环结果后，可对**同一模型启动身体**计算后续不再观测的保持/常速度位置基线：

```bash
python -m run.compare_closed_loop_baselines --rollout exp/my_closed_loop
```

`body_baselines.json` 只报告 dense22 位置；它没有身体旋转/FK 指标。随机初始化器最后两帧的速度噪声在长时外推时可能使常速度基线很差，不能与每步有 GT 历史的单步常速度混为一谈。两种协议的历史来源和启动误差必须一起阅读。

## 离线 SMPL-X 几何评估

官方 [SMPL-X 下载页](https://smpl-x.is.tue.mpg.de/download.php) 要求注册、登录并同意模型许可。已在本地 `body_models/smplx/SMPLX_NEUTRAL.npz` 找到授权资产；同目录 `version.txt` 标为 **Version 1.0**，实际模型与 EE4D GT 前 55 关节的最大差小于 0.1 mm。`pip install smplx` 只安装 Python 代码，不包含模型文件；模型资产被 Git 忽略，不提交仓库。

```bash
conda activate egorecover
cd /home/ld666/projects/EgoRecover/UEM-E7
python -m run.evaluate_closed_loop_smplx \
  --rollout exp/egorecover_closed_loop_v3_take0 \
  --output exp/egorecover_closed_loop_v3_take0/smplx_geometry_rerun.json \
  --smplx-dir body_models/smplx --device cuda
```

评估器只读取已经完成的预测历史轨迹。每帧从保存的 `committed_motion` 和 reference 恢复预测姿态，以启动阶段模型自身的 `beta_boot` 固定体型，调用 SMPL-X 得到关节和顶点。GT 只在**推理后**用于模型资产/坐标约定审计和误差计算；GT 前 55 个身体/手部关节与给定模型重建均值需在 5 mm 内、最大值在 20 mm 内，异常则终止且不写报告。还校验恢复的 dense22 与保存轨迹相符、与原闭环报告误差相符。手/身体 MPJPE 与逐帧 PA、头部旋转与眼关节位移、足部滑动/穿透/腾空/接触均沿用原 `eval.metrics` 几何定义；足部指标使用标注地面高度，并在报告中注明。`TMR` 语义相似度和 FID 还需要独立预训练编码器及论文的采样协议，此入口不计算它们。现有 200 帧工程片段的后 180 帧，也不能直接与论文的 80 帧正式验证数值比较。

已对真实保存的 12 条闭环轨迹、2160 帧完成预测状态解码预检；最大关节位置往返差 `9.53674e-7 m`，结果在 [`smplx_prediction_preflight.json`](verification/smplx_prediction_preflight.json)。实际 SMPL-X 结果见 [汇总](verification/smplx_geometry_summary.json)和两条开发视频的完整报告：[`take0`](exp/egorecover_closed_loop_v3_take0/smplx_geometry.json)、[`take1`](exp/egorecover_closed_loop_v3_take1/smplx_geometry.json)。两条 clean/Gaussian 的 SMPL22 MPJPE 分别为 172.38/261.24 mm；它们与原 dense22 的 154.58/220.95 mm 不同，因为固定体型 SMPL-X 正向运动学重新约束了预测旋转与骨长。预测 FK 与直接 dense22 平均相差 70.77/89.06 mm，根关节却几乎完全一致；这种不自洽需在后续模型训练和物理选择中处理，不能混称两类指标。

Issue #3 的可重跑诊断使用 `python -m run.diagnose_fk_dense --rollout exp/egorecover_closed_loop_v3_take0 --source-mode gaussian --variant clean --output verification/my_diagnosis.json`。两条已保存的随机 E7 回放分别见 [足球体型/旋转诊断](verification/issue3_fk_dense_soccer_gaussian_clean.json)、[Covid 体型/旋转诊断](verification/issue3_fk_dense_covid_gaussian_clean.json)。新评估入口还记录逐帧/逐关节误差、root-relative MPJPE 和配对故障增量，见 [足球](verification/issue3_smplx_soccer.json)与[Covid](verification/issue3_smplx_covid.json)。GT β/旋转替换只在离线诊断脚本中使用；两条旧轨迹均为随机启动，不能据此声称方案有效。

历史记录：v0（G 各 100 步）与 v1（G 各 1000 步）使用旧的自由三维 reference。v1 同组 64 帧误差为常速度 15.17 mm、P 38.15 mm、Gaussian G 152.77 mm、History G 168.24 mm；随机启动曾在索引 98 发散。当前参考系修复之后的实验另存，不覆盖这些结果，最新数值见 LOG。

完整阶段记录与下一步条件见 [`LOG.md`](LOG.md)。可运行 `python -m run.report_engineering` 生成 [汇总 JSON](verification/egorecover_engineering_summary.json)、[PNG](verification/egorecover_engineering_summary.png) 和 [PDF](verification/egorecover_engineering_summary.pdf)。正式 Q 标签与训练仍须有有效模型启动及相应可达预测历史，按已接通的 FK 指标重新生成；当前 diagnostic 标签不能替代它们。

本轮参考系/几何损失/预测历史适配的复核汇总由 `python -m run.report_repairs` 生成 [JSON](verification/egorecover_repair_summary.json)、[PNG](verification/egorecover_repair_summary.png)、[PDF](verification/egorecover_repair_summary.pdf)；脚本要求 v3 两个开发 take 的回放和基线先完成。v2 的 P 按物理开发指标保留第 0 步；v3 亦如此，**尚未验证学习型 P 的收益**。v3 的 50% 混合历史实验及完整闭环数值见上层 LOG。

## 后续原因排查

`run.diagnose_engineering` 使用冻结的 v1 权重做通道损失分解、积分步数、理想启动、P/保持、动作和 reference 平面约束对照。GT 启动及 GT 分量替换均明确属于离线归因。

此前完成了 12 条闭环归因对照：原解码均在三组配对中发散，仅约束 reference 后，理想启动的回放均完成，平均 dense22 175–206 mm。该证据已用于修复常规 codec，并统一覆盖初始化和缓存。原因排查脚本仍显式使用 legacy 来保留原对照，不代表当前默认行为。开发单步约从 153/168 mm 降至 109/110 mm，仍高于同帧常速度 15.17 mm；稳定不等于方法精度已成立。

复查第二个开发 take 的配对命令：

```bash
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_my_planar \
  --probe-set planar_pair --skip-single-step --take-index 1
python -m run.report_diagnosis
```

`--eval-split train|dev` 控制单步取样；`--rollout-noise-seed` 控制启动后的采样噪声；`--probe-set none` 只做单步诊断。细节见 [原因排查 JSON](verification/egorecover_cause_analysis.json)、[图](verification/egorecover_cause_analysis.png) 和上层 LOG。
