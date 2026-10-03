# EgoRecover 实验与修改日志

本日志记录当前工作目录中的实验、代码修改、验证命令和结论。指标均属于小规模 pilot，不能直接作为正式论文结果。

## 2026-09-28：12-take pilot 固定分割与 handoff

- 按用户要求跳过新的源文件哈希审查，采用已有数据审计报告作为通过依据。
- 数据集：`data/ee4d_mismatch_train_pilot_v1`。
- 规模：12 takes、84 variants、2400 physical frames；变体为 clean、delay、drift、freeze。
- 固定分割：8 train / 2 dev / 2 holdout，take-disjoint。
- 分割文件：`config/egorecover_train_pilot_split_v1.json`。
- handoff：`data/EE4D_MISMATCH_TRAIN_PILOT_READY.json`。
- 已有审计报告：`data/ee4d_mismatch_train_pilot_v1/audit/validation.json`，状态为 `passed`。
- 固定分割命令：

  ```bash
  conda run -n egorecover python -m run.finalize_train_pilot \
    --dataset data/ee4d_mismatch_train_pilot_v1 \
    --split-out config/egorecover_train_pilot_split_v1.json \
    --handoff-out data/EE4D_MISMATCH_TRAIN_PILOT_READY.json \
    --tests-passed 92
  ```

## 2026-09-28：E7 EMA 启动缓存

- 使用 E7 checkpoint：`exp/e7/last.ckpt`，权重来源为 `ema`。
- 使用 GPU 0，采样 seed `1062`，20-frame clean startup。
- 缓存：`exp/egorecover_train_pilot_v1/bootstrap.pt`。
- 缓存覆盖 train、dev、holdout 的 12 个 take。
- 缓存身份绑定 E7 checkpoint、stats、split manifest、dataset spec 和 reference mode。

## 2026-09-28：pass1 P/G 训练

- 入口：`run/engineering_pilot.py`。
- P 训练 400 steps，G Gaussian/History 各 400 steps，batch size 32。
- E7 EMA 作为初始化器；G 在 dev take 上用闭环 FK 选择 checkpoint。
- 产物目录：`exp/egorecover_train_pilot_v1/pass1`。
- 产物包括 `prior.pt`、`g_gaussian.pt`、`g_history.pt`、`report.json` 和 utility diagnostics。
- P 选中的 dev dense22 误差约 8.47 mm，选择逻辑保留 step 0 baseline；训练曲线中的中间点未被选中。
- Gaussian G dev 闭环 FK 选择点约 216.78 mm。
- History G dev 闭环 FK 选择点约 226.51 mm。

## 2026-09-28：预测历史 replay 缓存

- 使用 pass1 的 Gaussian/History 模型，在 train takes 上闭环生成预测历史。
- 变体：clean、freeze_3s、drift_0p03mps。
- 只使用 8 个 train takes，dev 和 holdout 未进入 replay 训练数据。
- 缓存：`exp/egorecover_train_pilot_v1/replay_cache/frames.pt`。
- 实现入口：`run/collect_predicted_histories.py`、`egorecover/replay.py`。

## 2026-09-28：pass2 replay 续训

- 从 pass1 的 P/G 权重继续训练。
- 训练时以 `replay_probability=0.5` 混合 GT history 与 train-only predicted history。
- 新增 `--init-experiment`，并校验 checkpoint、stats、split、bootstrap 和 reference identity。
- 产物目录：`exp/egorecover_train_pilot_v1/pass2_replay`。
- 结果：replay 在 holdout 上部分改善，但 dev 明显退化，不能判定为稳定收益。
- pass1/pass2 clean SMPL22 MPJPE：

  | 方法 | pass1 dev | pass2 dev | pass1 holdout | pass2 holdout |
  |---|---:|---:|---:|---:|
  | Gaussian | 216.8 mm | 284.7 mm | 296.1 mm | 283.3 mm |
  | History | 226.5 mm | 260.0 mm | 248.7 mm | 231.0 mm |

## 2026-09-28：闭环验证与 E7 baseline

- pass1、pass2 各完成 dev 2 takes 和 holdout 2 takes。
- 每个 take 跑 Gaussian、History、P-only 三种来源，以及 clean、freeze_3s、drift_0p03mps 三种变体。
- 所有闭环报告均通过 finite 检查，在线请求保持因果顺序，推理阶段没有读取 GT body/shape/floor。
- Frozen E7 EMA a11 baseline 也在相同 takes 上完成。
- 离线 SMPL-X 评估产物位于各 `closed_*/smplx.json`。
- 汇总文件：
  - `exp/egorecover_train_pilot_v1/pass1/dev_summary.json`
  - `exp/egorecover_train_pilot_v1/pass1/holdout_summary.json`
  - `exp/egorecover_train_pilot_v1/pass2_replay/dev_summary.json`
  - `exp/egorecover_train_pilot_v1/pass2_replay/holdout_summary.json`

## 验证记录

- `92 passed`：`tests` 与 `data_pipeline/tests`。
- `git diff --check` 通过。
- `python -m compileall -q egorecover run data_pipeline` 通过。
- 当前实验进程已结束，无后台训练、推理或评估进程。

## 当前结论

- 已完成 BLUE_PRINT 中 M2 范围内的小规模 P/G、预测历史和连续闭环工程验证。
- pass1 是当前较稳定的工程基线。
- replay 是历史分布适配方案，不是新的网络模块；当前 pilot 证据不足以将其设为默认训练方案。
- 尚未完成正式全量数据、Q 训练、正式官方测试集主结果或统计显著性结论。

## 2026-09-28：官方 train 扩建至 72 takes

- 新数据集：`data/ee4d_mismatch_train_72take_v1`，官方 train 来源，200 帧/episode，pilot 的 7 个变体。
- 规模：72 takes、72 episodes、504 variants、14,400 physical frames；最终 take-disjoint 分割为 48 train / 12 dev / 12 holdout。
- 保留原 12-take pilot 的全部 8/2/2 归属；新增 60 takes 按 40/10/10 补充。固定名单见 `config/egorecover_train_72take_split_v1.json`，原始归属与扩建调整见数据集的 `split_manifest.json`。
- 构建使用 seed `20260923`、`--base-split train`、`--purpose development`、`--num-takes 72`、`--frames 200`、`--dev-fraction 1/6`、`--holdout-fraction 1/6`。构建后为保持旧 pilot 归属，对新增 take 中的 3 个进行确定性重分配，并重新审计全部 504 个变体。
- 完整数据审计通过：算子回放、标签一致性、源索引、因果读取、take 隔离等检查均通过。按用户要求，本轮未重新计算官方源文件哈希；沿用原 pilot 的源文件指纹并核对文件大小，审计报告保留 `source_hashes_verified=false`，另记录 `source_hashes_accepted_by_user=true`。
- Handoff：`data/EE4D_MISMATCH_TRAIN_72TAKE_READY.json`。更新 `egorecover/data.py` 和 `run/finalize_train_pilot.py`，使显式接受源文件哈希的审计状态可以被读取，并让冻结脚本支持指定分割规模。
- 全部测试：`93 passed`；`git diff --check` 与 `compileall` 通过。本轮只完成数据扩建，尚未在 72-take 分割上训练 P/G 或运行闭环评估。

## 2026-09-28：72-take P/G pass1 训练

- 状态：已完成；报告 `exp/egorecover_train_72take_v1/pass1/report.json` 的 `completed=true`，耗时约 1158 秒。
- 数据集：`data/ee4d_mismatch_train_72take_v1`；固定分割：`config/egorecover_train_72take_split_v1.json`（48 train / 12 dev / 12 holdout）。
- 使用 E7 EMA checkpoint `exp/e7/last.ckpt`，E7 采样 seed `1062`，20-frame clean startup。启动缓存 `exp/egorecover_train_72take_v1/bootstrap.pt` 覆盖全部 72 takes，身份绑定新分割和数据 spec。
- P/G 输出目录：`exp/egorecover_train_72take_v1/pass1`。GPU 0，训练 seed `62`，P 400 steps，Gaussian/History G 各 400 steps，batch size `32`，geometry weight `1.0`，FK weight `0.0`，P 用 dev dense22 选点，G 每 200 步用 dev clean closed-loop FK 选点。
- 本轮不使用 replay；holdout 不参与训练与 checkpoint 选择。
- P 在 step 340 入选：dev teacher-forced dense22 从 35.117 mm 降至 28.970 mm，FK22 从 50.358 mm 降至 44.590 mm。
- Gaussian G 在 step 400 入选：dev clean 闭环 FK22 采样评估从 step 200 的 402.867 mm 降至 312.263 mm；History G 在 step 400 入选：从 482.349 mm 降至 352.444 mm。两者均为 dev 选点指标，不是独立 holdout 结果。
- 产物 `prior.pt`、`g_gaussian.pt`、`g_history.pt` 均可读取；还生成了两个 utility diagnostic checkpoint。尚未为这轮 72-take 模型生成预测历史或运行独立闭环对照。

## 2026-09-28：阶段 1-4 实验实施记录（进行中）

- 为旧 12-take 和新 72-take 模型增加共同 72-take dev/共同 E7 启动缓存的评估路径：`run/check_closed_loop.py` 支持训练与评估不同 manifest，同时拒绝训练 take 与评估 take 重叠；`run/evaluate_paired_development.py` 支持多模型、冻结 E7、take 分片、三种故障变体，先完成全部因果推理再读取监督作离线 SMPL-X 评价。
- `run/engineering_pilot.py` 增加外部 dev 选择集参数，让 12/72-take 预算实验在相同 12-take dev 上选点，保留各自训练 take 和原数据；checkpoint/report 记录选择集身份。
- `run/collect_predicted_histories.py` 增加按 take 分片、train/dev 独立 scope 和与分片无关的随机噪声；`run/merge_predicted_histories.py` 校验完整、不重叠的分片后合并；`run/summarize_paired_development.py` 按 take 汇总宏平均、故障差值及配对置信区间。
- 阶段 1 单 take 跨 manifest clean 冒烟已通过，`fair_bike_01_12` 共用 72-take E7 启动缓存：旧 pilot Gaussian/History/P-only SMPL22 MPJPE 为 243.6/293.6/364.8 mm，新 72-take 模型为 304.3/600.7/784.7 mm。这仅用于验证评估链路，不作为扩建收益结论。报告在 `exp/egorecover_stages_1_4/stage1_smoke/`。
- 阶段 1 多模型多变体单 take 冒烟运行中；阶段 2 预算任务 12-take/400、72-take/1200、72-take/2400 已启动，12-take/1200 与 /2400 待可用 GPU。原 72-take/400 为 `exp/egorecover_train_72take_v1/pass1`。
- 本轮修改已通过 `compileall` 与 `git diff --check`。未动 holdout；未重新审查官方源文件哈希（按用户既定要求接受）。
- 阶段 4 新增 `run/diagnose_four_actions.py`：从按组隔离的可达预测历史中每个 take/变体/来源抽取三个状态（故障前、故障中、恢复期或持续故障后段；clean 用早/中/晚），在相同历史、同一组 K=3 噪声下调用四动作生成，使用固定 E7 启动体型的世界 SMPL22 FK 误差；记录逐状态误差和 oracle 增益。尚未运行，需等预算选择和 replay 缓存。
- 阶段 1 单 take 批量冒烟完成，21 个模型/模式/变体组合全部通过，报告 `exp/egorecover_stages_1_4/stage1_batch_smoke/report.json`；旧 pilot Gaussian clean 243.563 mm，新 72-take Gaussian clean 304.279 mm，冻结 E7 clean 378.175 mm。随后将批量离线指标收窄为 SMPL22 FK 与故障差值，完整 12-take dev 分片 0 已启动。
- 修改后 `compileall`、`git diff --check` 通过；聚焦测试 `4 passed`。
- 全量回归测试：`93 passed, 5 warnings`（警告包括 NVML 初始化与 Lightning DataLoader worker 提示）。
- 阶段 2 `12take_400` 已完成，`report.json` 的 `completed=true`；共同 12-take dev 的 clean 闭环 FK 选点为 Gaussian step 200、232.975 mm，History step 400、231.778 mm。该数值是 checkpoint 选点指标，不是三变体完整配对评估。`12take_1200` 已在 GPU2 启动；72-take/1200、/2400 继续运行。
- 阶段 1 完整 12-take dev 的两个 6-take 分片已在 GPU0/GPU1 运行；当前仍在因果推理阶段，待两分片完成后用 `run/summarize_paired_development.py` 汇总。
- 阶段 2 预算选择规则已在看完整结果之前固定为：每个预算对共同 dev 的三个变体求 take 宏平均，Gaussian 与 History 各得一分，两者平均最小者入选；P-only 单列诊断，不参与预算选择。入口 `run/select_budget.py` 要求 12/72-take 各 400/1200/2400 的完整六格、全部 12 个共同 dev take，且拒绝其他变体组合；单元测试 `1 passed`。
- 阶段 4 抽样修正：经 manifest 核对，freeze_3s 示例故障区间为 61–91 帧，原固定取样会漏掉故障区间，现按每条记录真实事件窗口选择状态。

### 阶段 1 实验报告：扩建前后共同 dev 闭环对照

- 状态：已完成。报告：`exp/egorecover_stages_1_4/stage1_dev/summary.json`；两个完整分片各覆盖 6 takes，合计 12 个共同 dev takes，无 holdout。
- 协议：旧 12-take pilot pass1 与新 72-take/400 pass1 使用相同 72-take 数据集 dev、E7 EMA 启动缓存、seed 62 和 clean/freeze_3s/drift_0p03mps 三变体；先完成因果闭环预测，再离线读取 GT 计算 SMPL22 FK MPJPE。每个 take 对三变体取平均，再对 12 takes 宏平均；区间为按 take 配对 bootstrap 的 95% 区间。
- 结果（mm，越低越好）：

  | 方法 | 旧 12-take | 新 72-take/400 | 新减旧 |
  |---|---:|---:|---:|
  | Gaussian | 232.7 | 306.7 | +74.1 [44.6, 107.2] |
  | History | 240.0 | 313.6 | +73.7 [-22.0, 159.7] |
  | P-only | 359.8 | 749.2 | +389.4 |

- 冻结 E7 EMA Gaussian 为 368.6 mm。72-take/400 的 Gaussian 与 History 虽优于冻结 E7，但均差于旧 pilot；History 差值区间跨零，不能宣称稳定退化。P-only 明显变差，提示先验 P 及预算/选点需单独排查。本阶段仅是共同 dev 对照，不构成独立测试集结论。
- 数值交叉检查：批量评估改用 SMPL22 FK 后，`fair_bike_01_12` 的七个 clean 模型/模式结果与先前全量 SMPL-X 冒烟对应值的最大差异约 0.00003 mm。

### 阶段 2 实验报告：训练预算网格

- 状态：进行中。固定网格为 12/72-take 各 400、1200、2400 steps；选择规则在完整结果出现前已固定为共同 12-take dev 三变体的 Gaussian 与 History take 宏平均的均值最小。P-only 仅诊断；holdout 不参与选点。
- 已完成训练：12-take/400、12-take/1200、72-take/400、72-take/1200。72-take/2400 在 Gaussian 后被中断，History 未完成；12-take/2400 尚未训练。完整共同 dev 配对评估及预算选择尚未执行，不能据单一选点分数选赢家。
- 恢复记录：72-take/2400 的不完整产物保留在 `stage2_budget/72take_2400_interrupted`，不计入六格结果。重新启动 12-take/2400（GPU0）与 72-take/2400（GPU1），以及 12-take/400、/1200 的共同 dev 配对评估（GPU2/3）。初次命令缺少 `PYTHONPATH=.` 或误将数据目录作为 `--signal`，均在训练/推理前退出；改用对应 READY JSON 后重启，空训练输出目录已清理。CUDA 张量计算可用，NVML 查询出现驱动/库版本不一致提示。当前初始化在读取 21 GB 的训练 DINOv2 特征文件。
- 恢复时聚焦回归：预算选择与四动作抽样测试 `2 passed`。
- 为节省中断任务的重复计算，`run/engineering_pilot.py` 新增 `--resume-incomplete`：严格核对训练/选择集、E7、启动缓存、统计量、预算和 Gaussian checkpoint 身份后复用已完成 P/Gaussian，仅继续独立初始化的 History G。原 72-take/2400 重跑在复现 P 曲线后主动中止并保留于 `stage2_budget/72take_2400_restart_interrupted`；随后从最初中断目录续接。该续接不混用 replay；最终须以完整报告和共同 dev 评估验收。

### 阶段 3 实验报告：train-only 预测历史与 replay

- 状态：待阶段 2 选择 72-take 预算。计划对入选模型重建完整 48-take train-only 预测历史，对 p=0、0.25、0.5 做等额继续训练，并在共同 dev 上闭环配对比较；缓存身份和 train/dev 隔离均由脚本校验。

### 阶段 4 实验报告：四动作可达状态诊断

- 状态：待阶段 3 缓存。计划在固定 train/dev 可达预测历史上按真实故障窗口抽样，四动作共享 K=3 噪声，报告每动作误差、oracle 增益和场景分布；这只是 Q 训练前的上限诊断。

## 2026-09-28--17：40：阶段 2–4 恢复实施与资源核对

### 实际状态与已固定计划

- 17:32 核对进程和产物，17:40 再次确认：没有 EgoRecover 训练、闭环评估或预测历史收集进程。前文“已启动/继续运行”是历史记录，不代表当前仍在执行；退出原因尚未确定。
- 阶段 1 已完成，保留 `stage1_dev/summary.json`。阶段 2 已完成训练的仍为 12/72-take 各 400、1200 steps。12-take/2400 仅有 P checkpoint；72-take/2400 有 P/Gaussian，缺 History；两者报告均未标记完成。
- 旧 `stage2_eval/12take_400`、`stage2_eval/12take_1200` 分别有 43、45 条预测轨迹，无最终报告，最后写入约 17:21。因为缺少完整运行身份清单，本轮保留这些产物，并在新目录重新评估，不据文件名直接复用。
- 用户已确认后续计划：GPU 限定 0–3，有其他项目负载时等待；阶段 3 的 p=0、0.25、0.5 三组均从同一阶段 2 入选模型开始，P/Gaussian G/History G 各额外训练 400 步。holdout 不参与本轮选点，不重新计算已接受的官方源文件哈希。
- 阶段 2 恢复策略：12-take/2400 重新完整训练；72-take/2400 使用既有 `--resume-incomplete` 从最初中断目录校验并复用 P/Gaussian，仅重新完成 History。六格各完成共同 dev 的 108 条轨迹，再按预先固定的双来源均值规则分别选择预算。
- 阶段 3 计划收集完整 48-take train 缓存，预期 51,840 条帧记录。阶段 4 固定使用阶段 2 入选模型及其 train/dev 可达历史；不混用 replay 续训后的权重。train/dev 分别覆盖 48/12 takes，K=3 时预期 2,592/648 条状态×噪声诊断记录。

### 本轮已实现与验证

- 新增 `egorecover/evaluation_resume.py`，并为 `run/evaluate_paired_development.py` 增加 `--resume`。新评估保存模型/数据/启动缓存/代码/SMPL-X 资产身份及任务清单；轨迹和进度使用原子写入。续跑核对校验和、帧覆盖、关键张量形状及有限数；缺失或损坏轨迹重算，身份改变和旧无清单目录拒绝续跑。
- `run/diagnose_four_actions.py` 增加故障阶段标签、按 take 汇总，以及先平均 K 次噪声误差再选择动作的 oracle 统计，与逐噪声 oracle 分开报告。该修改尚未产生真实实验结果。
- 聚焦测试命令：`/root/miniconda3/envs/egorecover/bin/python -m pytest -q tests/test_evaluation_resume.py tests/test_four_action_sampling.py tests/test_budget_selection.py`；结果 **7 passed in 3.62s**。覆盖恢复、损坏/缺失轨迹、身份拒绝、故障取样和两种 oracle 统计的区别。`git diff --check` 通过。
- 使用已有 `/tmp/uniegomotion-nvml-host` 中的匹配库，设置 `LD_LIBRARY_PATH` 后 NVML 查询恢复正常，无需修改系统驱动。GPU 0–3 均有其他任务占用，因此本轮尚未启动训练或评估。
- 当前仍待完成：资源等待与自动调度、进一步回归验证、真实六格预算评估、三组 replay 对照、四动作诊断及最终结果汇总。代码实现和单元测试通过不等于实验完成；后续按实际进展追加本日志。

## 2026-09-28--17：49：自动队列已启动，等待 GPU 0–3

- 新增 `run/complete_stages.py` 和 `run/stage_job.py`，接通阶段 2 训练/六格评估/预算选择、阶段 3 train/dev 分片缓存/三组 400-step replay 对照、阶段 4 四动作诊断。每个任务保存命令、PID、GPU、日志、退出码和完成校验；阶段完成或调度失败时自动追加本日志，最终生成运行目录下的 `results.md`。
- 72-take/2400 原始中断产物的续跑配置核对无差异，Gaussian checkpoint 哈希与报告一致。继续训练入口新增 replay 缓存生成 P/G 与初始化权重的一致性检查。
- 全量回归：`/root/miniconda3/envs/egorecover/bin/python -m pytest -q tests data_pipeline/tests`，**103 passed, 5 warnings in 19.53s**。随后补充队列空闲确认与任务依赖测试，相关测试集 **10 passed in 3.62s**；`compileall`、`git diff --check` 均通过。警告包括默认 NVML 加载失败和 Lightning 提示；队列使用已核实的匹配 NVML 库。
- 调度启动时间 17:48:27，控制进程 PID **1312772**。命令：`/root/miniconda3/envs/egorecover/bin/python -u -m run.complete_stages --output exp/egorecover_stages_1_4/continuation_v1 --gpus 0 1 2 3`。
- 状态文件：`exp/egorecover_stages_1_4/continuation_v1/queue.json`；控制日志：同目录 `controller.log`。17:49 确认状态为 **waiting_for_idle_gpu**，阶段 2 的两个训练任务和六个评估任务均为 pending；尚无本轮实验结果。
- GPU 0–3 正在运行另一个项目 **MTU3D 的 R2R-CE 导航数据采集**：每卡三个仿真环境和 Qwen3-VL-4B-Instruct 推理。监督进程 PID 1281153，17:25 启动，参数 `--hours 24`，输出目录 `/gaozt-test1/projects/MTU3D/outputs/r2r_ce_around_memory_24h_20260928_1725`。若未提前结束，按启动预算可能持续至 9 月 29 日约 17:25，实际释放时间以进程状态为准。
- 我们的队列每 30 秒检查，仅在连续两次确认无计算进程、显存占用低于 1 GiB、利用率低于 5% 时分配该卡；每卡最多一个本队列任务。有其他任务占用时继续等待，不中断 MTU3D。当前属于本地后台等待队列，不是集群调度器预留资源。

## 2026-09-28--17：55：按用户要求切换至 GPU 4–7 共享运行

- 用户明确要求使用 GPU 4–7 进行实验，替代此前仅等待 0–3 空闲的资源策略。已确认旧队列全部任务尚为 pending 后，停止等待控制进程 1312772；没有中止任何训练任务。
- 新控制进程 PID **1318845**，17:54:49 启动，继续使用 `exp/egorecover_stages_1_4/continuation_v1`。命令：`/root/miniconda3/envs/egorecover/bin/python -u -m run.complete_stages --output exp/egorecover_stages_1_4/continuation_v1 --gpus 4 5 6 7 --allow-shared --minimum-free-gib 32`。
- 调度器新增显式共享模式：仅使用指定卡，连续两次确认 Default 计算模式且可用显存至少 32 GiB 后启动；每卡最多一个本队列任务。保留默认空闲模式；资源策略变更记录到 queue.json，存在运行中的实验时拒绝直接改策略。32 GiB 是启动前余量检查，不是运行期显存硬限制。
- GPU 4–7 已有 LingBot 四卡分布式训练，各卡约占 47–48 GiB。共享前保存最近 30 步基线：中位步时 **12.308 s**，平均 **12.262 s**；数据在运行目录 `sharing_baseline.json`。共享后的延迟/吞吐不能当作独占 GPU 性能。
- 资源策略与调度测试 **6 passed in 1.28s**；编译和 `git diff --check` 通过。另将阶段 1 的 21 条真实轨迹输入新增恢复校验器，全部通过。
- 本段记录时调度器正在验证已有模型并等待连续显存检查，尚不将 pending 标记为训练完成。首批计划为 GPU4：12-take/2400；GPU5：72-take/2400 History 续跑；GPU6/7：12-take/400、/1200 dev 评估。实际启动及影响另行追加。

### 共享启动后的路径问题与修复

- 17:55:46 首批四个任务已分配至 GPU4/5/6/7，各任务 CUDA 张量检查通过；随后均在加载数据时退出，尚未执行训练或推理。原因是 READY handoff 中的绝对路径仍指向搬迁前的 `/gaozt-test1/guanzerong/EgoRecover`，该目录当前不存在；并非显存不足。
- 修复 `egorecover/data.py`：仅当已记录的路径缺失、且当前项目下相同相对位置确实存在时，在运行时解析为当前路径；记录 `runtime_path_overrides`。数据 spec、manifest、READY 文件本身均未改写，仍按原哈希验证；源文件沿用已接受的审计并核对大小，未重新计算官方源文件哈希。修正默认项目根目录，并为数据读取显式传入解析后的 source root。
- 两份 train READY 的 spec/manifest 哈希检查和全部已登记源文件大小检查通过。搬迁路径与调度聚焦测试 **9 passed in 1.28s**，包含搬迁后仍拒绝篡改 spec 的测试。
- 保留第一次失败的 job 日志和中断输出，使用相同共享命令附加 `--retry-failed` 重启；新控制进程 PID **1321627**。后续实际运行状态以 queue.json 和各 job 日志为准。

### 17:59–18:01 共享运行检查

- 第二次启动于 17:58:58，四个任务的 CUDA 检查均通过，控制队列状态为 `running`：GPU4 的 12-take/2400 训练子进程 **1322580**；GPU5 的 72-take/2400 History 续跑子进程 **1322578**；GPU6 的 12-take/400 dev 评估子进程 **1322579**；GPU7 的 12-take/1200 dev 评估子进程 **1322577**。对应包装进程 PID 1322512–1322515。
- 运行记录分别为 `continuation_v1/jobs/train_12take_2400.2.log`、`train_72take_2400.2.log`、`eval_12take_400.2.log`、`eval_12take_1200.2.log`；`*.2.status.json` 保存实际子进程 PID 和退出码。
- 18:00 左右仍在读取数据/特征文件，四进程累计读取量均约 9.18 GB，尚未输出训练 step 或评估轨迹；未观察到新的异常或 OOM。此时 LingBot 步时尚不能代表完整计算负载下的共享影响。
- 路径修复后的全量回归 **107 passed, 5 warnings in 18.97s**；`compileall` 和 `git diff --check` 通过。已有 READY/spec/manifest 未改写。

## 2026-09-28--18：12：持续监控阶段 2 实际计算

- 用户要求继续监控直至完成四阶段。阶段 1 沿用已完成结果；当前四卡队列状态为 running，后续阶段按依赖自动推进。
- 12-take/2400 已完成 P 的 2400 步，Gaussian G 达到 200 步；72-take/2400 已进入 History G 的 200 步。每 200 步还需对共同 12-take dev 做闭环 FK 选点，因此 step 日志间隔可能较长，不按无新增 step 日志直接判定卡死。
- 12-take/400 与 /1200 的共同 dev 评估各已保存 7 条轨迹；尚未形成完整评估报告，不报告当前模型胜负。
- GPU4/5 总显存占用约 52.5/51.4 GiB，GPU6/7 约 48.6 GiB，未见 OOM。
- 共享计算后 LingBot 最近 20 步中位步时 **15.019 s**，相比共享前 30 步中位 **12.308 s** 增加约 **22%**。这是两个短窗口的观测比较，受负载波动影响；原始记录保存在 `continuation_v1/sharing_active.json`。

### 18:26 监控快照

- 阶段 1 已完成；阶段 2 仍在运行，阶段 3/4 等待预算选择和缓存。
- GPU4：12-take/2400 的 P 已完成，Gaussian G 达 400/2400，History 尚未开始。第 200 步共同 dev clean 闭环 FK 选点分数为 232.975 mm。
- GPU5：72-take/2400 的 History G 达 400/2400；原 P/Gaussian 已复用。第 200 步共同 dev clean 闭环 FK 选点分数为 325.117 mm。上述数值仅用于 checkpoint 选择，不能替代三变体完整评估。
- GPU6/7：12-take/400、/1200 的完整共同 dev 评估均已保存 19/108 条轨迹。另四格评估仍待调度。
- 控制进程 PID 1321627 存活，queue.json 持续更新，状态 running。独立监控每 45 秒写入 `continuation_v1/monitor.json` 和 `monitor.jsonl`。训练每 200 步执行 12-take 闭环选点，当前一轮约需 14 分钟，不能把这段日志静默当作训练停滞。

### 19:23 进度与剩余时间估计

- 阶段 2 控制队列仍为 running，无新失败。12-take/2400 的 Gaussian G 达 1200/2400，History 尚未开始；72-take/2400 的 History G 达 1120/2400 并继续推进。
- 两组独立评估分别完成 68/108、66/108 条轨迹，Gaussian 各 36 条已全生成，正在完成 History；P-only 尚未开始。尚无完整三变体评估结论。
- 近期 Gaussian/History 单条轨迹约 71 秒；训练每 200 步的训练＋共同 dev 闭环选点周期约 14–15 分钟。关键路径是 12-take/2400 剩余 Gaussian 选点、完整 History 训练及该模型的独立评估。
- 按当前共享 GPU 吞吐粗估，阶段 2 还需约 5.5–6.5 小时；阶段 3/4 另需约 3–5 小时。全流程暂估还需 **9–12 小时**，即 9 月 29 日约 **04:30–07:30**。后两阶段尚未实测，该区间不是完成时间承诺；按实际阶段耗时继续更新。

### 阶段 1 与已有预算曲线的阶段性判断

- 阶段 1 对方法收益的支持仍不足：Gaussian 扩建后从 232.7 增至 306.7 mm，配对差值 +74.1 mm、95% 区间 [44.6, 107.2]；History 从 240.0 增至 313.6 mm，但差值区间跨零。两个规模下 History 的点估计都未优于 Gaussian，尚未证明历史源均值的收益。
- P-only 从 359.8 增至 749.2 mm，值得优先排查。旧 pilot P 的选点是 step 0，零初始化残差使其等价于 codec 的物理延续 baseline；新模型采用学习过的 P。单步 GT-history 指标改善而闭环很差，提示预测历史分布偏移或 P 选点目标不匹配，但现有结果不能单独确定因果。
- 阶段 1 同时改变训练规模、每条样本的平均训练覆盖、P 选点和原模型开发集范围，不能把退化全部归因于新增数据。阶段 2 使用共同 dev 和六格预算正是为拆解这些因素。
- 已完成的 1200-step 训练报告给出积极但有限的线索：共同 dev clean 闭环 FK 选点值，12-take Gaussian/History 为 226.277/228.483 mm；72-take 为 221.852/228.852 mm。扩建模型在增加预算后已回到与小数据模型接近的水平。这些是 dev 最佳 checkpoint 选点值，尚非三变体完整配对结果；不得据此提前选择预算。
- 72-take/1200 Gaussian 的 clean dev 曲线在约 222–379 mm 间波动，说明训练/选点仍不稳定；不能假设训练更久会单调改善。
- freeze 对微调模型的故障期增量较小是已有观测，但可能来自更强鲁棒性，也可能来自模型较少使用观测；需阶段 4 同状态四动作结果区分。继续完成既定阶段 2–4，暂不把 replay 或 Q 收益当成已成立结论。

## 2026-09-28--17：56：阶段 2–4 调度停止

- 运行目录：`exp/egorecover_stages_1_4/continuation_v1`；原因：`Experiment failed; dependent stages were not started.`；未完成项不得计入结果。

## 2026-09-28--23：28：阶段 2 五格完整评估与阶段性结论

- 实时队列仍为 running，控制进程 PID 1321627 存活。上面的 09:56 UTC 调度停止条目属于已修复的首次路径失败，不是当前状态。
- 阶段 1 已完成。阶段 2 六格中已有五格完成独立共同 dev 三变体评估，每格均为 completed=true、108 条结果。12-take/2400 的 P/Gaussian 已完成，History 已到 2400/2400，正在等待最后闭环选点及报告写出；其独立评估尚未启动。阶段 3/4 尚未启动。
- 以下均为相同 12 个 dev takes、clean/freeze_3s/drift_0p03mps 的 SMPL22 FK take 宏平均，单位 mm；不是训练选点分数：

  | 预算 | Gaussian | History | P-only | Gaussian/History 均值 |
  |---|---:|---:|---:|---:|
  | 12-take/400 | 232.665 | 239.989 | 359.800 | 236.327 |
  | 12-take/1200 | 229.474 | 229.695 | 359.800 | 229.584 |
  | 72-take/400 | 306.720 | 313.643 | 749.177 | 310.181 |
  | 72-take/1200 | 221.045 | 226.541 | 747.343 | 223.793 |
  | 72-take/2400 | 201.347 | 213.305 | 751.582 | 207.326 |

- 72-take 的更长 P/G 预算带来明确开发集改善：2400 减 400 的 Gaussian 差值 -105.373 mm，take 配对 bootstrap 95% 区间 [-140.143, -75.173]；History 差值 -100.338 mm，区间 [-189.940, -5.660]。双来源均值改善约 33.2%。阶段 1 的 400-step 退化不能代表扩建模型在更充分预算下的最终表现。
- 72-take/2400 相比 12-take/1200 的点估计更好，但 Gaussian 差值 -28.126 mm、95% 区间 [-72.606, 5.892]；History 差值 -16.390 mm、区间 [-44.481, 9.844]。两者区间均跨零，而且预算不相同，尚不能将其表述为稳定的数据扩建收益。相同 1200-step 预算下两模式差值区间也均跨零。
- History 尚未显示优于 Gaussian：72-take/2400 的 History 减 Gaussian 为 +11.958 mm，配对区间 [-22.309, 58.848]，不能宣称显著优劣。学习后 P-only 仍约 747–752 mm，增加预算并未修复其闭环表现；P/历史适配仍是后续重点。
- 上述区间由五份完整报告按现有 take 配对 bootstrap 方法只读计算。六格尚未齐全，未提前运行正式预算选择，也未以中间结论替代阶段 3 replay 或阶段 4 四动作诊断。

## 2026-09-28--23：36：论文指标对照与 clean 误差诊断

- 原始来源：[UniEgoMotion，ICCV 2025，Table 1 / §4](https://arxiv.org/html/2508.01126v1#S4)。论文将 AvatarPoser、EgoEgo、EgoAllo 在 EE4D-Motion 上重新训练评估；以下是该统一对照表中的数值，不是各方法原论文跨数据集成绩。MPJPE/PA 从米换算为毫米：AvatarPoser 116/68，EgoEgo 130/75，EgoAllo 163/71，UniEgoMotion 100/53。
- 本地 `continuation_v1/stage2_eval/72take_2400/report.json` 的 completed=true 完整结果，12 个共同 dev takes：Gaussian clean/freeze/drift = **200.449/200.457/203.137 mm**；History = **212.846/213.448/213.620 mm**；P-only = **751.582 mm**。此前 201.347/213.305 是三条件宏平均；大误差并非主要由这两种人工扰动抬高。
- 同协议 frozen E7 的 clean = **357.685 mm**，Gaussian clean 相比它下降约 44.0%。这是当前工程协议内的改善，不能替代与公开方法的正式比较；本地 E7 是上游 Flow 变体，不能当作论文官方 diffusion checkpoint 的直接复现。
- 协议差异：论文用官方 train/val、8 秒/80 帧片段、整段重建条件；当前是官方 train 内部划出的 12-take dev、2 秒公共模型启动后 18 秒因果闭环，模型只访问当前及历史观测、历史身体状态来自预测，体型固定为 beta_boot。当前微调 train 为 48 takes；开发集还用于选点。这些差异使 200.4 与 100 mm 只能作量级参照，尚不能据此计算正式方法退化率。我们目前的表是未对齐世界 SMPL22 FK MPJPE，不能与论文 PA-MPJPE 混比。
- 对已存逐帧误差作只读分段统计：按启动后 0–6/6–12/12–18 秒分箱，Gaussian clean = **205.838/198.463/197.044 mm**，History = **198.684/216.458/223.396 mm**，P-only = **410.687/728.356/1115.703 mm**。均为每 take 先平均再宏平均。Gaussian 的约 20 cm 误差不能全部归因于随时间累积的漂移；P-only 的增长尤为明显。该分段统计本身不能分离姿态、体型、根平移和朝向误差。
- 后续诊断优先项：同一模型/同一轨迹补充 root-relative、PA、根平移与头部误差；另设匹配论文数据划分、输入可见范围、80 帧采样和指标的评估。前者用于定位误差来源，后者用于正式公开比较。当前尚未执行这些新增评估，不改变既定四阶段队列或选模规则。
- 23:35 队列心跳正常、状态 running；阶段 2 仍五格完成，12-take/2400 训练收尾，其独立评估 pending；阶段 3/4 等待依赖。

## 2026-09-28--23：38：E7 改动审计与 baseline 含义纠正

- 用户指出原始 E7 的指标与 UniEgoMotion 接近。审计发现此前简称“冻结 E7”的 357.685 mm clean / 368.576 mm 三条件结果，并非原始 E7 原生 80 帧整段重建：`run/evaluate_paired_development.py` 的 `path is None` 分支实际实例化 `HistoryUniEgoMotion`，迁入 E7 EMA 后冻结，配合 `HistoryFlow(gaussian, sigma=1)` 和 `run_episode` 逐帧闭环。其准确含义为“E7 权重迁入当前帧历史接口、未适配训练的冻结对照”。此前简称容易混淆，特此纠正；200.449 相比 357.685 的改善不能表述成相对原生 E7 的精度提升。
- 原始骨干仍为 12 层/768 维 Transformer、243 维运动表示、18 维轨迹条件和 1024 维 DINOv2 特征；Euler10、x0 目标、全局增量权重 8 保留。`verification/upstream_e7_parity.json` 证明保留的原始代码路径在合成随机权重下与上游逐位一致，不能证明新历史接口或真实 checkpoint 的原生精度已复现。
- G 改动：原先整个序列作为 Flow 状态，改为只对当前一帧采样；20 帧身体历史固定输入网络。过去原始图像/轨迹被 mask，仅当前帧提供实际观测；不再使用未来观测。新增状态、逐帧 Flow 时间和 prior_mu 三个零初始化条件支路，共 779,520 参数。零初始化保留旧参数，但不能保证在新输入结构下与原始 E7 输出等价。
- P/source 改动：新增 4 层、宽 256 的身体历史先验 P，在 codec 物理延续上预测残差；先训 P 再冻结训练 G。Gaussian 从 sigma*epsilon 开始，History 从 mu+sigma*epsilon 开始；两者都接收身体历史和 mu 条件。当前 sigma=1。Gaussian 也已经是改造模型，不是原生 E7。
- 训练改动：实际 stage2 报告为 predicted_history_cache=null；用 GT 身体历史训练、用模型预测历史闭环评估。G 全部原可训练骨干参数及新分支参与微调；新 AdamW，lr=8.5e-5、batch=32，无新 G 的 EMA/原调度状态恢复。每样本均匀抽取四动作，替代旧网络的条件遮蔽流程。当前几何损失权重 1、FK 损失权重 0：监督的是 dense 世界关节位置，选模/报告却使用 SMPL22 FK。其目标不一致是应检验的解释，尚未完成因果消融。
- 坐标/解码改动：当前轨迹增量连接前一预测 reference，历史换窗重编码，提交 reference 约束为水平位置与 heading；地面由 20 帧启动估计；FK 固定模型启动体型 beta_boot。预测 reference、地面和体型的误差都可能影响结果，不能直接当作已确认原因。
- Q/观测选择目前没有训练上线；正常闭环固定 a11。阶段 3 的预测历史 replay 和阶段 4 四动作诊断仍等待阶段 2 完成，不能将当前退化归因于已启用的 Q。
- 后续归因需要原生 E7 同数据同关节口径基线，并依次控制因果窗口、历史观测保留、身体历史来源和 FK 监督。另发现 `config/e7.yaml` 默认 KEY_JOINTS_ONLY=true（12 关节）；若旧成绩来自该配置，应核对它是否为 mpjpe_key，当前为全部 22 身体关节。尚未找到用户所指原始成绩文件，不推断其指标错误。

### UniEgoMotion 原始损失核对

- 核对官方仓库提交 `580c92c6d70a91672c4106bab30ca82cdd80f379`：[训练损失](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/mydiffusion/gaussian_diffusion.py#L695)、[diffusion 配置](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/module/utils.py#L35)、[运动表示](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/dataset/representation_utils.py#L213)。原始训练采用有效帧上归一化运动表示的 MSE，无额外世界 dense 位置损失、无通过 SMPL FK 反传的位置损失；表示本身包含关节位置和旋转，不能将“无额外位置损失”等同于“无位置监督”。
- 若仅映射到本工程新增项，原方法对应 geometry_weight=0、fk_weight=0；这不表示原 diffusion 与 E7 Flow 的整体训练等价。E7 保留表示监督，并将全局增量 198:207 的权重设为 8；当前 EgoRecover 又增加 geometry_weight=1 的世界 dense 误差平方项，按 (0.1m)^2 缩放。该权重的实际相对影响需记录分项 loss/梯度，不能仅看数值 1 就断言大小。
- 对上一轮解释的限定：fk_weight=0 本身不是偏离 UniEgoMotion 的证据，也不能据此认定当前精度差的原因。需要消融的是新增 dense 项、FK 项是否有益，以及它们与因果历史改造的交互；未改变运行中的实验配置。

## 2026-09-28--23：58：启动 G 损失消融，GPU 4–7

- 用户将新实验用卡从 0–3 更正为 4–7；0–3 从未启动本次训练。任务配置见 `exp/egorecover_loss_ablation/g00_g11_v1/PLAN.md`、哈希与固定协议见 `experiment.json`。
- GPU4：geometry=0/fk=0 Gaussian；GPU5：0/0 History；GPU6：1/1 Gaussian；GPU7：1/1 History。每个 G 从同一 E7 EMA 独立初始化，训练 2400 步；batch=32、sigma=1、Euler10、训练/选点/评估 seed=62/1062/62。
- 为单独检验 G 损失，四任务复用同一个 72-take/2400 P，按原字节复制并校验哈希，P 不训练。该 P 原训练 geometry=1/fk=0；因此组名仅指 G 损失。固定 48 train/12 dev/12 holdout、GT 历史训练、无 replay、公共模型启动与 beta_boot、每 200 步共同 clean-dev 闭环 FK 选点，holdout 不使用。
- 各组完成后自动运行 36 条完整 dev 轨迹；主指标 clean 世界 SMPL22 take 宏平均，辅指标三条件宏平均；对 0/0、1/1 和已有 1/0 进行配对比较并输出 bootstrap 区间。训练记录表示/dense/FK 分项损失。
- 新增单 source 训练、固定 P 身份校验、GPU 指定分配和独立消融控制器；保留原入口默认双 source 行为。`tests` 回归 87 passed、5 warnings（包含新增前四项测试）；另增汇总比较测试单独验证。编译及 git diff --check 通过。
- 4–7 启动前各剩余约 46–48 GiB，已有 LingBot；GPU4 另有原四阶段最后一格评估。新消融共享使用，增加并发可能使两套实验及 LingBot 变慢。原四阶段控制器 PID1321627 保持运行。
- 新控制器 PID **1865814**；当前状态 `running`。已分配的包装进程如下；实际训练进度继续核对 job 日志：
  - train_g0f0_gaussian：GPU4，包装 PID1865904，`exp/egorecover_loss_ablation/g00_g11_v1/jobs/train_g0f0_gaussian.1.log`。
  - train_g0f0_history：GPU5，包装 PID1865905，`exp/egorecover_loss_ablation/g00_g11_v1/jobs/train_g0f0_history.1.log`。
  - train_g1f1_gaussian：GPU6，包装 PID1865906，`exp/egorecover_loss_ablation/g00_g11_v1/jobs/train_g1f1_gaussian.1.log`。
  - train_g1f1_history：GPU7，包装 PID1865907，`exp/egorecover_loss_ablation/g00_g11_v1/jobs/train_g1f1_history.1.log`。

### 2026-09-29 00:02（UTC+8）：四卡消融已进入实际训练

- GPU4，子进程 PID1865951：`G/gaussian 200/2400: train=0.067521`。
- GPU5，子进程 PID1865955：`G/history 200/2400: train=0.057717`。
- GPU6，子进程 PID1865952：`G/gaussian 140/2400: train=0.530220`。
- GPU7，子进程 PID1865954：`G/history 140/2400: train=0.503063`。
- 四份 prior.pt 字节哈希完全相同：`628f412f95fb0a0caf18b7f5ebde4115b972f87925155a847596839734764a5d`。0/0 两项附加损失均为 0；1/1 已输出非零 dense/FK 项并完成多次反向传播，无新报错或 OOM。训练损失不可作为最终闭环精度结论。
- 新增配对汇总测试通过，`tests/test_loss_ablation.py` 共 **5 passed in 1.35s**。实际进程/显存快照见 `startup_verified.json`；训练和自动评估由独立控制器继续执行。

### 2026-09-29 00:04（UTC+8）：损失消融预计耗时

- 四个 G 均达到 200/2400 步，正在第一次共同 12-take clean 闭环 FK 选点；当前日志静默属于验证过程。控制器正常、无失败，GPU4–7 均有计算负载。
- 参考此前同协议每 200 步约 14–15 分钟（主要为 dev 闭环推理），考虑新增共享并发，暂估训练含全部选点还需 3–5 小时，约 03:00–05:00；其后完整三条件评估另需约 1–2 小时，结果暂估 04:00–07:00。新四卡任务尚未完成首轮验证，区间有不确定性，不能按前 200 步纯梯度更新速度线性外推。

## 2026-09-29--00：17：独立 GitHub 仓库发布准备

- 用户要求在 sxh-kk 下新建 EgoRecover，并明确选择公开仓库。将当前工作区代码、配置、文档及轻量验证报告制作成独立首次提交；排除 exp/、数据、模型权重、SMPL-X 资产、缓存和测试 XML，不携带旧 Git 历史中的实验轨迹。
- README 更新为 EgoRecover 总览；原始 E7 说明保留于 E7.md，SOURCE.md 保留上游仓库及提取提交。发布快照独立于训练工作区；原分支和运行中训练进程保持原状态。
- 上传前测试：`tests data_pipeline/tests` **113 passed, 5 warnings in 18.45s**。已检查常见凭据模式，待上传代码未发现匹配项。GitHub 创建和推送结果待认证完成后补记。

### 2026-09-29 00:20（UTC+8）：公开仓库已创建并完成首次推送

- 仓库：[sxh-kk/EgoRecover](https://github.com/sxh-kk/EgoRecover)，可见性 PUBLIC，默认分支 main。首次代码提交 `385ba304063c419443b4a8dc3ee7ca2487a5741a`，181 个文件、约 4.22 MB；远端 main 与本地发布提交一致。
- 干净快照中再次执行完整测试：**113 passed, 5 warnings in 18.33s**。数据、训练权重、SMPL-X、exp/ 与测试 XML 均未上传。当前训练工作区仍保留原 UEM-update 分支及 remote；运行中实验未中止。

## 2026-09-29--00：49：两套实验进度

- 原四阶段控制器与 G 损失消融控制器均存活，queue 心跳正常，未发现失败或 OOM。GPU4–7 利用率接近 100%，各剩余约 42–44 GiB。
- 四阶段：阶段 1 完成；阶段 2 六组训练全部完成、五组完整评估完成。最后一组 12-take/2400 已生成 50/108 条轨迹（Gaussian 36/36、History 14/36、P-only 0/36），最近 History 单条中位耗时约 80.3 秒。按剩余 22 条 G 轨迹及 P-only/离线指标处理估算，阶段 2 约再需 35–50 分钟；阶段 3/4 尚未开始。
- G 损失消融：GPU4 的 0/0 Gaussian 达 600/2400，已完成选点到 400；GPU5 的 0/0 History 达 800/2400，已完成选点到 600；GPU6/7 的 1/1 Gaussian/History 均达 600/2400，已完成选点到 400。无 step 新日志期间正在 12-take clean 闭环选点，未判断为停滞。
- 已完成选点的最佳 clean FK：0/0 Gaussian 201.492 mm @200；0/0 History 207.259 mm @200；1/1 Gaussian 244.356 mm @400；1/1 History 245.900 mm @400。0/0 前期点估计较好，但曲线并非单调（Gaussian step400=284.285，History step600=271.904）；1/1 从 step200 的 339.979/377.149 降至 step400 的 244.356/245.900。以上是选模 seed1062 的中间结果，不能替代 seed62 的最终完整配对评估。
- 按已运行的训练/选点周期约 15–17 分钟暂估，新消融训练含选点约 03:00–04:00 完成，完整评估及汇总约 04:00–05:30。之后原队列进入缓存/replay 会改变共享负载，时间仍可能延后。

## 2026-09-29--01：13：P 预测历史适配方案及延后启动约束

- 用户要求：固定 G 生成 train takes 预测历史，与下一帧 GT 配对训练 P；在固定 dev 预测历史上比较 P、保持和常速度的下一帧 SMPL22 FK 误差。
- 采用独立 P 适配实验，复用四阶段流程中冻结 P/G 生成的 train/dev 历史缓存；训练仅用 train，dev 仅用于 FK 选模与对照，holdout 不使用。保留原 P 作为额外对照，训练损失暂保持 geometry=1、fk=0，以隔离历史分布适配与选模方式的影响。
- 用户明确要求等当前实验结束后再开新实验：须等待原四阶段队列和 geometry/fk 消融队列均成功完成，才允许启动新 P 训练。现在只准备代码与检查，不新增 GPU 训练任务；现有实验继续运行。
- 新实验目前处于实现准备阶段，尚无训练结果。后续启动、完成、异常及指标结论持续追加到本 LOG.md。

### 2026-09-29T01:18:38.029118+08:00：等待范围明确为 EgoRecover，允许与 LingBot 并行

- 用户补充：不需要等待 LingBot 相关进程结束；仅等待当前 EgoRecover 四阶段与 geometry/fk 消融全部完成，再启动新 P 实验。
- 调度条件据此设为两份 EgoRecover queue.json 均 complete 且所有子任务 complete；不会以整机 GPU 空闲或 LingBot 退出作为条件。完成后从 GPU4–7 选择可共享且剩余显存至少 32 GiB 的卡。
- 正在实现独立训练入口及缓存隔离/FK 对照/启动依赖检查；目前未启动新 GPU 训练。后续实际启动与结果由入口自动追加本日志。

## 2026-09-29--01：21：固定 G 历史上的 P 适配实现完成，已挂起等待前置实验

- 新增 `run/adapt_prior_on_predictions.py` 与 `egorecover/prior_adaptation.py`；README 增加入口和指标定义。缓存校验覆盖 train/dev 隔离、完整帧键、统计/划分/冻结 P/G 哈希、模型启动体型及 SMPL-X 资产一致性。原有训练入口与运行中实验协议未修改。
- 从阶段 2 选中的冻结 P 初始化，只用固定 G 在 48 train takes 上生成的历史训练 P；复用 Gaussian/History 两种冻结策略及 clean/freeze_3s/drift_0p03mps 三条件，共 51840 train 帧、12960 dev 帧。每一行历史截至 t-1，监督和 FK 评分对应 t；适配 P 不参与重新生成缓存。
- 默认额外训练 2400 步，batch32、AdamW lr3e-4、seed62、geometry=1/fk=0；每 200 步使用全部固定 dev 历史的下一帧 world SMPL22 FK 选模，take/生成器/条件等权宏平均，step0 可入选。报告原 P、新 P、保持、常速度及逐 take 配对区间；保留逐帧误差、各条件指标、选模曲线、可恢复优化器/RNG checkpoint。dev 选模结果不作为独立 holdout 或闭环提升证据。
- 自动等待控制器 PID **1898417**；输出目录 `exp/egorecover_prior_adaptation/v1`。已核实状态 `waiting_for_existing_experiments`，两份前置 EgoRecover 队列均仍 running；未开始新 P 训练，等待控制器不占 GPU。
- 启动门槛：当前 EgoRecover 四阶段与 geometry/fk 消融两队列及其所有子任务均成功完成；若失败则不提前启动。**不等待 LingBot 结束**，届时在 GPU4–7 中选择剩余显存至少 32 GiB 的卡共享训练。
- 检查：新增 6 项测试通过；完整 `tests data_pipeline/tests` **119 passed, 5 warnings in 18.99s**，CLI 帮助及 `git diff --check` 通过。等待/共享用卡规则、缓存泄漏拒绝、同状态 FK 对照和 take 宏平均均有覆盖；真实新训练指标尚未产生。
- 监控：`cat exp/egorecover_prior_adaptation/v1/status.json`；`tail -n 30 -F exp/egorecover_prior_adaptation/v1/controller.log`。训练启动、每次 FK 选点、完成与异常自动追加 LOG.md。

## 2026-09-29--01：34：阶段 2 六格预算完成

- 运行目录：`exp/egorecover_stages_1_4/continuation_v1`；汇总与选择：`stage2_summary.json`、`stage2_selection.json`。
- 共同 dev 三变体双来源均值（mm）：12take_1200=229.584；12take_2400=229.584；12take_400=236.327；72take_1200=223.793；72take_2400=207.326；72take_400=310.181。
- 分别入选：12-take/1200、72-take/2400；holdout 未使用。

## 2026-09-29--01：39：按用户新意见暂缓 100% 预测历史实验自动启动

- 用户提出更倾向按概率混合真实历史与预测历史，并要求搜索相关工作。为避免旧 100% 方案在讨论期间自动启动，仅停止尚未训练的等待控制器 PID 1898417；状态改为 deferred_pending_protocol_revision。
- 当前四阶段与 geometry/fk 消融继续运行；未停止任何现有训练、评估、历史采集或 LingBot 进程。后续新实验仍遵循仅等待现有 EgoRecover 实验完成、允许与 LingBot 并行的约束。

## 2026-09-29--01：41：GT/预测历史混合训练文献核对及方案建议（未启动新训练）

- 已核实阶段 2 选中 72take_2400；用于后续缓存的 G 为从 E7 EMA 初始化后训练、按 dev 闭环 FK 选出的权重，Gaussian 选中 step600、History 选中 step1400（两者训练预算均 2400）。并非直接冻结原生 E7。原四阶段现已进入四个 train 历史 shard 的采集。
- [Scheduled Sampling，NeurIPS 2015](https://research.google/pubs/scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks/)：训练中按概率使用真实前序输入或模型生成输入，并随训练调整比例，以缓解训练/推理输入分布差异。用户所说“dropout 掺入预测历史”更接近历史替换/混合；置零或 mask 是不同操作。
- [MotionStreamer，ICCV 2025](https://arxiv.org/html/2503.15451v2)：Two-Forward 先生成预测 motion latents，再用预测替换部分 GT latents，第二次前向反传；附录 A 的替换比例使用余弦调度，从 0 增至 1。该工作支持渐进引入预测历史，但不能据此认定某一混合比例对 EgoRecover 最优。
- [Martinez 等，CVPR 2017](https://arxiv.org/pdf/1705.02445)：人体运动预测中让 decoder 在训练时连续接收自身输出，并结合残差结构。说明全部自预测反馈也有有效先例，但其在线多步自反馈训练不同于我们的冻结 G 离线缓存单步 P 适配。
- [DAgger，AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)：反复收集学习策略遇到的状态并查询专家标签、聚合训练数据；可启发后续更新 P/G 后刷新历史，但我们的一次固定缓存监督不等同完整 DAgger。
- 建议下一轮以 P-only 的 GT/预测历史混合为主，保持同一初始 P、冻结 G 和相同训练预算，比较预测历史比例 0、0.25、0.5 及 0→0.5 渐增；这些比例是本项目建议，非论文已验证结论。主评估仍用固定 dev 预测历史 FK，辅看 GT 历史 FK，再验证接回 G 后闭环。
- 需区分：已有阶段 3 的 replay_probability 按完整样本选择 GT 或预测历史，同时继续训练 P/G；用户如需同一 20 帧窗口内部分替换，应在统一物理坐标下重编码历史及目标，避免直接拼接不同参考系的归一化向量。当前未改运行中的阶段 3 协议。
- 100% 新实验等待控制器已停止，状态 deferred_pending_protocol_revision；混合方案当前为建议，尚未修改为新的自动训练任务。当前 EgoRecover/LingBot 任务继续执行。

## 2026-09-29--01：48：相关工作对 EgoRecover 修改的具体启发（设计分析）

- Scheduled Sampling / [MotionStreamer](https://arxiv.org/html/2503.15451v2) 启发先保留 GT 历史、逐渐增加预测历史比例。下一轮 P-only 比较可用 0/0.25/0.5/0→0.5；混合比例与调度是待验证假设，当前未据此新增训练任务。
- 应优先使用模型实际产生的、时间连续的预测历史；若进一步研究窗口内部替换，可从 GT 前缀展开连续短段预测，逐渐延长反馈跨度，避免在不同参考系的 243D 编码间直接逐帧拼接。这属于本项目实现建议，不能表述为论文已证明最优。
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html) 启发分轮适配并刷新 train 历史：P/G 更新会改变实际运行状态分布，因此旧冻结缓存只代表旧策略。第一轮固定缓存适合公平定位 P，后续可收集新策略历史并与旧数据/GT 聚合；不把此监督流程宣称为完整 DAgger。
- P 和 G 都读取历史，两者都可能需要适配。现有阶段 3 同时更新 P/G 的 replay 对照保留；P-only 补充实验单独定位先验收益。固定 G 权重后替换新/旧 P 的实际闭环评估仍必要，因为 P 的变化会改变 G 条件及后续历史。
- [Martinez 等 CVPR 2017](https://arxiv.org/pdf/1705.02445) 的残差与多步反馈训练说明需要关注动作变化和误差传播。现有 P 已是物理保持基线加残差；常速度基线加残差、显式速度输入可留作后续结构消融，不与本轮历史分布变化同时修改。
- 判断顺序：固定预测历史的单步 FK 与原 P/保持/常速度对照；GT 历史单步 FK 检查是否退化；接回冻结 G 后实际闭环 FK 检查系统收益。单步 dev 改善不能直接推出闭环改善，最终独立泛化结论需要保留集验证。

## 2026-09-29--01：53：实验顺序修订为先训练验证 P，再接入 G

- 用户明确目标：先单独训练 P，确认预测精度后再接入 G，进行定性分析。独立计划见 `P_FIRST_PLAN.md`。
- 第一阶段使用 train GT 历史预测下一帧 GT；在同一 dev GT 历史上比较原 P、新 P、保持和常速度，按 world SMPL22 FK 选模，并检查根部/身体误差及代表性可视化。原阶段 2 本来就是 GT 历史训练 P，本次重点是验证物理预测精度并建立完整基线，不能将其表述为新增数据训练方式。
- 确认 P 的收益后冻结 P，再在匹配条件下训练 G；首轮固定 source mode、初始化、预算和损失，分析学习先验相对保持先验的作用。接入后的单步与闭环结果分别检查。
- 若 P 在 GT 历史表现好、在 G 预测历史上退化，再研究 GT/预测历史混合；预测历史适配不作为 P 首轮独立验证的前置步骤。当前不预先保证精度或宣称已达到指标。
- 核实原等待控制器 PID1898417 已不在运行；原 100% 方案保持 deferred_pending_protocol_revision。未启动新的 P/G 实验；现有四阶段和损失消融继续运行。后续仍仅等待当前 EgoRecover 实验完成，允许在 GPU4–7 与 LingBot 并行。

## 2026-09-29--02：02：参考 MotionStreamer 提出 P 的具体改进方案

- 用户澄清要参考相关方法改进 P，而不仅是设计评估表。完成方法文档 `P_MOTIONSTREAMER_PLAN.md`，并更新总计划 `P_FIRST_PLAN.md`。当前仅设计，未实现或启动新训练。
- 核对 [MotionStreamer 论文 §3.3/附录 A](https://arxiv.org/html/2503.15451v2) 及[官方训练代码](https://github.com/zju3dv/MotionStreamer/blob/8aace3f6f5f564ae1518aa77d925ae1380070c80/train_motionstreamer.py)，参考提交 `8aace3f6f5f564ae1518aa77d925ae1380070c80`。作者使用第一次 diffusion 分支的 pred_xstart 替换部分 GT latents，detach 后第二次前向监督仍对应 GT，替换比例从 0 余弦增加到 1；不能将其第一次前向说成完整自由自回归 rollout。
- 本项目改进：保留现有 20 帧→243D 的 P，先 GT 预热，再以当前 P 严格因果单步预测产生历史候选，按概率替换 GT 历史，第二次 P 前向对下一帧 GT 反传。只训练 P，不调用 G；无需先依赖 G 的训练或历史缓存。部署仍只运行一次 P 前向。
- 初始建议为 20% 更新 GT 预热、60% 更新将替换概率余弦升至 0.5、最后 20% 保持 0.5。这是本项目适配，非论文原配置；P 预测必须 detach、逐时刻因果对齐，替换在 BodyState 上进行，再重编码历史 reference 链、base_mu 和当前 GT target。
- 第二项方法改动为加入可微 FK 监督，直接约束目标 SMPL22 物理误差；以 GT/Two-Forward × fk0/fk1 的四组匹配对照区分贡献，geometry 保持 1。FK 监督是本项目针对评价指标提出的改进，不归因于 MotionStreamer 的 AR 损失。
- 主验证为 GT 历史下一帧 FK，辅验证为 P 自反馈 200/400/800/1000ms、根部/关节误差和代表性骨架可视化。第一遍 GT 条件单步预测尚不覆盖长期累计误差，也不能保证适配 G 的误差；先独立确认 P 再接入 G。
- 旧 100% G 历史缓存方案继续暂停；当前实验不变。新实验须等待当前 EgoRecover 两队列完成，可与 LingBot 在 GPU4–7 并行。

## 2026-09-29--02：03：EgoRecover 运行进程快照

- 当前实际 GPU 工作为 4 个 G 损失消融训练/选点任务，加 4 个原四阶段 train 历史缓存推理任务，全部使用 GPU4–7；两套队列心跳正常、未报告失败。
- GPU4：geometry0/fk0 Gaussian，PID1865951；train_shard0 采集 PID1902470。GPU5：0/0 History，PID1865955；train_shard1 PID1902467。GPU6：1/1 Gaussian，PID1865952；train_shard2 PID1902468。GPU7：1/1 History，PID1865954；train_shard3 PID1902469。
- 四个 G 均已到 1600/2400，最新输出处于每 200 步一次的 dev 闭环选点阶段，最终完整评估仍 pending；这套实验固定 P，仅更新 G。
- 原四阶段：阶段 2 已全部完成；正在为阶段 3/4 收集固定 P/G 的 train 预测历史。每卡 12 个 train takes，四个 shard 均完成 Gaussian 三条件及 History/clean，正在 History/freeze_3s（第 5/6 条件组合）；dev 四个缓存任务仍 pending。后续 replay 训练和四动作诊断尚未开始。
- 控制器 PID1321627（四阶段）、PID1865814（G 损失消融）。GPU4–7 总显存占用约 53–54 GiB/卡（含并行其他任务），利用率 100%。
- 旧 100% G 历史 P 适配自动启动保持停止；参考 MotionStreamer 的 P 改进仅完成设计，未启动。现有缓存采集属于原四阶段协议。

## 2026-09-29--02：29：独立 P 的 Two-Forward / FK 改进已实现；文档分类整理

- 方法与协议：[P 改进方案](docs/experiments/p-motionstreamer.md)、[运行说明](docs/experiments/p-two-forward-runbook.md)。P 保留原 4 层 Transformer / 20 帧 / 243D 接口，单独训练，不调用 G。
- 候选历史由当前 P 在 no_grad/eval 下对每个历史位置的 20 帧 GT 前缀预测；在 BodyState 物理坐标中按整帧替换，再重新编码历史、base_mu 和 GT 当前目标。第二遍才反传；不是完整自由 rollout，也不是固定 G 的预测缓存。
- 首轮 A=GT+dense、B=Two-Forward+dense、C=GT+dense+FK、D=Two-Forward+dense+FK；geometry 均为 1，FK 为 0/1。分别绑定 GPU4/5/6/7。相同 seed62 初始化、t40…199、batch32、2400 更新、AdamW3e-4；前 20% GT 预热，中间 60% 余弦升至替换率 0.5，末 20% 保持。
- 每 100 更新按全部 dev GT 下一帧世界 SMPL22 FK 选点，包含 step0。另报匹配 t≥40 子集、root/root-relative/dense/rotation 分解；每 take 固定 9 个起点做 100–1000ms 纯 P 自反馈；固定 t80 全部 dev takes 可视化。保持、常速度、原 P 同协议参照。首轮最优 B/C/D 与 A 补 seed63/64，报告按 take 配对区间；不使用 holdout，不自动接入 G。
- 新增 `egorecover/prior_two_forward.py`、`run/prepare_prior_sequences.py`、`run/train_prior_two_forward.py`、`run/prior_two_forward_experiment.py`；支持缓存审计、checkpoint/产物哈希、随机状态恢复、计算量与排除评估的同步训练墙钟耗时。
- 验证：全套 tests + data_pipeline/tests 为 126 passed（20.95s）；最终改动后 P/队列/旧适配相关 19 passed（8.20s）。新增 7 项测试覆盖因果性、p0 等价、重编码、第一遍 detach/第二遍 FK 梯度、split/哈希隔离、自反馈、矩阵及恢复；模拟 step1 中断后的 step2 参数和选模曲线与连续训练完全相同。
- 实际 SMPL-X 资产 + 现有统计 + 完整 P 的 CPU 合成运动前向/反向检查通过，未执行优化器更新：`verification/prior_two_forward_cpu_smoke.json`。图表生成检查通过。以上是实现验证，不构成精度结果。
- 文件管理：[docs 索引](docs/README.md)。根目录仅保留 README.md、LOG.md、SOURCE.md；方法、设计、数据、参考、历史草稿分别归档。迁移后 README/SOURCE/docs 的本地 Markdown 链接目标均存在。LOG 历史原文与旧路径保留。
- 迁移：P_FIRST_PLAN.md → docs/experiments/p-first.md；P_MOTIONSTREAMER_PLAN.md → docs/experiments/p-motionstreamer.md；EGORECOVER.md → docs/design/interfaces.md；BLUE_PRINT.md → docs/design/blueprint.md；DATASET.md → docs/data/dataset.md；EXPERIMENTS.md → docs/archive/engineering-experiments.md；ICCV2027.md → docs/archive/iccv2027-draft.md；E7.md → docs/reference/e7.md。
- 正式训练须等待现有四阶段队列和 G 损失消融队列总状态与所有任务全部 complete；失败不释放条件。不等待 LingBot，GPU4–7 共享且可用显存至少 32 GiB。当前四阶段已进入 dev 历史缓存生成；G 消融仍在运行。

## 2026-09-29--02：29：独立 P 改进等待调度器已启动

- PID 1923633；`exp/egorecover_prior_two_forward/v1/controller.log`，状态 `queue.json`。此时仅 CPU 等待进程，尚未启动新的数据准备或 GPU 训练。
- 旧 `exp/egorecover_prior_adaptation/v1` 保持停止；现有实验成功完成后新队列自动执行数据审计、四组训练评估和种子复核。

## 2026-09-29--03：30：G 损失消融中断

- 运行目录：`exp/egorecover_loss_ablation/g00_g11_v1`；原因：`Experiment failed; dependent stages were not started.`；未完成结果不计入比较。

## 2026-09-29--05：24：阶段 3 replay 对照完成

- 运行目录：`exp/egorecover_stages_1_4/continuation_v1`；训练缓存 51,840 帧；三组 P/G 各额外训练 400 步。
- 共同 dev 双来源均值（mm）：before_replay=207.326；replay_0=193.043；replay_0p25=254.719；replay_0p5=224.966。
- 共同 dev 配对汇总：`stage3_summary.json`；综合比较：`stage3_comparison.json`。
- 三组最低分：replay_0；相对 p=0 的 replay 改善证据：False。

## 2026-09-29--05：34：阶段 4 四动作诊断完成

- 报告：`exp/egorecover_stages_1_4/continuation_v1/stage4/train.json`、`exp/egorecover_stages_1_4/continuation_v1/stage4/dev.json`。
- 完整结果表：`exp/egorecover_stages_1_4/continuation_v1/results.md`。
- 固定阶段 2 入选 P/G，train/dev 分别覆盖 48/12 takes，2,592/648 条状态×噪声记录；包含逐噪声和先平均噪声的 oracle。
- 阶段 2–4 已完成；这些是单训练种子开发实验及同状态诊断，未训练 Q，未使用 holdout。

## 2026-09-29--12：01：状态检查：四阶段完成；修复并恢复 G 消融评估

- 截至 2026-09-29 12:00（UTC+8），原四阶段队列于 05:34 完成全部任务。阶段 3 三条件、双来源平均 FK：继续 GT 训练 p0=193.043mm；p0.25=254.719mm；p0.5=224.966mm；继续训练前为207.326mm。p0.25/p0.5 相对 p0 分别增加61.676/31.923mm，take配对95%区间分别[17.149,101.053]/[5.738,66.165]mm。
- 阶段3三组新 P 均选择本轮 step0（保留原 P）；P-only 长时闭环均为751.582mm。收益不能归因为 P 已改善，且旧整窗口 replay 不等同新 P Two-Forward 帧内替换。
- 阶段4 dev 同状态、先平均噪声 oracle 相对 a11 收益仅0.297–0.548mm；当前四动作选择上限小。未训练Q，未进行holdout评估。
- G消融4个训练均已完成2400步。评估于03:30报告失败：权重哈希代码无条件读取 g_gaussian.pt/g_history.pt，单History目录缺少前者。该异常发生在建立评估进度前，未产生正式配对结论。
- 修复 run/evaluate_paired_development.py：仅检查请求来源使用的checkpoint，prior_only沿用加载器的Gaussian映射。新增3项参数化回归检查；损失消融/评估恢复/队列相关17 passed（3.68s），真实四组checkpoint路径预检通过。原模型与训练配置均保留，旧失败状态备份为 queue.before_eval_repair_20260929.json。
- 新P队列因前置消融failed一直处于等待，尚未开始任何训练；恢复消融评估成功完成后会自动启动新P实验。

## 2026-09-29--12：01：G 损失消融评估调度恢复

- 控制器 PID 2016646；使用GPU4–7。仅重试未完成评估，不重新训练；等待中的P控制器PID1923633继续使用原完成条件。

## 2026-09-29--12：08：按用户要求停止 G 消融评估，立即切换独立 P 改进实验

- 用户最新指令：停止评估，进行 P 的修改实验。已停止 G 消融控制器2016646及4个实际评估进程2017046/2017048/2017047/2017049，核实全部退出。已训练4组G权重和已有产物保留，队列标记stopped_by_user，不标记评估完成，不自动重启。
- 停止旧P等待控制器1923633；新增显式 --skip-loss-ablation-wait 选项，移除G消融完成依赖。原四阶段已complete，P数据与2400步四组对照协议保持不变。仍使用GPU4–7，可与LingBot共享。

## 2026-09-29--12：08：独立 P 改进调度重新启动

- 控制器PID2020807；先在GPU4准备48 train/12 dev共60条身体序列并做SMPL-X资产审计，完成后四组P分别在GPU4/5/6/7训练。输出`exp/egorecover_prior_two_forward/v1`。

## 2026-09-29--12：08：P 实验启动条件调整

- 按用户要求停止 G 消融评估，P 不再等待其完成；四阶段完成条件保留。
- 仅修改尚未启动任务的调度依赖，P 的数据、训练与评估协议不变。

## 2026-09-29--12：10：P 实验已进入数据准备

- 核实新队列状态running，prepare_data包装进程2021091、实际进程2021157，绑定GPU4；四个旧G评估进程已全部退出。数据审计完成后自动启动四组P训练。

## 2026-09-29--12：12：按用户要求暂缓 P 训练，检查 GPU0–3 是否可共享

- 已停止P调度器2020807。停止时只有prepare_data任务运行，尚无任何P训练任务启动；现有数据准备可继续完成，但不会自动开始训练。队列标记paused_by_user，待用户明确恢复。
- GPU0–3当前各占约10.3GiB/95.6GiB，利用率42–47%；GPU4–7各占47–48GiB，利用率89–99%。0–3存在其他用户kjx的WEM/Qwen3-VL-4B captions推理任务，不能视为空闲卡。
- 显存角度可以考虑与0–3现有任务共享；共享会竞争算力和带宽。当前P调度仅支持固定4–7，尚未扩展到0–7，也未启动额外GPU任务。

## 2026-09-29--12：18：按用户要求在 GPU0–3 启动独立 P 四组训练

- 只使用GPU0/1/2/3，分别为A_gt_dense、B_two_forward_dense、C_gt_fk、D_two_forward_fk；geometry均为1、FK分别0/0/1/1。与现有Qwen推理共享；旧G消融评估保持停止。
- 复用已完成的48 train/12 dev身体序列缓存：[exp/egorecover_prior_two_forward/v1/data/report.json](exp/egorecover_prior_two_forward/v1/data/report.json)；缓存已核验哈希与60条完整审计记录。此前数据准备使用GPU4属于历史任务，新训练只在0–3。
- 四组相同seed62/2400步/batch32；每100步按完整dev下一帧FK选模，包含step0。之后最优B/C/D候选与A补seed63/64，继续只用0–3。
- 结果记录要求：每个实验写明配置、seed、进度/状态、所选step、指标及限制；LOG条目提供可点击的配置、选点曲线、report.json和checkpoint链接。结束时记录下一帧与1秒自反馈FK、逐take报告路径；异常明确记录，不能写成完成。
- 运行状态：[exp/egorecover_prior_two_forward/v1/queue.json](exp/egorecover_prior_two_forward/v1/queue.json)；总日志：[exp/egorecover_prior_two_forward/v1/controller.log](exp/egorecover_prior_two_forward/v1/controller.log)。旧四阶段完整结果：[exp/egorecover_stages_1_4/continuation_v1/results.md](exp/egorecover_stages_1_4/continuation_v1/results.md)。

## 2026-09-29--12：18：GPU0–3 P 训练调度器已启动

- 控制器PID 2025072；[exp/egorecover_prior_two_forward/v1/controller.log](exp/egorecover_prior_two_forward/v1/controller.log)；[exp/egorecover_prior_two_forward/v1/queue.json](exp/egorecover_prior_two_forward/v1/queue.json)。

## 2026-09-29--12：18：P 实验调度配置调整

- GPU [4, 5, 6, 7] → [0, 1, 2, 3]；等待队列 ['exp/egorecover_stages_1_4/continuation_v1/queue.json']。
- P 训练尚未开始，只改变资源/等待配置；沿用已完成的数据准备，训练与评估协议不变。

## 2026-09-29--12：19：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=True，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/initial.pt)。

## 2026-09-29--12：19：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=False，geometry=1/fk=1，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/initial.pt)。

## 2026-09-29--12：19：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=False，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/initial.pt)。

## 2026-09-29--12：19：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=True，geometry=1/fk=1，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/initial.pt)。

## 2026-09-29--12：20：P 四组训练运行确认与文件入口

- 四组均已进入实际训练，控制器PID2025072。下列链接指向每个实验的实时日志、配置与已生成的进度文件；完整结果完成后会由训练器自动追加可点击的report.json链接。
- A_gt_dense_s62：GPU0，实际PID2025432，最近已完成选点step=500。日志：[exp/egorecover_prior_two_forward/v1/jobs/A_gt_dense_s62.1.log](exp/egorecover_prior_two_forward/v1/jobs/A_gt_dense_s62.1.log)；配置：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/experiment.json)。进度：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/progress.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/progress.json)；选点：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/selection.json)。
- B_two_forward_dense_s62：GPU1，实际PID2025434，最近已完成选点step=500。日志：[exp/egorecover_prior_two_forward/v1/jobs/B_two_forward_dense_s62.1.log](exp/egorecover_prior_two_forward/v1/jobs/B_two_forward_dense_s62.1.log)；配置：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/experiment.json)。进度：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/progress.json](exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/progress.json)；选点：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/selection.json)。
- C_gt_fk_s62：GPU2，实际PID2025431，最近已完成选点step=200。日志：[exp/egorecover_prior_two_forward/v1/jobs/C_gt_fk_s62.1.log](exp/egorecover_prior_two_forward/v1/jobs/C_gt_fk_s62.1.log)；配置：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/experiment.json)。进度：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/progress.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/progress.json)；选点：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/selection.json)。
- D_two_forward_fk_s62：GPU3，实际PID2025433，最近已完成选点step=200。日志：[exp/egorecover_prior_two_forward/v1/jobs/D_two_forward_fk_s62.1.log](exp/egorecover_prior_two_forward/v1/jobs/D_two_forward_fk_s62.1.log)；配置：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/experiment.json](exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/experiment.json)。进度：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/progress.json](exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/progress.json)；选点：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/selection.json)。

## 2026-09-29--12：22：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=42.246 mm。
- 1秒自反馈 FK=265.880 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/report.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：25：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1900；dev next-frame FK=40.321 mm。
- 1秒自反馈 FK=300.967 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/report.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：26：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1900；dev next-frame FK=43.276 mm。
- 1秒自反馈 FK=267.015 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/report.json](exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/B_two_forward_dense_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。



## 2026-09-29--12：27：实验日志改为结论摘要

- 按用户要求，LOG不再逐次记录独立P选点；已清理此前自动追加的选点段落。完整曲线仍保留在各实验selection.json，训练输出保留在jobs日志中。
- 每轮结束在LOG写清实验结论：相对基线改善/退化多少、Two-Forward和FK各自的贡献、跨种子方向是否一致、短时自反馈是否退化；附完整结果表与报告的可点击链接。开始/异常保留简短必要记录。
- 本轮P仍在运行，尚未形成完整四组和种子复核结论；训练不重启。只重启调度器以采用新的最终结论汇总格式，当前GPU0–3训练任务继续。





## 2026-09-29--12：29：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1100；dev next-frame FK=41.771 mm。
- 1秒自反馈 FK=281.364 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/report.json](exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/selection.json](exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/D_two_forward_fk_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：30：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=False，geometry=1/fk=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/experiment.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/initial.pt)。

## 2026-09-29--12：30：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=False，geometry=1/fk=1，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/experiment.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/initial.pt)。

## 2026-09-29--12：30：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=False，geometry=1/fk=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/experiment.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/initial.pt)。

## 2026-09-29--12：30：独立 P Two-Forward 对照开始/恢复

- [exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63](docs/release/migration.md#为什么选择这些checkpoint)；Two-Forward=False，geometry=1/fk=1，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/experiment.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/experiment.json)；初始权重：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/initial.pt)。

## 2026-09-29--12：33：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2100；dev next-frame FK=41.707 mm。
- 1秒自反馈 FK=237.220 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/report.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/selection.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：33：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2100；dev next-frame FK=42.353 mm。
- 1秒自反馈 FK=255.818 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/report.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/selection.json](exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/A_gt_dense_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：33：P 改进首轮结论（seed62，待种子复核）

- 首轮四组均完成，当前仅seed62；候选C正在与A进行seed63/64复核。完整dev下一帧使用GT历史，1秒测试只给20帧GT前缀、随后使用自己的预测；holdout未使用。
- Two-Forward当前未带来收益：B相比A下一帧误差增加1.029mm，按take配对95%区间[0.088,2.265]mm；D比C增加1.450mm。B/D的一秒自反馈也均未优于A。这是当前实现/配置的结果，不外推为所有历史混合方法无效。
- FK监督(C)的下一帧均值最好40.321mm，比A降低1.925mm（约4.6%），但配对95%区间[-5.142,1.040]mm跨零；1秒误差从265.880升至300.967mm。尚不能认为获得稳定的整体改进。
- 常速度下一帧36.750mm，优于所有学习型P；保持的一秒239.051mm也优于四组P。当前P未超过简单基线，进一步改进仍有必要。
- 下一帧精度和长时自反馈存在取舍，不能仅凭FK选点均值下降宣称P已适合接入G。待种子复核后更新最终结论。
- 完整对照表：[exp/egorecover_prior_two_forward/v1/SCREENING.md](exp/egorecover_prior_two_forward/v1/SCREENING.md)；配对区间/逐take指标：[exp/egorecover_prior_two_forward/v1/screening.json](exp/egorecover_prior_two_forward/v1/screening.json)。

## 2026-09-29--12：36：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=39.721 mm。
- 1秒自反馈 FK=263.057 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/report.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/selection.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：36：独立 P 对照完成

- [exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=39.634 mm。
- 1秒自反馈 FK=262.359 mm；完整指标/逐take结果：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/report.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/report.json)；选点曲线：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/selection.json](exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/selection.json)；所选权重：[exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_two_forward/v1/train/C_gt_fk_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--12：36：独立 P 改进实验结论

- seed62 下一帧 FK 相对 A 的变化：Two-Forward(B) +1.029 mm；FK监督(C) -1.925 mm；组合(D) -0.475 mm。负值表示改善。
- 候选 C_gt_fk 的种子复核：三个种子均降低下一帧误差；seed62/63/64 相对 A 分别为 -1.925/-2.072/-2.632 mm。
- 1秒自反馈相对 A 的变化（seed62/63/64）：+35.087/+25.139/+7.239 mm；正值表示自反馈退化。
- 候选三种子平均下一帧FK 39.892 mm；常速度基线 36.750 mm。以上是dev选模结果，尚未验证独立holdout或接入G后的收益。
- 完整结果表与图：[exp/egorecover_prior_two_forward/v1/RESULTS.md](exp/egorecover_prior_two_forward/v1/RESULTS.md)；配对区间和逐take指标：[exp/egorecover_prior_two_forward/v1/summary.json](exp/egorecover_prior_two_forward/v1/summary.json)。

## 2026-09-29--12：37：P 误差诊断：固定体型影响明显，运动预测仍未超过常速度

- 对同一12-take dev共2160个下一帧进行CPU离线诊断：GT当前姿态/根位置 + 模型启动beta的FK相对记录GT误差28.353mm；同样GT姿态配GT beta后为0.272mm。28.353mm是参考误差，不是严格可达下界；不能从P的40mm中直接相减。GT beta只用于离线诊断，没有修改正式P或评价协议。
- 统一到相同启动体型后，预测FK相对GT姿态FK的误差：常速度16.839mm、A24.637mm、B26.172mm、C28.401mm、D29.097mm。P运动预测仍弱于常速度；C的正式FK改善不等于姿态预测改善，结果与通过姿态补偿体型差异的可能性一致，不能直接作因果定论。
- 下一轮建议优先检验常速度基线加可学习残差、显式平移/角速度输入；再分别研究体型一致的监督与根部/局部旋转输出，以及真实2–4步自反馈训练。先验证优于常速度，再讨论增加模型规模或继续Two-Forward。当前为建议，未启动新训练。
- 诊断完整指标：[exp/egorecover_prior_two_forward/v1/diagnostics/shape_reference.json](exp/egorecover_prior_two_forward/v1/diagnostics/shape_reference.json)；复算脚本：[exp/egorecover_prior_two_forward/v1/diagnostics/shape_reference.py](exp/egorecover_prior_two_forward/v1/diagnostics/shape_reference.py)。

## 2026-09-29--14：13：P 常速度残差实验设计

- 用户同意设计“常速度＋残差”和标准MPJPE评价实验。本次完成方案，尚未修改训练代码、启动训练或产生新结果。
- 四组：保持/常速度 × FK=0/1；geometry=1、GT历史和网络结构固定，seeds62/63/64全部配对训练，共12任务。另评估纯保持与纯常速度。区别“换基准的收益”和“残差学习超过常速度的收益”，step0纳入选模。
- 主指标为世界SMPL22 Body MPJPE；补充PA-MPJPE、根位置、世界/局部旋转、统一体型及dense位置指标。完整单步与100–1000ms自反馈分别报告，按take配对统计。
- 工程目标：三种子平均Body MPJPE≤35mm且统一体型MPJPE≤15mm，更强目标为Body MPJPE≤33mm；还需确认跨种子超过纯CV。门槛是项目建议，不是论文标准。冻结配置后再做独立holdout，接G收益另行验证。
- 实施时延续GPU0–3；新产物预定存放exp/egorecover_prior_cv_residual/v1/。完整设计与文件规划：[P常速度残差实验](docs/experiments/p-cv-residual.md)；[文档索引](docs/README.md)。

## 2026-09-29--14：20：P 常速度残差实现与启动前验证

- 用户最新指定GPU4–7，覆盖本轮设计中的GPU0–3。首轮H0/V0/H1/V1分别绑定4/5/6/7，三个种子依次运行，与LingBot共享资源。
- 独立训练入口与评价已实现；15项测试通过，覆盖物理基准、旋转外推、PA指标、精确断点恢复和跨种子统计。旧Two-Forward训练代码及产物未改动。
- 真实GPU4资产检查通过：纯CV Body MPJPE=36.7505mm、PA-MPJPE=13.0426mm、统一体型MPJPE=16.8395mm；零残差模型与CV逐项一致，FK反传有限。这些是基线验证，尚不是新训练结论。
- 验证记录：[verification/prior_cv_residual_gpu_smoke.json](verification/prior_cv_residual_gpu_smoke.json)；实验设计：[P常速度残差](docs/experiments/p-cv-residual.md)。

## 2026-09-29--14：20：P 常速度残差实验启动

- GPU[4, 5, 6, 7]；与现有任务共享，空闲显存门槛32 GiB；GT历史四组×三个种子，共12次训练。
- 计划：[exp/egorecover_prior_cv_residual/v1/plan.json](exp/egorecover_prior_cv_residual/v1/plan.json)；状态：[exp/egorecover_prior_cv_residual/v1/queue.json](exp/egorecover_prior_cv_residual/v1/queue.json)；设计：[常速度残差](docs/experiments/p-cv-residual.md)。

## 2026-09-29--14：21：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=hold，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/experiment.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/initial.pt)。

## 2026-09-29--14：21：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=hold，geometry=1/fk=1，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/experiment.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/initial.pt)。

## 2026-09-29--14：21：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/experiment.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/initial.pt)。

## 2026-09-29--14：21：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=1，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/experiment.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/initial.pt)。

## 2026-09-29--14：23：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=42.246 mm。
- 1秒自反馈 FK=265.880 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/report.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/selection.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：25：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1700；dev next-frame FK=36.000 mm。
- 1秒自反馈 FK=346.027 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/report.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/selection.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：25：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1900；dev next-frame FK=40.321 mm。
- 1秒自反馈 FK=300.967 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/report.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/selection.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：25：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1700；dev next-frame FK=33.369 mm。
- 1秒自反馈 FK=484.691 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/report.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/selection.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：26：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=hold，geometry=1/fk=1，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/experiment.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/initial.pt)。

## 2026-09-29--14：26：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=hold，geometry=1/fk=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/experiment.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/initial.pt)。

## 2026-09-29--14：26：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=1，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/experiment.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/initial.pt)。

## 2026-09-29--14：26：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/experiment.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/initial.pt)。

## 2026-09-29--14：27：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2100；dev next-frame FK=41.707 mm。
- 1秒自反馈 FK=237.220 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/report.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/selection.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：27：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2300；dev next-frame FK=35.749 mm。
- 1秒自反馈 FK=370.270 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/report.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/selection.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：28：P 常速度残差首轮结论（seed62，非最终）

- 更换基准的单步收益：V0相对H0从42.246降到36.000mm（约14.8%）；V1相对H1从40.321降到33.369mm（约17.2%）。
- 残差相对纯CV的单步收益较小：V0为36.000 vs 36.750mm（约2.0%）；V1为33.369mm（约9.2%）。不能把换基准的全部收益算成残差学习收益。
- V0的PA-MPJPE=11.618mm、统一体型=16.451mm；纯CV为13.043/16.839mm。V1加FK后正式MPJPE更低，但PA=13.442mm、统一体型=19.445mm、局部旋转=5.582°，均弱于V0（局部旋转2.650°）；与姿态补偿体型差异的可能性一致，但尚不能证明因果。
- 连续预测：V0/V1在200ms为55.486/54.343mm，优于纯CV59.419mm；到400ms为116.147/134.163mm，纯CV124.728mm；到1000ms为346.027/484.691mm，均弱于纯CV320.266mm及保持239.051mm。V1的单步改善伴随明显的自反馈退化。
- 目前没有配置同时达到Body≤35mm与统一体型≤15mm目标。V0更适合作为后续运动预测改进的参照，V1需重点排查监督与自反馈稳定性；最终选择仍按预定三种子主指标，不能依据单个种子更改结论。
- 阶段结果：[seed62分析](exp/egorecover_prior_cv_residual/v1/SCREENING.md)；[逐take配对指标](exp/egorecover_prior_cv_residual/v1/screening.json)。其余种子继续运行，最终结论由队列完成后另行汇总。

## 2026-09-29--14：28：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2200；dev next-frame FK=33.471 mm。
- 1秒自反馈 FK=488.621 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/report.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/selection.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：28：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=39.634 mm。
- 1秒自反馈 FK=262.359 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/report.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/selection.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：29：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=1，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/experiment.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/initial.pt)。

## 2026-09-29--14：29：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=hold，geometry=1/fk=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/experiment.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/initial.pt)。

## 2026-09-29--14：29：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/experiment.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/initial.pt)。

## 2026-09-29--14：29：P 常速度残差对照开始/恢复

- [exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=hold，geometry=1/fk=1，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/experiment.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/experiment.json)；初始权重：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/initial.pt)。

## 2026-09-29--14：30：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2100；dev next-frame FK=42.353 mm。
- 1秒自反馈 FK=255.818 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/report.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/selection.json](exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H0_hold_dense_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：30：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=36.069 mm。
- 1秒自反馈 FK=352.340 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/report.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/selection.json](exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V0_cv_dense_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：31：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1900；dev next-frame FK=33.110 mm。
- 1秒自反馈 FK=504.955 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/report.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/selection.json](exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/V1_cv_fk_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：31：P 常速度残差单组完成

- [exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=39.721 mm。
- 1秒自反馈 FK=263.057 mm；完整指标/逐take结果：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/report.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/report.json)；选点曲线：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/selection.json](exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/selection.json)；所选权重：[exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_cv_residual/v1/train/H1_hold_fk_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：31：P 常速度残差实验结论

- V0_cv_dense 相对 H0_hold_dense：Body MPJPE变化 -6.163 mm，take配对95%区间 [-8.010, -4.506]。
- V1_cv_fk 相对 H1_hold_fk：Body MPJPE变化 -6.576 mm，take配对95%区间 [-8.726, -4.764]。
- 候选 V1_cv_fk：Body MPJPE 33.317 mm，PA-MPJPE 13.780 mm，统一体型 19.784 mm；超过纯CV的跨种子/区间要求：未达到；35/15 mm工程目标：未达到。
- 候选相对纯CV的seed62/63/64差值：-3.381/-3.279/-3.641 mm。
- 候选200ms自反馈MPJPE 55.202 mm；相对配对保持残差组 -8.317 mm；该时长最强物理基线 59.419 mm。
- 候选400ms自反馈MPJPE 135.676 mm；相对配对保持残差组 +19.320 mm；该时长最强物理基线 124.728 mm。
- 候选1000ms自反馈MPJPE 492.756 mm；相对配对保持残差组 +217.294 mm；该时长最强物理基线 239.051 mm。
- 以上为dev选模结果；holdout尚未评估，尚未验证接入G的收益。
- 完整结果：[exp/egorecover_prior_cv_residual/v1/RESULTS.md](exp/egorecover_prior_cv_residual/v1/RESULTS.md)；统计：[exp/egorecover_prior_cv_residual/v1/summary.json](exp/egorecover_prior_cv_residual/v1/summary.json)。

## 2026-09-29--14：43：P 显式速度输入实现与验证

- 用户要求在常速度＋残差、FK=0上加入显式速度。两组×三个种子配对重训，保持网络主体、数据、损失和2400步预算。最新资源授权为GPU4–7。
- 70维速度特征由过去身体历史计算：根部平移3、根部空间角速度3、21关节局部角速度63、有效位1；先恢复共同参考坐标再差分。dt=0.1s，不读GT当前帧或未来；新增零初始化70→256线性分支，共17920参数。
- 14项测试通过：物理单位、坐标不变性、因果性、共有权重/RNG一致、精确恢复和配对统计；真实GPU4反传与checkpoint重载通过，速度分支可学习，step0严格等于CV。
- 用户调整评价优先级：100ms单步精度优先，纯P长程只作诊断，不作单独淘汰条件；G预测历史适配和接G效果以后独立验证。
- 方案：[显式速度输入](docs/experiments/p-explicit-velocity.md)；验证：[verification/prior_velocity_smoke/report.json](verification/prior_velocity_smoke/report.json)。尚无本轮训练结论。

## 2026-09-29--14：43：P 显式速度输入实验启动

- GPU[4, 5, 6, 7]；常速度/FK=0，两组×seeds62/63/64，共6次训练。
- 计划：[exp/egorecover_prior_velocity/v1/plan.json](exp/egorecover_prior_velocity/v1/plan.json)；队列：[exp/egorecover_prior_velocity/v1/queue.json](exp/egorecover_prior_velocity/v1/queue.json)；[方案](docs/experiments/p-explicit-velocity.md)。

## 2026-09-29--14：44：P 显式速度对照开始/恢复

- [exp/egorecover_prior_velocity/v1/train/control_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，velocity_input=False，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_velocity/v1/train/control_s62/experiment.json](exp/egorecover_prior_velocity/v1/train/control_s62/experiment.json)；初始权重：[exp/egorecover_prior_velocity/v1/train/control_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/control_s62/initial.pt)。

## 2026-09-29--14：44：P 显式速度对照开始/恢复

- [exp/egorecover_prior_velocity/v1/train/control_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，velocity_input=False，geometry=1/fk=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_velocity/v1/train/control_s63/experiment.json](exp/egorecover_prior_velocity/v1/train/control_s63/experiment.json)；初始权重：[exp/egorecover_prior_velocity/v1/train/control_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/control_s63/initial.pt)。

## 2026-09-29--14：44：P 显式速度对照开始/恢复

- [exp/egorecover_prior_velocity/v1/train/velocity_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，velocity_input=True，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_velocity/v1/train/velocity_s62/experiment.json](exp/egorecover_prior_velocity/v1/train/velocity_s62/experiment.json)；初始权重：[exp/egorecover_prior_velocity/v1/train/velocity_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/velocity_s62/initial.pt)。

## 2026-09-29--14：44：P 显式速度对照开始/恢复

- [exp/egorecover_prior_velocity/v1/train/velocity_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，velocity_input=True，geometry=1/fk=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_velocity/v1/train/velocity_s63/experiment.json](exp/egorecover_prior_velocity/v1/train/velocity_s63/experiment.json)；初始权重：[exp/egorecover_prior_velocity/v1/train/velocity_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/velocity_s63/initial.pt)。

## 2026-09-29--14：46：P 显式速度单组完成

- [exp/egorecover_prior_velocity/v1/train/control_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2300；dev next-frame FK=35.749 mm。
- 1秒自反馈 FK=370.270 mm；完整指标/逐take结果：[exp/egorecover_prior_velocity/v1/train/control_s63/report.json](exp/egorecover_prior_velocity/v1/train/control_s63/report.json)；选点曲线：[exp/egorecover_prior_velocity/v1/train/control_s63/selection.json](exp/egorecover_prior_velocity/v1/train/control_s63/selection.json)；所选权重：[exp/egorecover_prior_velocity/v1/train/control_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/control_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：48：P 显式速度单组完成

- [exp/egorecover_prior_velocity/v1/train/velocity_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2300；dev next-frame FK=35.703 mm。
- 1秒自反馈 FK=365.333 mm；完整指标/逐take结果：[exp/egorecover_prior_velocity/v1/train/velocity_s63/report.json](exp/egorecover_prior_velocity/v1/train/velocity_s63/report.json)；选点曲线：[exp/egorecover_prior_velocity/v1/train/velocity_s63/selection.json](exp/egorecover_prior_velocity/v1/train/velocity_s63/selection.json)；所选权重：[exp/egorecover_prior_velocity/v1/train/velocity_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/velocity_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：49：P 显式速度单组完成

- [exp/egorecover_prior_velocity/v1/train/control_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1700；dev next-frame FK=36.000 mm。
- 1秒自反馈 FK=346.027 mm；完整指标/逐take结果：[exp/egorecover_prior_velocity/v1/train/control_s62/report.json](exp/egorecover_prior_velocity/v1/train/control_s62/report.json)；选点曲线：[exp/egorecover_prior_velocity/v1/train/control_s62/selection.json](exp/egorecover_prior_velocity/v1/train/control_s62/selection.json)；所选权重：[exp/egorecover_prior_velocity/v1/train/control_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/control_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：50：P 显式速度单组完成

- [exp/egorecover_prior_velocity/v1/train/velocity_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1700；dev next-frame FK=35.883 mm。
- 1秒自反馈 FK=359.662 mm；完整指标/逐take结果：[exp/egorecover_prior_velocity/v1/train/velocity_s62/report.json](exp/egorecover_prior_velocity/v1/train/velocity_s62/report.json)；选点曲线：[exp/egorecover_prior_velocity/v1/train/velocity_s62/selection.json](exp/egorecover_prior_velocity/v1/train/velocity_s62/selection.json)；所选权重：[exp/egorecover_prior_velocity/v1/train/velocity_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/velocity_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：50：P 显式速度对照开始/恢复

- [exp/egorecover_prior_velocity/v1/train/control_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，velocity_input=False，geometry=1/fk=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_velocity/v1/train/control_s64/experiment.json](exp/egorecover_prior_velocity/v1/train/control_s64/experiment.json)；初始权重：[exp/egorecover_prior_velocity/v1/train/control_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/control_s64/initial.pt)。

## 2026-09-29--14：50：P 显式速度对照开始/恢复

- [exp/egorecover_prior_velocity/v1/train/velocity_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，velocity_input=True，geometry=1/fk=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_velocity/v1/train/velocity_s64/experiment.json](exp/egorecover_prior_velocity/v1/train/velocity_s64/experiment.json)；初始权重：[exp/egorecover_prior_velocity/v1/train/velocity_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/velocity_s64/initial.pt)。

## 2026-09-29--14：52：P 显式速度单组完成

- [exp/egorecover_prior_velocity/v1/train/control_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=36.069 mm。
- 1秒自反馈 FK=352.340 mm；完整指标/逐take结果：[exp/egorecover_prior_velocity/v1/train/control_s64/report.json](exp/egorecover_prior_velocity/v1/train/control_s64/report.json)；选点曲线：[exp/egorecover_prior_velocity/v1/train/control_s64/selection.json](exp/egorecover_prior_velocity/v1/train/control_s64/selection.json)；所选权重：[exp/egorecover_prior_velocity/v1/train/control_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/control_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：53：P 显式速度单组完成

- [exp/egorecover_prior_velocity/v1/train/velocity_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=35.613 mm。
- 1秒自反馈 FK=345.192 mm；完整指标/逐take结果：[exp/egorecover_prior_velocity/v1/train/velocity_s64/report.json](exp/egorecover_prior_velocity/v1/train/velocity_s64/report.json)；选点曲线：[exp/egorecover_prior_velocity/v1/train/velocity_s64/selection.json](exp/egorecover_prior_velocity/v1/train/velocity_s64/selection.json)；所选权重：[exp/egorecover_prior_velocity/v1/train/velocity_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_velocity/v1/train/velocity_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--14：53：P 显式速度输入实验结论

- 显式速度相对配对对照：Body MPJPE 35.939→35.733 mm，差值 -0.206 mm，take配对95%区间 [-0.457, +0.048]。
- seed62/63/64差值：-0.118/-0.046/-0.456 mm。
- PA-MPJPE 11.755→11.679 mm；统一体型 16.374→15.914 mm。
- 单步稳定改善要求（各seed均改善且配对区间低于0）：未达到。纯P长程只作辅助诊断，不作为本轮单独淘汰条件。
- 统一体型误差差值−0.461mm的take配对95%区间为[−0.753,−0.164]，三个seed均改善；相较正式MPJPE，这项运动预测诊断的改善证据更明确，但仍属于dev选模结果。
- 相对纯CV Body MPJPE差值 -1.018 mm；35/15 mm工程目标：未达到。
- dev参与选模；holdout、G预测历史上的单步适配及接入G收益尚未验证。
- 结果：[exp/egorecover_prior_velocity/v1/RESULTS.md](exp/egorecover_prior_velocity/v1/RESULTS.md)；统计：[exp/egorecover_prior_velocity/v1/summary.json](exp/egorecover_prior_velocity/v1/summary.json)。

## 2026-09-29--15：11：P 统一体型FK监督实现与验证

- 用户已明确启动正式训练。四组新损失权重0/0.1/0.3/1.0，均为常速度＋残差、geometry=1、原FK=0、无显式速度；三个种子共12次训练，GPU4–7。
- 新目标为预测姿态和GT姿态各自使用相同模型启动体型的FK关节差；正式MPJPE仍对原始GT计算。12项测试与真实GPU反传通过，batch32实测峰值PyTorch reserved约0.314GiB，不含CUDA上下文及其他进程。
- 启动显存门槛设为12GiB，连续两次满足后启动；显存不足的卡自动等待。纯P长程仅作诊断。
- 方案：[统一体型FK监督](docs/experiments/p-same-shape-fk.md)；验证：[verification/prior_same_shape_gpu_smoke.json](verification/prior_same_shape_gpu_smoke.json)。

## 2026-09-29--15：11：P 统一体型监督实验启动

- GPU[4, 5, 6, 7]；四个权重×三个种子，共12次训练；显式速度关闭。
- 空闲显存门槛12GiB；计划：[exp/egorecover_prior_same_shape/v1/plan.json](exp/egorecover_prior_same_shape/v1/plan.json)；状态：[exp/egorecover_prior_same_shape/v1/queue.json](exp/egorecover_prior_same_shape/v1/queue.json)。

## 2026-09-29--15：12：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/control_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/control_s62/experiment.json](exp/egorecover_prior_same_shape/v1/train/control_s62/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/control_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/control_s62/initial.pt)。

## 2026-09-29--15：12：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose10_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=1，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose10_s62/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose10_s62/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose10_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose10_s62/initial.pt)。

## 2026-09-29--15：12：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose01_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0.1，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose01_s62/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose01_s62/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose01_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose01_s62/initial.pt)。

## 2026-09-29--15：12：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose03_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0.3，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose03_s62/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose03_s62/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose03_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose03_s62/initial.pt)。

## 2026-09-29--15：13：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/control_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1700；dev next-frame FK=36.000 mm。
- 1秒自反馈 FK=346.027 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/control_s62/report.json](exp/egorecover_prior_same_shape/v1/train/control_s62/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/control_s62/selection.json](exp/egorecover_prior_same_shape/v1/train/control_s62/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/control_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/control_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：13：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/control_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/control_s63/experiment.json](exp/egorecover_prior_same_shape/v1/train/control_s63/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/control_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/control_s63/initial.pt)。

## 2026-09-29--15：15：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose10_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1700；dev next-frame FK=35.771 mm。
- 1秒自反馈 FK=356.080 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose10_s62/report.json](exp/egorecover_prior_same_shape/v1/train/pose10_s62/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose10_s62/selection.json](exp/egorecover_prior_same_shape/v1/train/pose10_s62/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose10_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose10_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：15：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose01_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=36.383 mm。
- 1秒自反馈 FK=364.120 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose01_s62/report.json](exp/egorecover_prior_same_shape/v1/train/pose01_s62/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose01_s62/selection.json](exp/egorecover_prior_same_shape/v1/train/pose01_s62/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose01_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose01_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：15：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose03_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=35.905 mm。
- 1秒自反馈 FK=358.387 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose03_s62/report.json](exp/egorecover_prior_same_shape/v1/train/pose03_s62/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose03_s62/selection.json](exp/egorecover_prior_same_shape/v1/train/pose03_s62/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose03_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose03_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：15：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose10_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=1，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose10_s63/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose10_s63/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose10_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose10_s63/initial.pt)。

## 2026-09-29--15：15：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose01_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0.1，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose01_s63/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose01_s63/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose01_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose01_s63/initial.pt)。

## 2026-09-29--15：15：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/control_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2300；dev next-frame FK=35.749 mm。
- 1秒自反馈 FK=370.270 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/control_s63/report.json](exp/egorecover_prior_same_shape/v1/train/control_s63/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/control_s63/selection.json](exp/egorecover_prior_same_shape/v1/train/control_s63/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/control_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/control_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：15：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose03_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0.3，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose03_s63/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose03_s63/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose03_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose03_s63/initial.pt)。

## 2026-09-29--15：15：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/control_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/control_s64/experiment.json](exp/egorecover_prior_same_shape/v1/train/control_s64/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/control_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/control_s64/initial.pt)。

## 2026-09-29--15：17：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/control_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=36.069 mm。
- 1秒自反馈 FK=352.340 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/control_s64/report.json](exp/egorecover_prior_same_shape/v1/train/control_s64/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/control_s64/selection.json](exp/egorecover_prior_same_shape/v1/train/control_s64/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/control_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/control_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：17：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose01_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1500；dev next-frame FK=35.881 mm。
- 1秒自反馈 FK=356.104 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose01_s63/report.json](exp/egorecover_prior_same_shape/v1/train/pose01_s63/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose01_s63/selection.json](exp/egorecover_prior_same_shape/v1/train/pose01_s63/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose01_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose01_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：17：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose10_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=35.874 mm。
- 1秒自反馈 FK=345.877 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose10_s63/report.json](exp/egorecover_prior_same_shape/v1/train/pose10_s63/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose10_s63/selection.json](exp/egorecover_prior_same_shape/v1/train/pose10_s63/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose10_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose10_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：18：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose01_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0.1，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose01_s64/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose01_s64/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose01_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose01_s64/initial.pt)。

## 2026-09-29--15：18：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose03_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1500；dev next-frame FK=35.934 mm。
- 1秒自反馈 FK=366.213 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose03_s63/report.json](exp/egorecover_prior_same_shape/v1/train/pose03_s63/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose03_s63/selection.json](exp/egorecover_prior_same_shape/v1/train/pose03_s63/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose03_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose03_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：18：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose10_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=1，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose10_s64/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose10_s64/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose10_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose10_s64/initial.pt)。

## 2026-09-29--15：18：P 统一体型监督对照开始/恢复

- [exp/egorecover_prior_same_shape/v1/train/pose03_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/original_fk=0/same_shape=0.3，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_same_shape/v1/train/pose03_s64/experiment.json](exp/egorecover_prior_same_shape/v1/train/pose03_s64/experiment.json)；初始权重：[exp/egorecover_prior_same_shape/v1/train/pose03_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose03_s64/initial.pt)。

## 2026-09-29--15：20：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose01_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1600；dev next-frame FK=35.866 mm。
- 1秒自反馈 FK=361.816 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose01_s64/report.json](exp/egorecover_prior_same_shape/v1/train/pose01_s64/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose01_s64/selection.json](exp/egorecover_prior_same_shape/v1/train/pose01_s64/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose01_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose01_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：20：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose10_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step2400；dev next-frame FK=35.749 mm。
- 1秒自反馈 FK=345.314 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose10_s64/report.json](exp/egorecover_prior_same_shape/v1/train/pose10_s64/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose10_s64/selection.json](exp/egorecover_prior_same_shape/v1/train/pose10_s64/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose10_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose10_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：20：P 统一体型监督单组完成

- [exp/egorecover_prior_same_shape/v1/train/pose03_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step1600；dev next-frame FK=35.697 mm。
- 1秒自反馈 FK=347.099 mm；完整指标/逐take结果：[exp/egorecover_prior_same_shape/v1/train/pose03_s64/report.json](exp/egorecover_prior_same_shape/v1/train/pose03_s64/report.json)；选点曲线：[exp/egorecover_prior_same_shape/v1/train/pose03_s64/selection.json](exp/egorecover_prior_same_shape/v1/train/pose03_s64/selection.json)；所选权重：[exp/egorecover_prior_same_shape/v1/train/pose03_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_same_shape/v1/train/pose03_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--15：21：P 统一体型监督实验结论

- 同体型权重0.1：Body MPJPE变化+0.104 mm，95%区间[-0.209, +0.414]；统一体型变化+0.021 mm；局部旋转变化+0.079°。
- 权重0.1的seed62/63/64正式MPJPE差值：+0.383/+0.132/-0.203 mm。
- 同体型权重0.3：Body MPJPE变化-0.094 mm，95%区间[-0.454, +0.211]；统一体型变化-0.314 mm；局部旋转变化+0.161°。
- 权重0.3的seed62/63/64正式MPJPE差值：-0.095/+0.185/-0.372 mm。
- 同体型权重1：Body MPJPE变化-0.141 mm，95%区间[-0.520, +0.183]；统一体型变化-0.390 mm；局部旋转变化+0.348°。
- 权重1的seed62/63/64正式MPJPE差值：-0.229/+0.125/-0.319 mm。
- 按三种子正式MPJPE选出的候选为pose10：Body 35.798 mm、PA 11.526 mm、统一体型 15.984 mm。
- 候选相对配对对照的各seed/区间改善要求：未达到；35/15 mm工程目标：未达到。
- 主指标始终对原始GT计算；纯P长程仅作诊断。dev用于checkpoint及三个权重的选择，区间未作多重比较校正，不能代替独立验证。
- holdout和接入G收益尚未验证；显式速度分支未启用。
- 结果：[exp/egorecover_prior_same_shape/v1/RESULTS.md](exp/egorecover_prior_same_shape/v1/RESULTS.md)；完整统计：[exp/egorecover_prior_same_shape/v1/summary.json](exp/egorecover_prior_same_shape/v1/summary.json)。

## 2026-09-29--15：24：P 后续主线与数据预算实验设计

- 用户决定采用常速度＋残差、FK=0：原FK及统一体型FK均关闭，显式速度分支不采用。现有实验产物保留，不修改已完成结果。
- 统一体型FK三种子结果：权重1正式MPJPE改善仅0.141mm，配对95%区间跨0；局部旋转2.670°→3.018°，尚无稳定整体收益。
- 下一轮建议只改变数据量和训练预算，采用当前/扩充train片段 × 2400/9600步的四组对照；数据扩充先审计同48个train takes的可用时长，dev/holdout固定。
- 本次仅设计，未准备扩充数据、未启动新训练。方案：[数据量与训练预算](docs/experiments/p-data-budget.md)；FK结果：[exp/egorecover_prior_same_shape/v1/RESULTS.md](exp/egorecover_prior_same_shape/v1/RESULTS.md)。

## 2026-09-29--15：44：P D组扩量队列启动

- 按用户决定只做D组，目标192个train take、每take最多4个独特20秒片段；seeds62/63/64，各9600步，GPU4–7。
- 常速度残差、geometry=1、所有FK损失=0、不加显式速度；先审计数据再自动训练，当前尚无新效果结论。
- 计划：[exp/egorecover_prior_data_budget/v1/plan.json](exp/egorecover_prior_data_budget/v1/plan.json)；队列：[exp/egorecover_prior_data_budget/v1/queue.json](exp/egorecover_prior_data_budget/v1/queue.json)；完成后结果：[exp/egorecover_prior_data_budget/v1/RESULTS.md](exp/egorecover_prior_data_budget/v1/RESULTS.md)。

## 2026-09-29--15：50：P D组数据清单冻结

- 已选择192个train take（保留48、新增144）、562个互不重叠20秒片段，共187.33分钟，是原16分钟的11.71倍。原dev与holdout保持不变。当前正在逐take审计，尚无新训练效果结论。
- 6项扩量测试及原7项常速度测试通过，覆盖划分隔离、跨源时间去重、take均衡采样、dev逐张量固定、断点恢复和配对结果汇总；新take的真实E7启动与SMPL审计已开始通过。
- 方案：[P D组](docs/experiments/p-data-budget.md)；冻结清单：[exp/egorecover_prior_data_budget/v1/data/manifest.json](exp/egorecover_prior_data_budget/v1/data/manifest.json)；数据进度：[exp/egorecover_prior_data_budget/v1/data/progress.json](https://huggingface.co/datasets/sxhkk/EgoRecover-data/resolve/fdbff9f6a9856ff5b13e9818edf8677e395c6ee2/exp/egorecover_prior_data_budget/v1/data/progress.json)；队列：[exp/egorecover_prior_data_budget/v1/queue.json](exp/egorecover_prior_data_budget/v1/queue.json)。

## 2026-09-29--15：52：P D组扩量数据审计完成

- 192个train take、562个不重叠20秒片段；原12个dev逐张量一致；holdout未使用。
- 数据报告：[exp/egorecover_prior_data_budget/v1/data/report.json](exp/egorecover_prior_data_budget/v1/data/report.json)；冻结片段清单：[exp/egorecover_prior_data_budget/v1/data/manifest.json](exp/egorecover_prior_data_budget/v1/data/manifest.json)。

## 2026-09-29--15：52：P D组扩量训练开始/恢复

- [exp/egorecover_prior_data_budget/v1/train/D_s62](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=0，seed=62，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_data_budget/v1/train/D_s62/experiment.json](exp/egorecover_prior_data_budget/v1/train/D_s62/experiment.json)；初始权重：[exp/egorecover_prior_data_budget/v1/train/D_s62/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_data_budget/v1/train/D_s62/initial.pt)。

## 2026-09-29--15：52：P D组扩量训练开始/恢复

- [exp/egorecover_prior_data_budget/v1/train/D_s64](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=0，seed=64，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_data_budget/v1/train/D_s64/experiment.json](exp/egorecover_prior_data_budget/v1/train/D_s64/experiment.json)；初始权重：[exp/egorecover_prior_data_budget/v1/train/D_s64/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_data_budget/v1/train/D_s64/initial.pt)。

## 2026-09-29--15：52：P D组扩量训练开始/恢复

- [exp/egorecover_prior_data_budget/v1/train/D_s63](docs/release/migration.md#为什么选择这些checkpoint)；base=constant_velocity，geometry=1/fk=0，seed=63，从 step0 开始。仅使用 train 身体状态训练。
- 运行配置：[exp/egorecover_prior_data_budget/v1/train/D_s63/experiment.json](exp/egorecover_prior_data_budget/v1/train/D_s63/experiment.json)；初始权重：[exp/egorecover_prior_data_budget/v1/train/D_s63/initial.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_data_budget/v1/train/D_s63/initial.pt)。

## 2026-09-29--16：06：P D组扩量阶段性结论（训练未结束）

- 数据审计全部通过：192 train take、562片段、187.33分钟；三个种子仍在GPU4/5/6训练。当前各seed按Body MPJPE选中的dev结果均值：Body 34.346mm、PA 10.707mm、统一体型 14.119mm、局部旋转 2.470°。
- 相对旧48take/2400步基线（Body35.939、PA11.755、统一体型16.374mm、局部旋转2.670°），目前三个seed的Body均改善，其他关键单步指标均值也改善。阶段性结果支持继续扩大数据和训练预算；尚未完成9600步及最终统计，不能分别归因于数据量或步数，holdout未验证。
- 最终结论会在队列结束后自动记录；数据审计：[exp/egorecover_prior_data_budget/v1/data/report.json](exp/egorecover_prior_data_budget/v1/data/report.json)；当前队列：[exp/egorecover_prior_data_budget/v1/queue.json](exp/egorecover_prior_data_budget/v1/queue.json)；完整曲线：[seed62](exp/egorecover_prior_data_budget/v1/train/D_s62/selection.json)、[seed63](exp/egorecover_prior_data_budget/v1/train/D_s63/selection.json)、[seed64](exp/egorecover_prior_data_budget/v1/train/D_s64/selection.json)。

## 2026-09-29--16：09：P D组扩量单组完成

- [exp/egorecover_prior_data_budget/v1/train/D_s63](docs/release/migration.md#为什么选择这些checkpoint)；选中 step7500；dev next-frame FK=34.102 mm。
- 1秒自反馈 FK=273.794 mm；完整指标/逐take结果：[exp/egorecover_prior_data_budget/v1/train/D_s63/report.json](exp/egorecover_prior_data_budget/v1/train/D_s63/report.json)；选点曲线：[exp/egorecover_prior_data_budget/v1/train/D_s63/selection.json](exp/egorecover_prior_data_budget/v1/train/D_s63/selection.json)；所选权重：[exp/egorecover_prior_data_budget/v1/train/D_s63/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_data_budget/v1/train/D_s63/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--16：10：P D组扩量单组完成

- [exp/egorecover_prior_data_budget/v1/train/D_s62](docs/release/migration.md#为什么选择这些checkpoint)；选中 step9400；dev next-frame FK=34.072 mm。
- 1秒自反馈 FK=305.149 mm；完整指标/逐take结果：[exp/egorecover_prior_data_budget/v1/train/D_s62/report.json](exp/egorecover_prior_data_budget/v1/train/D_s62/report.json)；选点曲线：[exp/egorecover_prior_data_budget/v1/train/D_s62/selection.json](exp/egorecover_prior_data_budget/v1/train/D_s62/selection.json)；所选权重：[exp/egorecover_prior_data_budget/v1/train/D_s62/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_data_budget/v1/train/D_s62/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--16：14：P D组扩量单组完成

- [exp/egorecover_prior_data_budget/v1/train/D_s64](docs/release/migration.md#为什么选择这些checkpoint)；选中 step8400；dev next-frame FK=34.122 mm。
- 1秒自反馈 FK=267.374 mm；完整指标/逐take结果：[exp/egorecover_prior_data_budget/v1/train/D_s64/report.json](exp/egorecover_prior_data_budget/v1/train/D_s64/report.json)；选点曲线：[exp/egorecover_prior_data_budget/v1/train/D_s64/selection.json](exp/egorecover_prior_data_budget/v1/train/D_s64/selection.json)；所选权重：[exp/egorecover_prior_data_budget/v1/train/D_s64/prior.pt](https://huggingface.co/sxhkk/EgoRecover/resolve/8b8190b6964f7c257346cbd65d31010a48007f28/exp/egorecover_prior_data_budget/v1/train/D_s64/prior.pt)。
- dev 用于选模，holdout 未使用；计算量与训练耗时记录在 report.json 的 compute 字段。

## 2026-09-29--16：14：P D组扩量实验结论

- D组Body MPJPE 34.099±0.025 mm，PA 10.552 mm，统一体型 14.054 mm。
- 相对原48take/2400步：Body变化-1.841 mm，逐take配对95%区间[-2.389,-1.343]；三个seed差值-1.928/-1.646/-1.947 mm。
- 三个seed均改善且配对区间低于0：达到。
- 本轮同时增加take、片段与训练步数，只能评价组合收益，无法拆分各因素贡献；dev参与多轮选模，holdout和接入G收益尚未验证。
- 结果：[exp/egorecover_prior_data_budget/v1/RESULTS.md](exp/egorecover_prior_data_budget/v1/RESULTS.md)；统计：[exp/egorecover_prior_data_budget/v1/summary.json](exp/egorecover_prior_data_budget/v1/summary.json)。

## 2026-09-29--16：26：P实际历史适应性验证设计（冻结原E7，新旧P与CV/Hold配对）

- D组已完成：Body 34.099±0.025mm、PA10.552mm、统一体型14.054mm。固定CV残差、geometry=1、FK=0、无显式速度；下一轮评估其处理模型历史的能力，不更新P/G。
- 历史来源固定原E7 EMA网络，Euler10、最多80帧因果观测窗口，每帧只提交末帧，不回改过去结果；与已有单帧HistoryUniEgoMotion适配器明确区分。新旧P各3个已选seed加CV/Hold，共8个预测器读取同一历史。
- 12个dev take×clean/freeze_3s/drift_0p03mps×3个E7采样draw，共108条计划轨迹；主评估t40…199，匹配GT对照，保留holdout。主比较为clean新P−旧P及新P−CV，以take为单位配对统计；另测历史质量、速度抖动及故障恢复阶段。
- 六个P checkpoint哈希及dev隔离已核对。当前完成设计和配置，尚未实现新因果窗口缓存器、未启动该轮评估，尚无预测历史适应性实测结论。
- 方案：[P实际历史适应性](docs/experiments/p-real-history.md)；执行配置：[egorecover_prior_real_history_v1.json](config/egorecover_prior_real_history_v1.json)；D组结果：[exp/egorecover_prior_data_budget/v1/RESULTS.md](exp/egorecover_prior_data_budget/v1/RESULTS.md)。

## 2026-09-29--16：35：新P实际历史评估方案简化（仅替换历史输入，复用GT指标）

- 按用户要求取消本轮旧P、常速度、保持基线，只评估D组新P的三个已选seed；不训练P/G。核心比较为同一个P在E7预测历史与GT历史上的误差差值。
- 主评估改为原12个dev take、t20…199，与已有D组指标完全对齐。三个dev_prior.pt逐帧产物哈希与2160目标帧覆盖已核对，GT参照直接复用；t40…199及t100…199辅助子集从已有产物筛选，不重复GT推理。
- 已有GT参照：Body34.099mm、PA10.552mm、统一体型14.054mm。结果将量化输入历史变化造成的退化，不据此声称新P在E7历史上优于其他方法。
- 本次更新方案和配置，尚未启动评估。方案：[新P实际历史适应性](docs/experiments/p-real-history.md)；配置：[egorecover_prior_real_history_v1.json](config/egorecover_prior_real_history_v1.json)。

## 2026-09-29--17：09：新P实际历史评估启动（冻结E7，复用GT对照）

- 按用户指令启动：GPU4–7，先准备观测和2个train take工程检查，再并行生成12dev×3变体×3采样的因果E7历史，仅评估新P三个已选seed。旧P/CV/Hold不重跑，不训练P/G。
- 队列：[exp/egorecover_prior_real_history/v1/queue.json](exp/egorecover_prior_real_history/v1/queue.json)；控制日志：[exp/egorecover_prior_real_history/v1/controller.log](exp/egorecover_prior_real_history/v1/controller.log)；计划：[exp/egorecover_prior_real_history/v1/plan.json](exp/egorecover_prior_real_history/v1/plan.json)。结束后自动写入比较结论与结果链接。

## 2026-09-29--17：15：新P实际历史正式评估（GPU4–7，原E7因果历史）

- 数据准备完成；2个train take的真实E7 EMA/SMPL GPU工程检查通过，覆盖短窗口、80帧窗口和滑窗起点变化。6项新增检查及原7项CV检查通过，包含部分历史断点恢复逐项一致。
- 四个正式分片已在GPU4/5/6/7启动，每片3个dev take×3变体×3个E7 draw，共108条轨迹。仅评估新P seeds62/63/64，复用已有GT逐帧结果；当前尚无正式效果结论。
- GPU检查：[exp/egorecover_prior_real_history/v1/smoke/report.json](exp/egorecover_prior_real_history/v1/smoke/report.json)；队列：[exp/egorecover_prior_real_history/v1/queue.json](exp/egorecover_prior_real_history/v1/queue.json)；控制日志：[exp/egorecover_prior_real_history/v1/controller.log](exp/egorecover_prior_real_history/v1/controller.log)。

## 2026-09-29--17：16：新P实际历史适应性结论（E7历史对照已有GT指标）

- clean：新P Body 213.535mm，原GT历史 34.099mm，变化+179.436mm（+526.2%），按take配对95%区间[+122.537,+257.145]；PA 66.087mm，统一体型 210.949mm。
- freeze_3s：新P Body 214.560mm，原GT历史 34.099mm，变化+180.462mm（+529.2%），按take配对95%区间[+123.776,+257.743]；PA 68.116mm，统一体型 211.772mm。
- drift_0p03mps：新P Body 215.065mm，原GT历史 34.099mm，变化+180.966mm（+530.7%），按take配对95%区间[+124.474,+257.987]；PA 66.555mm，统一体型 212.454mm。
- 本轮只量化同一个新P更换历史输入后的变化，没有重新训练或评估旧P/CV/Hold；不能据此判断预测历史上替代方法的优劣。
- 历史来源为冻结原E7的因果窗口推理；不是训练后G闭环，且dev已参与先前选模，holdout未使用。
- 结果：[exp/egorecover_prior_real_history/v1/RESULTS.md](exp/egorecover_prior_real_history/v1/RESULTS.md)；完整统计：[exp/egorecover_prior_real_history/v1/summary.json](exp/egorecover_prior_real_history/v1/summary.json)。

## 2026-09-29--17：25：新P实际历史结果分析（先核查E7历史质量，再判断P适配）

- 本轮评估已完成：clean Body34.099→213.535mm，PA10.552→66.087mm，统一体型14.054→210.949mm。三个P seed均明显退化；不能据GT历史成绩宣称已经适应模型生成历史。
- 输入端诊断：clean的20帧E7历史自身平均Body187.205mm、根位置173.106mm、根相对70.140mm。历史本身误差很大；这些值对齐历史各自时刻GT，不能与下一帧分数相减后解释为P独自产生的误差。
- 排除startup混入后t40…199的P Body214.434mm；完整80帧E7窗口子集t100…199为218.379mm。大误差不是只发生在启动过渡。统一体型误差仍为210.949mm，单纯体型差异不足以解释退化。
- 结论：当前配置尚不能证明实际历史适应性。优先审计因果E7窗口输入、地面/坐标与历史重建质量，再决定是否进行train-only GT/预测历史混合适配；本轮不据此否定已验证的GT历史P收益，也不自动启动新训练。
- 正式结果：[exp/egorecover_prior_real_history/v1/RESULTS.md](exp/egorecover_prior_real_history/v1/RESULTS.md)；输入历史诊断：[exp/egorecover_prior_real_history/v1/history_input_summary.json](exp/egorecover_prior_real_history/v1/history_input_summary.json)。

## 2026-09-29--17：28：E7历史来源审计（原权重保留，推理与评价流程存在多项变化）

- 代码核对确认：原E7网络、EMA权重及Euler10保留；本轮改为逐帧因果滑窗只取末帧、每个时刻重采样、启动头高估计地面、固定startup beta用于SMPL评价，以及MotionCodec逐步planar参考投影。评价数据/关节集合/时间统计也与原E7默认流程不同。
- 原E7数据路径使用source floor_height（为0时由GT脚高度回退），v4_beta重建使用整段预测beta均值；原普通MPJPE不等于PA，不能把差异简单解释为GT beta或PA对齐。
- 此前称为原E7历史不够准确，应称原E7权重的修改版因果历史生成流程。当前213.535mm是该流程输入下的新P下一帧误差，187.205mm是输入历史质量；均不能当作原E7复现成绩，不能把退化全部归因于P。
- 已有检查验证当前实现内部性质，尚无原E7完整流程的同片段数值对照。先复现原流程，再拆分滑窗/地面/体型/坐标影响；本次仅审计和修订解释，未启动新训练。
- 详细差异：[实际历史实验审计](docs/experiments/p-real-history.md)；本轮原始结果：[exp/egorecover_prior_real_history/v1/RESULTS.md](exp/egorecover_prior_real_history/v1/RESULTS.md)。

## 2026-09-29--17：40：原E7仓库克隆与结果核查（保留独立源码基线）

- 已从 [sxh-kk/UEM-update](https://github.com/sxh-kk/UEM-update) 克隆至 `/gaozt-test1/sxh/UEM-update-original`；HEAD `156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08` 与 EgoRecover 提取来源一致，工作树干净。原 E7 配置、训练和评估入口均存在。
- 上游完整SMPL22报告：Recon E7 Body/PA/Root/Head 为90.73/54.20/71.58/41.70mm，原论文Diffusion为97.51/52.61/79.90/53.54mm。来自相同EE4D val 256窗口的已有报告，本次未复跑；支持原E7精度接近原版UniEgoMotion，但不能直接与P-dev因果历史及下一帧指标比较。
- Git仓库没有训练权重和原始评估产物；本地已有 `exp/e7/last.ckpt`，并非此次下载。尚未启动新训练或精度评估。
- 配置、代码与结果链接汇总：[原E7源码核查](docs/reference/e7.md#原仓库重新克隆核查2026-09-29)；上游原表：[完整22关节结果](https://github.com/sxh-kk/UEM-update/blob/156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08/result_400M.md#4-完整-22-关节论文指标)。

## 2026-09-29--21：03：E7最小适配四组对照启动（窗口×地面，冻结权重）

- 按用户最终要求收敛为四组：E0原整窗＋原地面，E1因果末帧滑窗＋原地面，E2原整窗＋启动估计地面，E3因果末帧滑窗＋启动估计地面。体型、解码、观测编码和短窗padding均恢复原E7；不再执行80配置方案。
- 同一原E7 EMA/Euler10、12个dev take的clean观测、3个采样draw，统一评价20…199帧的原始世界坐标SMPL22。报告窗口影响、地面影响及交互量；其余旧改动不在本轮单独估计，不使用holdout，不训练P/G。
- 6项测试通过，覆盖因果性、补齐、地面通道隔离、目标覆盖、噪声配对、结果汇总与交互计算。首次真实片段核对的轨迹编码完全一致；SMPL FK与保存标签存在约0.55mm差异，因此坐标还原检查改为dense关节往返，FK差异单独报告；完整检查尚待队列完成。
- 用户授权执行后，控制器PID3815752已启动；GPU4–7空闲。当前等待数据准备PID3814471完成，随后自动运行verify、两条train片段smoke、四卡正式评估和总结。只有状态进入formal_evaluation才表示正式GPU评估已开始；本轮没有梯度训练。
- 方案：[四组设计](docs/experiments/e7-minimal-ablation.md)；配置：[JSON](config/e7_minimal_ablation_v1.json)；状态：[queue.json](exp/e7_minimal_ablation/v1/queue.json)；控制日志：[controller.log](exp/e7_minimal_ablation/v1/controller.log)。完成后自动写入结论和RESULTS.md链接。

## 2026-09-29--21：05：E7四组对照执行中断

- smoke failed with exit 1; see jobs/smoke.log
- 状态：[queue.json](exp/e7_minimal_ablation/v1/queue.json)；[运行目录](docs/release/migration.md#为什么选择这些checkpoint)。未生成完整四组结论。

## 2026-09-29--21：07：E7四组对照执行中断

- smoke failed with exit 1; see jobs/smoke.log
- 状态：[queue.json](exp/e7_minimal_ablation/v1/queue.json)；[运行目录](docs/release/migration.md#为什么选择这些checkpoint)。未生成完整四组结论。

## 2026-09-29--21：09：E7四组正式对照运行（GPU4–7，工程检查通过）

- 已修复上游缺失ipdb依赖和EMA加载顺序问题：先载入原权重及EMA，再冻结参数。上游仓库保持干净，网络、采样器及原SMPL解码未修改。数据缓存保留原生产代码身份；修改验证/推理代码不要求重读原始数据，模型输出仍严格校验当前代码身份。
- 7项自动测试通过；2条train片段×4窗口的原版输入编码最大差为0，dense世界坐标往返最大误差0.00730mm；2条train片段的四组真实E7 GPU试跑全部完成。SMPL FK与保存标签的小差异单列，不作模型精度结论。检查：[verification.json](exp/e7_minimal_ablation/v1/verification.json)；试跑：[smoke.log](exp/e7_minimal_ablation/v1/jobs/smoke.log)。
- 正式状态为formal_evaluation。控制器PID3819864；GPU4/5/6/7上的worker PID分别3820459/3820460/3820461/3820462，每卡负责3个dev take，并运行E0–E3及三个draw。已进入真实模型推理；本轮冻结E7，不更新权重。
- 数据核对：12个dev take和2个smoke take的source floor均非零，本轮不会触发GT脚部地面回退。当前尚无完整四组结果，结束后自动汇总配对差值和交互量，并追加结论。
- 状态：[queue.json](exp/e7_minimal_ablation/v1/queue.json)；进度：[shard0](exp/e7_minimal_ablation/v1/jobs/shard0.log)、[shard1](exp/e7_minimal_ablation/v1/jobs/shard1.log)、[shard2](exp/e7_minimal_ablation/v1/jobs/shard2.log)、[shard3](exp/e7_minimal_ablation/v1/jobs/shard3.log)。

## 2026-09-29--21：12：E7最小适配四组对照完成（exp/e7_minimal_ablation/v1）

- window：2个配对背景，B−A范围[+9.317, +15.296] mm；方向一致。具体方向、分阶段数值及区间见 [summary.json](exp/e7_minimal_ablation/v1/summary.json)。
- floor：2个配对背景，B−A范围[+112.554, +118.533] mm；方向一致。具体方向、分阶段数值及区间见 [summary.json](exp/e7_minimal_ablation/v1/summary.json)。
- 滑窗×地面交互：-5.979 mm，95%区间[-9.858884811401367, -2.2593626976013184]。正值表示估计地面在滑窗条件下带来更大的误差增量。

这些是冻结E7推理因素的结果；尚未评价P或重新训练G。

- 结果：[RESULTS.md](exp/e7_minimal_ablation/v1/RESULTS.md)；完整配对：[summary.json](exp/e7_minimal_ablation/v1/summary.json)。

## 2026-09-29--21：21：E7四组结论分析（地面估计主导位置退化，滑窗损害时间连续性）

- 四组均已完成：E0原整窗/原地面59.958mm，E1滑窗/原地面75.254mm，E2原整窗/估计地面178.491mm，E3滑窗/估计地面187.808mm。12个dev take、3个draw；冻结E7，未训练模型。
- 地面估计为本轮位置误差主要来源：整窗增量+118.533mm（95%区间[51.604,204.718]），滑窗增量+112.554mm（[46.028,200.261]）；12/12个take退化。去掉CPR异常片段后仍分别+82.896/+76.279mm。
- 启动地面估计绝对偏差平均248.6mm；CPR启动头高约0.71m，统一平均头高约1.40m，估计地面低693.5mm，使因果Body42.727→554.314mm。滑窗组根位置z误差13.780→151.046mm，证实竖直位置是主要受损部分。
- 原地面下改滑窗增加15.296mm（95%区间[9.993,21.131]），12/12个take退化；完整80帧阶段仍增加16.026mm。速度误差177.388→592.463mm/s（3.34倍），不能进一步区分独立噪声、末帧位置和因果观测的贡献。
- 三个draw均保持E0<E1<E2<E3；相同窗口末帧预测最大差0.00173mm。交互为−5.979mm，未显示额外正向放大。E0与原仓库90.73mm评价片段不同，不直接比较；当前结果不能推算修复地面后的P成绩。
- 下一步建议：先用E1原地面流程复测既有P，排除错误地面输入，再研究部署地面标定和跨窗连续性。本次仅分析，没有自动开启新实验。
- 详细结论：[RESULTS.md](exp/e7_minimal_ablation/v1/RESULTS.md)；逐take地面、逐seed及坐标诊断：[analysis_diagnostics.json](exp/e7_minimal_ablation/v1/analysis_diagnostics.json)。

## 2026-09-29--21：38：历史条件G重新训练的理论分析（文献依据、输入修正与四组训练建议）

- 核查EgoForce、HMD²、MotionStreamer、WarmPrior、PrediFlow及EgoAllo原文。支持专门训练因果历史条件G，并在训练中接触预测历史；P作为Flow源可能改善有限NFE下的修正，但不保证最终MPJPE优于同条件Gaussian。详细引用和适用边界见下方研究文档。
- 已完成E0–E3使用冻结原E7，不能据此否定重新训练后的HistoryUniEgoMotion。正确地面下因果75.254mm可作为诊断参照；离线59.958mm使用未来观测，不能预先承诺在线模型超过它。
- 训练前优先修复地面：平均头高公式消去了启动真实高度差异，增加训练量不保证补回信息。当前G仅接收当前原始观测，建议补入短期因果图像/头轨迹缓存；该架构改动尚未实现。P保持常速度残差/FK=0，但须按新坐标与地面协议重评，必要时重新训练同结构P。
- 建议先从E7初始化全参数训练G；四组为Gaussian/History源×GT/混合预测历史。各组条件、数据、P和预算一致，混合组首轮使用同一冻结教师缓存，后续再研究自身历史刷新。预测历史比例与低NFE收益均待验证。
- 数据划分风险：现12个dev take来自官方train；全量预训练可能与其重叠，且须核查E7初始化的训练覆盖。最终结论应来自预训练也未见过的独立take划分。
- 本次仅完成代码核查、理论分析与方案记录，未修改训练架构、未启动预训练。完整文档：[历史条件G预训练分析](docs/design/g-pretraining-review.md)。

## 2026-09-29--21：58：G重新训练四组方案（Flow起点×预测历史适配）

- 四组固定为T0 Gaussian＋GT、T1 History＋GT、T2 Gaussian＋混合历史、T3 History＋混合历史；所有组均读取同样的身体历史、P预测及过去/当前观测。主要评价统一为自身预测历史闭环，不跨不同历史条件直接比成绩。
- 共同设置：正确地面、统一planar参考和预测启动体型、20帧历史/10Hz、E7 EMA初始化后全参数训练G；P采用常速度残差、geometry=1/FK=0，经新协议校准后冻结。新增过去观测分支；G损失固定表示＋dense geometry=1/FK=0。
- 公平性：同目标帧配对GT/教师预测历史，替换历史后重算P及参考编码；T2/T3使用同一冻结因果E7教师缓存，混合概率先0、再线性增加到50%。现有replay会抽取不同目标样本，不能直接用于该配对设计。
- 初版预算每组24,000步、有效batch64、三个训练seed，共四种配置12次训练；共享数据/P校准/缓存成本单列。GPU4–7作为后续调度方案，时间需吞吐试跑后估计。当前只设计，未启动训练或占用GPU。
- 数据计划扩大到可审计官方train，四组同一清单；独立dev/留出集需核查E7和本项目历史使用范围后冻结。旧12take及75.254/59.958mm保留为诊断，不能直接充当新数据基线。
- 主要看Body MPJPE、G相对P的纠错和三seed配对差值；辅以PA/root、速度误差、短闭环和低NFE。分别报告两个因素及交互，不预设T3必胜。完整设计、实现前置项与产物规划：[G重新训练四组方案](docs/experiments/g-pretraining-four-way.md)。

## 2026-09-29--22：28：G四组实验队列启动（1608训练take，GPU4–7先生成配对历史）

- 已按用户授权启动完整队列，控制器PID3897298；GPU4/5/6/7的数据worker PID3897300/3897307/3897309/3897311。当前阶段为preparing_paired_histories，正在生成本轮正确地面的教师历史；正式G权重更新尚未开始，不是在等待其他项目结束。
- 冻结1,608个train take、4,369个20秒片段（24.27小时），保留原192train take；224个dev、222个本轮留出take，各1片段。dev预先按任务选16take进行训练选点；原E7曾使用官方val验证，本轮留出不宣称为原E7从未接触的全新测试集。完整[清单](exp/egorecover_g_pretraining/v1/data/manifest.json)与[划分](exp/egorecover_g_pretraining/v1/data/split.json)。
- 已实现过去观测的显式输入、同目标帧GT/预测历史配对、重新编码目标与P条件、统一地面/启动体型、梯度累积、EMA、恢复及闭环选点。G固定geometry=1/FK=0；四组配置和源码快照存于本轮目录。
- 27项测试通过；3个真实片段通过原生解码、参考系往返与SMPL资产核对，原生解码最大绝对差1.19e−7m，SMPL源标签审计最大差1.72mm；四组各完成2步反传及短闭环试跑。发现并修复“逐帧数组混入标量汇总”的错误，失败试跑保留于verification/*_attempt1。试跑只证明执行正确，不作为模型精度结论。[检查报告](exp/egorecover_g_pretraining/v1/verification/smoke.json)。
- 队列自动继续：四卡配对缓存→共享P在新协议下最多2,400步校准（含原权重选点）→T0/T1/T2/T3在GPU4/5/6/7运行，每组24,000步、有效batch64，seed62/63/64三轮共12次训练→冻结选点后评价留出集并汇总。缓存单片段实测约4–6秒，四卡预计约1.5–2小时，含冷加载/资源波动；这不是整个训练完成时间。
- 状态：[queue.json](exp/egorecover_g_pretraining/v1/queue.json)；控制日志：[controller.log](exp/egorecover_g_pretraining/v1/controller.log)；四卡缓存日志：[GPU4](exp/egorecover_g_pretraining/v1/jobs/prepare_gpu4.log)、[GPU5](exp/egorecover_g_pretraining/v1/jobs/prepare_gpu5.log)、[GPU6](exp/egorecover_g_pretraining/v1/jobs/prepare_gpu6.log)、[GPU7](exp/egorecover_g_pretraining/v1/jobs/prepare_gpu7.log)。结束后自动写入实验结论及RESULTS.md链接，不向LOG逐次记录选点。

## 2026-09-30--01：13：G四组正式梯度训练阶段启动

- 配对数据准备、坐标核对、真实片段试跑及共享P协议校准已通过。T0/T1/T2/T3按GPU4/5/6/7调度，三个seed逐轮运行，每组24,000步。
- 进度：[queue.json](exp/egorecover_g_pretraining/v1/queue.json)；共享P校准：[report.json](exp/egorecover_g_pretraining/v1/prior/report.json)。此时尚无四组精度结论。

## 2026-09-30--01：24：G四组早期训练分析（P校准小幅改善，G尚无正式闭环结论）

- 数据缓存与共享P校准已完成；四组G于01：13进入正式训练，目前seed62约1080–1280/24,000步，四个worker均在运行。后续seed63/64与留出集评价尚未开始。
- 已有结论：相同16个dev take、每take固定18个GT历史目标上，P Body从26.908降至25.844mm，改善1.064mm（3.96%），12/16个take改善。表明该结构可在新协议下取得小幅校准收益；选点使用了这些dev数据，不是独立测试或实际预测历史成绩。旧34.099mm来自不同数据与协议，不能据此宣称34→26mm的直接提升。
- G训练目标下降：同预算700–900步日志均值Gaussian约0.227、History约0.191；它们是不同源输入下的训练目标，不能转换成MPJPE或用于最终排名。尚未到2,000步首次正式闭环评估，smoke目录数值不作实验结论。
- 当前T2/T3混合概率仍为0，前2,000步与对应GT组执行相同训练条件；2,000–6,000步才逐渐升至50%。因此现在无法评价预测历史训练的收益或两因素交互。继续按既定计划训练，待共同预算的闭环Body、P→G纠错及短时反馈指标后分析。
- 状态与分析快照：[本次分析](exp/egorecover_g_pretraining/v1/analysis/20260930-0124-progress.json)；P结果：[report.json](exp/egorecover_g_pretraining/v1/prior/report.json)；[逐take校准记录](exp/egorecover_g_pretraining/v1/prior/selection.json)。本次未改变训练配置或运行进程。

## 2026-09-30--10：08：G四组阶段结论（前两seed完成，T3同预算最优但闭环误差仍偏大）

- 8/12次训练完成，seed62/63四组均完成24,000步；seed64约11240–11840步。四卡仍在训练，222个留出take的正式评价尚未开始。
- 相同16个dev take、同预算24,000步、EMA/Euler10、draw1062、真实自身历史闭环t40…199：T0/T1/T2/T3 Body两seed均值162.05/146.28/140.85/129.54mm。T3比T0降低32.50mm（20.06%）；两seed上各项配对均值方向一致，但T2−T0的take配对区间仍包含0，结论限于开发集趋势。
- 组合未显示额外超加性收益：交互+4.46mm。按各组dev最优点比较，T2/T3为128.42/126.28mm，差距较小且seed方向不一致；不能把同预算末点排名等同于最终选中模型结论。
- 当前精度仍不理想：T3 Body约129.54mm，root106.75mm，PA75.94mm；在其自身历史上，P约132.12mm，G仅平均修正2.58mm。不能将GT历史P的25.84mm与闭环G直接相减。T3较T1的收益主要体现于位置，PA几乎相同。
- 早期部分GT训练组闭环发生严重位置发散，并伴随PA非有限值；后期末点未见同量级异常，仍需正式评价稳定性。旧E7的75.25mm不是本轮同数据参照，暂不直接比较。本次不改变训练配置。
- 详细阶段结论：[分析报告](exp/egorecover_g_pretraining/v1/analysis/20260930-1008-dev-summary.md)；[配对统计](exp/egorecover_g_pretraining/v1/analysis/20260930-1008-dev-summary.json)。预计剩余训练约2小时，最终评价另计。

## 2026-09-30--10：24：G精度目标与改进方向（论文量级参考，优先观测位置约束与自身历史训练）

- 建议闭环Body先≤100mm/PA≤60mm，主要目标Body80–90mm/PA45–55mm，进阶Body70–80mm。这是项目目标，不是通用合格线；是否优于原E7仍须本轮相同因果协议评价，不能用旧12take的75.25mm直接排序。
- 原文核查：UniEgoMotion100mm/PA53mm；HMD²低延迟91.9mm（Nymeria、约0.17s延迟、设备头参考）；EgoForce在线73mm（2026预印本，额外可见手腕控制）。数据、输入和评价差异决定这些数值只能用于量级参考。MotionStreamer用于自身预测历史训练的启发，其压缩器24.89mm不作为本任务目标。
- 当前T3同预算两seed均值Body129.54/root106.75/PA75.94mm，G在相同自身历史上仅比P改善2.58mm。建议先研究当前头观测确定世界参考，以及训练中使用G自身生成历史；其次测试显式残差纠错、诊断dense监督与SMPL评价差异。P暂保留常速度残差/FK=0。
- 后续可做“观测头参考×固定教师/自身G历史”四组，当前仅为建议；未修改正在运行的训练或启动新任务。引用、数值边界和代码依据见[G精度目标与下一步改进](docs/design/g-targets-and-next-steps.md)。

## 2026-09-30--12：50：G十二次训练完成（留出集首轮结果仍未超过因果E7参照）

- T0–T3×seed62/63/64全部完成24,000步，最后训练报告于12:11:03写出。已完成因果E7参照；GPU4–7当前并行评估seed62四组，控制器和四个评价进程正常运行。
- 首轮完整222take结果：seed62、last权重、NFE10、draw1062、自身历史t40…199，T0/T1/T2/T3 Body分别160.004/151.543/139.347/134.727mm。T3相对T0降低25.276mm（15.8%），尚不是三seed、多draw或dev选中best模型的最终结论。
- 同一批222take、t40…199、draw1062的冻结因果E7参照：共同启动体型115.018mm；原生窗口体型114.635mm。当前T3末点比共同体型E7高19.709mm，首轮仍未证明超过原E7。两种方法历史上下文和计算结构不同，后续应同时报告延迟与计算量。
- 每组计划评价last的NFE10×3draw及best的NFE10/1/3/5×3draw，共15个完整配置，另有固定历史诊断，三个训练seed逐轮运行。首个NFE10配置实测25.0–26.6分钟；完整队列仍需多小时，初估剩余约10–15小时，低NFE耗时和GPU共享会影响估计。
- 依据：[实时队列](exp/egorecover_g_pretraining/v1/queue.json)、[E7参照](exp/egorecover_g_pretraining/v1/evaluation/e7_baseline.json)、[T0评价日志](exp/egorecover_g_pretraining/v1/jobs/eval_T0_seed62.log)、[T1评价日志](exp/egorecover_g_pretraining/v1/jobs/eval_T1_seed62.log)、[T2评价日志](exp/egorecover_g_pretraining/v1/jobs/eval_T2_seed62.log)、[T3评价日志](exp/egorecover_g_pretraining/v1/jobs/eval_T3_seed62.log)。本次未改变队列、权重或评价配置。

## 2026-09-30--12：55：按要求切换最小评估（best模型、三训练seed、Euler10、单draw）

- 停止原完整评估控制器及其四个评价worker，保留已完成末点结果；所有12次训练及权重保持不变。
- 新范围：T0–T3×seed62/63/64，仅开发集已选定best权重、222个留出take、Euler10、draw1062；复用原E7共同体型115.018mm参照。取消剩余末点、多draw、低NFE与额外固定历史诊断，不改变模型选点或删减评价take。
- 新增独立评价入口并核对原smoke源码哈希、训练完成状态和12个best权重哈希；语法检查与合成数据汇总检查通过，验证三seed均值/标准差、配对差值和E7差值。训练与原推理代码未修改。
- 最小队列已实际启动：控制器782826；GPU4–7上的seed62 worker为782954/782956/782963/782972，均运行中。按此前每轮25–27分钟估计三轮约1.5小时，预计14:15–14:30附近完成，资源共享会影响时间。
- 结束后自动写入最终结论及RESULTS.md/minimal_summary.json链接；仅标记最小协议完成。当前没有新的best留出精度结果。[冻结协议](exp/egorecover_g_pretraining/v1/evaluation/minimal_plan.json)、[实时队列](exp/egorecover_g_pretraining/v1/queue.json)、[控制日志](exp/egorecover_g_pretraining/v1/minimal_controller.log)、[方案变更](docs/experiments/g-pretraining-four-way.md)。

## 2026-09-30--14：05：G四组最小留出评估完成

- T0 Body 153.907±10.771mm；T1 Body 152.557±6.155mm；T2 Body 136.185±9.246mm；T3 Body 133.856±3.053mm
- T1-T0 -1.350mm，稳定收益证据不足；T2-T0 -17.722mm，支持降低误差；T3-T2 -2.330mm，稳定收益证据不足；T3-T1 -18.702mm，支持降低误差；T3-T0 -20.051mm，支持降低误差
- 原E7共同体型参照115.018mm；best权重、三训练seed、Euler10、单draw1062。
- 仅完成缩减后的评价范围；[完整结论](exp/egorecover_g_pretraining/v1/RESULTS.md)；[配对统计](exp/egorecover_g_pretraining/v1/minimal_summary.json)。

## 2026-09-30--14：42：G最小评估结论分析（混合历史有效，History源收益不稳定，位置精度仍落后E7）

- 最小评估于14:05完成：222take、dev选中best、三训练seed、单draw1062、Euler10。T0/T1/T2/T3 Body153.907/152.557/136.185/133.856mm，原E7共同体型115.018mm。
- 混合历史收益在两种源下均跨三seed一致：T2−T0 −17.722mm，T3−T1 −18.702mm。History源增益分别仅−1.350/−2.330mm且种子方向不一致；T3对T2的take区间虽略低于0，仍不足以说明稳定的跨训练种子收益。
- T3比T0改善20.051mm（13.03%），176/222take改善；比E7高18.838mm（16.38%），仅39/222take优于E7。此前开发集末点约20%的结论不能替代本次留出best结果。
- T3速度误差281.28mm/s，E7滑窗711.57mm/s，降低约60.5%；四组G速度误差相近，该优势不能单独归因于History源。不能据此证明没有过度平滑或延迟。T3自身历史P→G仅改善2.470mm；位置和姿态精度仍需改进。
- 下一步建议保留P常速度残差/FK=0及混合历史训练，以T2作为强对照，优先验证头观测位置约束和G自身历史适配。本次只分析，未开启新实验或恢复取消的评价项目。
- 完整解释与边界：[结论分析](exp/egorecover_g_pretraining/v1/analysis/minimal-evaluation-conclusions.md)；[主结果](exp/egorecover_g_pretraining/v1/RESULTS.md)；[配对统计](exp/egorecover_g_pretraining/v1/minimal_summary.json)。

## 2026-09-30--14：49：后续G修改方向与相关工作核查（观测上下文、设备参考与自身历史适配）

- 核查HMD²、EgoAllo、EgoForce、MotionStreamer、DAgger和Diffusion Forcing原文；建议以T2为后续对照，保留P常速度残差/FK=0及预测历史混合训练，不将History起点列为已证实优势。
- 优先诊断E7的21/80帧观测差异，再考虑身体历史20帧与观测memory最长80帧解耦；另测试当前设备观测确定世界参考。两项可组成四组，随后单独测试当前G自身历史缓存适配。
- 代码已确认历史帧Flow时间固定0、历史不参与去噪；这没有显式表达历史不确定性，但尚不能证明模型因此过度依赖历史。历史噪声/可信度建模列为后续独立方向，不宣称已复现完整Diffusion Forcing。
- 各文献机制、可迁移部分、设备偏移/因果输入/测试集复用边界及四组建议已更新至[设计文档](docs/design/g-targets-and-next-steps.md)。本次没有修改模型、启动训练或新增评估。

## 2026-09-30--17：23：观测长度×设备参考四组正式训练启动

- 冻结E7开发集观测诊断：21帧110.334mm，80帧116.380mm；不是新G的成绩。
- R0/R1为预测参考，R2/R3为当前设备planar参考；R0/R2读取21帧，R1/R3最多80帧；全部Gaussian源、混合历史、相同冻结P和E7初始化。
- 四组真实数据反传和96帧闭环试跑通过，坐标往返核验通过。GPU4–7每卡一组，24,000步×三seed；之后仅best/Euler10/draw1062评价。
- [状态](exp/egorecover_observation_reference/v1/queue.json)；[控制日志](exp/egorecover_observation_reference/v1/controller.log)；[配置](exp/egorecover_observation_reference/v1/plan.json)；[诊断](exp/egorecover_observation_reference/v1/diagnostic/report.json)。

## 2026-09-30--17：32：R0–R3训练队列与验证记录（GPU4–7，当前冷加载训练缓存）

- 四组22项接口/坐标测试通过，额外汇总检查验证R0–R3配对因子和三seed统计；四组真实数据2步反传及96帧闭环通过，目标坐标往返最大误差5.96e−8m。
- E7观测诊断21帧110.334mm、80帧116.380mm，13/16个dev take上21帧更好；因此较长观测的收益仍是待验证假设，不能用观测缩短直接解释上一轮G的退化。
- 正式控制器1129454，R0/R1/R2/R3 worker分别1130475/1130483/1130485/1130487，GPU4/5/6/7。进程正在冷加载4,369个训练片段，尚未出现正式progress.json；之后自动运行24,000步×三个seed，并进行最小评价。
- 4–7卡与其他任务共享，未调整其他进程。四组固定同一P/数据/优化预算，新观测接口全部重跑；参考设置只确定设备坐标，不强行令SMPL头关节等于设备中心。
- [实验方案与监控命令](docs/experiments/g-observation-reference.md)；[验证报告](exp/egorecover_observation_reference/v1/verification/report.json)；[状态](exp/egorecover_observation_reference/v1/queue.json)；[控制日志](exp/egorecover_observation_reference/v1/controller.log)。

## 2026-09-30--17：34：R0–R3四卡已进入实际参数更新

- 冷加载已结束，四组seed62均已出现正式梯度更新：R0(GPU4) 120/24000步；R1(GPU5) 100/24000步；R2(GPU6) 120/24000步；R3(GPU7) 100/24000步。当前loss仅用于运行检查，不作精度排名。
- 22项测试、四组真实数据试跑与坐标核验通过。正式训练及后续最小评估自动继续，结论写入LOG并链接结果文件。
- [方案与监控命令](docs/experiments/g-observation-reference.md)；[实时队列](exp/egorecover_observation_reference/v1/queue.json)；[控制日志](exp/egorecover_observation_reference/v1/controller.log)。

## 2026-09-30--17：42：观测长度×设备参考实验队列中断

- R3_seed62 failed (exit=1); see jobs/R3_seed62.log. Other launched jobs may still be running.
- [状态](exp/egorecover_observation_reference/v1/queue.json)。已完成产物保留。

## 2026-09-30--17：43：R3非有限梯度导致首轮中断（保留失败数据，统一检查四组数值稳定性）

- R3 seed62约1400步发生non-finite gradient norm；队列失败。R0/R1/R2也出现较大loss/梯度，已停止本项目三个剩余worker以保留公平对照，其他GPU进程未调整。
- 失败产物保留在exp/egorecover_observation_reference/v1，不计入有效结果。初步检查指向新观测接口的数值稳定性，尚未确认唯一原因；后续修复将使用新版本目录统一重跑。
- 当前4–7卡每卡约7个其他GPU进程，外部PID在容器中不可见，不能确认项目名称；异常日志为梯度非有限，不是CUDA OOM。

## 2026-09-30--17：51：修正E7观测接口并在4–7卡启动v2恢复验证

- 首轮R3约1,400步非有限梯度，R0–R2同步停止；失败记录完整保留。尚未得到四组精度结论。
- v2保留头轨迹→身体分支及最近21帧原位置编码；更早设备轨迹使用相邻增量，新增memory轨迹投影零初始化。四组相同接口；数据、P、学习率、24k×三seed及最小评估范围不变。尚未逐项确认发散原因。
- 24项测试通过；真实4样本R0初始化与旧G最大输出差1.43e−6；短试跑不等于长期稳定，需要超过原失败步数。
- 17:49检查：GPU4–7各有7个其他进程，显存合计约24–29GiB/卡，PID属于当前容器不可见命名空间，无法确认项目名；未操作这些进程。
- [设计与失败说明](docs/experiments/g-observation-reference.md)；[v2状态](exp/egorecover_observation_reference/v2/queue.json)；[恢复日志](exp/egorecover_observation_reference/v2/controller.log)；[接口核验](exp/egorecover_observation_reference/v2/verification/interface_diagnostic.json)。

## 2026-09-30--17：51：观测长度×设备参考四组正式训练启动

- 冻结E7开发集观测诊断：21帧110.334mm，80帧116.380mm；不是新G的成绩。
- R0/R1为预测参考，R2/R3为当前设备planar参考；R0/R2读取21帧，R1/R3最多80帧；全部Gaussian源、混合历史、相同冻结P和E7初始化。
- 四组真实数据反传和96帧闭环试跑通过，坐标往返核验通过。GPU4–7每卡一组，24,000步×三seed；之后仅best/Euler10/draw1062评价。
- [状态](exp/egorecover_observation_reference/v2/queue.json)；[控制日志](exp/egorecover_observation_reference/v2/controller.log)；[配置](exp/egorecover_observation_reference/v2/plan.json)；[诊断](exp/egorecover_observation_reference/v2/diagnostic/report.json)。

## 2026-09-30--19：59：四组G中期开发集结果：长观测暂未改善，设备参考未提高全身精度

- 仍在seed62，四组约2万/2.4万步；seed63/64与222take最终评价未开始。比较使用相同18,000步EMA及16个固定dev take；不能与前轮222take最终成绩直接比较。
- R0（21/预测）Body125.813、Head43.153mm；R1（80/预测）141.914、50.924mm；R2（21/设备）131.510、40.002mm；R3（80/设备）142.237、47.382mm。
- 80帧相对21帧Body增加16.101mm（预测参考）和10.728mm（设备参考）；设备参考21帧下Head改善3.151mm、Body恶化5.697mm。当前不支持两项改动提高全身精度，等待三seed最终评价。
- G在各自历史上的P→G单步改善仅约2–3mm，当前纠错收益有限；不等同于独立P闭环对照。
- 当前优化loss与梯度正常，但早期闭环曾极端发散，数值保留在selection曲线中；尚不能宣称整个训练过程的闭环稳定性得到解决。
- [中期结论与限制](docs/experiments/g-observation-reference.md)；[固定18k统计](exp/egorecover_observation_reference/v2/interim/seed62_step18000.json)。

## 2026-10-01--01：13：G观测长度×设备参考四组评估完成

- R0 Body 131.913±2.465mm；R1 Body 143.894±9.140mm；R2 Body 135.626±3.935mm；R3 Body 139.053±3.682mm
- R1-R0 +11.981mm，一致增加误差；R2-R0 +3.714mm，一致增加误差；R3-R2 +3.426mm，稳定收益证据不足；R3-R1 -4.841mm，稳定收益证据不足；R3-R0 +7.140mm，一致增加误差
- 原E7共同体型参照115.018mm；best权重、三训练seed、Euler10、单draw1062。
- 仅完成缩减后的评价范围；[完整结论](exp/egorecover_observation_reference/v2/RESULTS.md)；[配对统计](exp/egorecover_observation_reference/v2/summary.json)。

## 2026-10-02--02：00：四组G最终结论：短观测基线最好，纠错收益仍小，速度改善但位置落后E7

- 四组×三个seed均完成24,000步训练和222take最小评估，队列10-01 01:13完成。R0/R1/R2/R3 Body：131.913±2.465 / 143.894±9.140 / 135.626±3.935 / 139.053±3.682mm。
- R1−R0 +11.981mm、R2−R0 +3.714mm，三个seed都更差；R3−R0 +7.140mm，三个seed都更差。本实现的80帧与设备参考均未超越21帧预测参考。
- R0较上一轮最佳T3平均改善1.943mm，三seed方向一致，但take配对95%区间[−4.160,+0.231]mm包含0，不能宣称可靠突破；相对原E7 115.018mm仍高16.895mm（14.7%）。
- R0速度误差278.344mm/s，E7为711.570mm/s，降低约60.9%；体现速度一致性改善与位置精度落后的取舍。当前各组P→G单步改善仅2.011–2.650mm，纠错能力仍有限。
- v2最终成绩正常，但部分早期开发checkpoint闭环极端发散，完整曲线保留；GT/固定E7训练历史与实际G历史存在分布差异，dense监督到FK评分也需诊断，尚未确认各自因果贡献。
- 建议下一步冻结R0做固定历史、观测条件及dense/FK误差诊断；尚未启动新实验。仅单draw、已知地面/公共预测体型、重复使用基准；未验证低NFE、实际速度、受损观测或新独立测试集。
- [最终分析](exp/egorecover_observation_reference/v2/analysis/final-conclusions.md)；[成绩表](exp/egorecover_observation_reference/v2/RESULTS.md)；[跨轮配对统计](exp/egorecover_observation_reference/v2/analysis/cross-round-comparisons.json)。

## 2026-10-02--02：12：设计G几何残差×实际R0历史四组对照（参考MotionStreamer与PrediFlow）

- 依据最终R0 131.913mm、P→G仅2.386mm以及80帧/设备参考无收益，固定21帧观测、20帧身体历史和预测planar参考，冻结P。
- S0：完整输出+50%GT/50%E7；S1：P物理状态+几何残差，原历史混合；S2：完整输出+50%GT/25%E7/25%R0历史；S3：几何残差+该混合。比较结构、历史来源与交互。
- 每seed对应R0 best骨干续训12k步×三个seed；绝对组保留输出头，残差组171D头零初始化，学习合法SO(3)/平移及辅助修正并重编码完整243D x0。所有组Gaussian源、geometry1/FK0；残差结构不与History源同时改变。
- R0只在train episode上因果闭环生成固定缓存，GT仍占50%；这是固定R0分布适配，借鉴MotionStreamer暴露偏差与DAgger数据收集思想，未采用完整在线Two-Forward/迭代刷新。PrediFlow提供冻结预测器+残差修正机制，任务/表示/求解器不同。
- 主评估沿用222重复基准、best/Euler10/单draw；新增少量固定dev历史单步纠错、dense/FK误差和模态屏蔽诊断。目标是分清误差原因，不扩大长程评价。
- 当前仅完成文献检索和设计，未创建训练缓存、未改模型训练代码、未启动新实验。
- [完整方案](docs/design/g-residual-history-four-way.md)；[计划配置](config/egorecover_g_residual_history_v1.json)；[MotionStreamer](https://arxiv.org/html/2503.15451v2)；[PrediFlow](https://arxiv.org/html/2512.13903v1)。

## 2026-10-02--23：10：代码与实验产物全量盘点及GitHub/Hugging Face发布规划

- 已完成本地盘点与逐文件去向规划，未创建远程仓库、上传、打包或调整训练进程。当前origin仍为sxh-kk/UEM-update，拟发布的独立公开仓库为sxh-kk/EgoRecover。
- 仓库内14,878个普通文件、4个内部目录软链接，逻辑大小142.27 GiB；exp约106.56 GiB、data约35.54 GiB。软链接不重复遍历；本次报告目录自身排除。外部原E7参考仓库另行统计。
- GitHub候选1,318文件、约64.14 MiB：当前工作树代码、配置、文档、轻量实验结论/统计、冻结源码、必要清单及图表。需按白名单保留原exp相对路径，避免.gitignore造成LOG结果链接遗漏；候选尚需内容审核。
- 模型checkpoint共202个、37.69 GiB；续训状态77个、51.46 GiB。推荐首发R0三seed best加共享校准P，共1.096 GiB；24个正式T/R组G best共8.630 GiB。当前G配套的校准P不能用原D_s62 P直接替换。
- 35.53 GiB原始处理数据优先官方引用；10.39 GiB缓存和5.64 GiB预测/评估tensor需要派生数据审核，部分文件含GT。SMPL-X模型资产排除上传；原E7 EMA保存在optimizer状态，不能直接删optimizer作为EMA推理导出。
- 尚未进行全量大文件内容哈希/去重、实际权重导出、异机恢复或上传验证。强凭证模式扫描未发现匹配，但不覆盖全部二进制/Git对象历史；README、许可来源、绝对路径和原身份哈希需要发布前整理。新S0–S3仍仅设计。
- [完整发布方案](docs/release/github-huggingface-plan.md)；[全量文件清单](exp/release_audit/20261002/files.csv)；[逐文件去向](exp/release_audit/20261002/release-manifest.csv)；[权重索引](exp/release_audit/20261002/model-release-index.json)；[按实验统计](exp/release_audit/20261002/experiments.json)。

## 2026-10-03--09：30：公开迁移代码、实验记录及可复现资产

- 目标已确认：GitHub `sxh-kk/EgoRecover`，直接提交main；HF模型与数据均公开。当前上传进行中，完成状态见[发布状态](docs/release/publication.json)。
- 选中158个模型产物和10,424个数据/缓存/评估文件，共73.311 GiB；保留正式G四组对照、有效P迭代、原E7 EMA、当前6个resume及公平配对所需初始权重。未选旧状态的理由见[迁移说明](docs/release/migration.md)和[排除清单](docs/release/excluded-files.csv)。
- 本地183项测试通过。移位目录中G strict加载及两帧CPU推理通过；D组P完整原dev复现34.071924941mm，与原报告差约0.000002mm，支持保存模型与评估路径的正确迁移；没有新增训练结论。
- 原始数据ZIP已在HF服务端复制，8个本地成员与固定官方归档目录核对一致。大权重/缓存改为64 MiB字节分片；510个唯一分片SHA256全部核对，真实R0权重合并与G缓存解压均验证原SHA，错误分片拒绝且不写入目标文件。
- [G核验](verification/migration_g_smoke.json)、[P完整dev核验](verification/migration_p_dev.json)、[分片恢复核验](verification/migration_parts_restore.json)、[迁移入口](docs/release/migration.md)。人体模型资产需要单独授权获取；跨机器完整训练轨迹与严格resume尚未验证。

## 2026-10-03--09：55：GitHub main与公有HF全量迁移完成

- [GitHub main](https://github.com/sxh-kk/EgoRecover/tree/main)保留当前代码、配置、测试、文档、CV及轻量实验记录；旧main历史保留，没有强推。
- [HF模型](https://huggingface.co/sxhkk/EgoRecover)保存158个选中模型产物，逻辑大小25.625 GiB；[HF数据](https://huggingface.co/datasets/sxhkk/EgoRecover-data)保存原processed数据、必要缓存与最终评估证据，恢复10,424个原文件、47.686 GiB。模型与数据均公有。
- 全部分片/直接文件已经提交。远端逐对象核对大小及LFS SHA256或Git blob ID通过；固定model revision为 `8b8190b6964f7c257346cbd65d31010a48007f28`，dataset revision为 `fdbff9f6a9856ff5b13e9818edf8677e395c6ee2`。
- 183项原有测试通过；移位目录的独立P完整dev复现34.071924941mm，与原报告差约0.000002mm。进一步从公开HF实际下载5个原文件、9个字节分片，SHA256全部一致，并在干净目录完成G strict加载与两帧CPU推理。错误分片/整文件哈希、危险路径和已有文件冲突检查通过。
- 结论：代码、已选权重、冻结输入、协议与实验结论可迁移保存；没有新增训练成绩。SMPL-X需另行授权获取，跨机器完整训练/精确续训轨迹仍未验证，未选71个旧resume等产物保留在原机器。
- [迁移与恢复命令](docs/release/migration.md)、[发布状态](docs/release/publication.json)、[文件/SHA256索引](docs/release/artifacts.json)、[远端对象核验](verification/migration_remote_objects.json)、[公开HF下载复现](verification/migration_remote_g_smoke.json)、[排除清单](docs/release/excluded-files.csv)。

> 迁移说明：本文部分原产物未选入发布包，相应链接指向迁移范围说明；具体原路径和原因见发布排除清单。
