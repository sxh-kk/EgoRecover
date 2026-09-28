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

## 2026-09-28 17:40（UTC+8）：阶段 2–4 恢复实施与资源核对

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

## 2026-09-28 17:49（UTC+8）：自动队列已启动，等待 GPU 0–3

- 新增 `run/complete_stages.py` 和 `run/stage_job.py`，接通阶段 2 训练/六格评估/预算选择、阶段 3 train/dev 分片缓存/三组 400-step replay 对照、阶段 4 四动作诊断。每个任务保存命令、PID、GPU、日志、退出码和完成校验；阶段完成或调度失败时自动追加本日志，最终生成运行目录下的 `results.md`。
- 72-take/2400 原始中断产物的续跑配置核对无差异，Gaussian checkpoint 哈希与报告一致。继续训练入口新增 replay 缓存生成 P/G 与初始化权重的一致性检查。
- 全量回归：`/root/miniconda3/envs/egorecover/bin/python -m pytest -q tests data_pipeline/tests`，**103 passed, 5 warnings in 19.53s**。随后补充队列空闲确认与任务依赖测试，相关测试集 **10 passed in 3.62s**；`compileall`、`git diff --check` 均通过。警告包括默认 NVML 加载失败和 Lightning 提示；队列使用已核实的匹配 NVML 库。
- 调度启动时间 17:48:27，控制进程 PID **1312772**。命令：`/root/miniconda3/envs/egorecover/bin/python -u -m run.complete_stages --output exp/egorecover_stages_1_4/continuation_v1 --gpus 0 1 2 3`。
- 状态文件：`exp/egorecover_stages_1_4/continuation_v1/queue.json`；控制日志：同目录 `controller.log`。17:49 确认状态为 **waiting_for_idle_gpu**，阶段 2 的两个训练任务和六个评估任务均为 pending；尚无本轮实验结果。
- GPU 0–3 正在运行另一个项目 **MTU3D 的 R2R-CE 导航数据采集**：每卡三个仿真环境和 Qwen3-VL-4B-Instruct 推理。监督进程 PID 1281153，17:25 启动，参数 `--hours 24`，输出目录 `/gaozt-test1/projects/MTU3D/outputs/r2r_ce_around_memory_24h_20260928_1725`。若未提前结束，按启动预算可能持续至 9 月 29 日约 17:25，实际释放时间以进程状态为准。
- 我们的队列每 30 秒检查，仅在连续两次确认无计算进程、显存占用低于 1 GiB、利用率低于 5% 时分配该卡；每卡最多一个本队列任务。有其他任务占用时继续等待，不中断 MTU3D。当前属于本地后台等待队列，不是集群调度器预留资源。

## 2026-09-28 17:55（UTC+8）：按用户要求切换至 GPU 4–7 共享运行

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

## 2026-09-28 18:12（UTC+8）：持续监控阶段 2 实际计算

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

## 2026-09-28T09:56:17.003790+00:00：阶段 2–4 调度停止

- 运行目录：`exp/egorecover_stages_1_4/continuation_v1`；原因：`Experiment failed; dependent stages were not started.`；未完成项不得计入结果。

## 2026-09-28 23:28（UTC+8）：阶段 2 五格完整评估与阶段性结论

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

## 2026-09-28 23:36（UTC+8）：论文指标对照与 clean 误差诊断

- 原始来源：[UniEgoMotion，ICCV 2025，Table 1 / §4](https://arxiv.org/html/2508.01126v1#S4)。论文将 AvatarPoser、EgoEgo、EgoAllo 在 EE4D-Motion 上重新训练评估；以下是该统一对照表中的数值，不是各方法原论文跨数据集成绩。MPJPE/PA 从米换算为毫米：AvatarPoser 116/68，EgoEgo 130/75，EgoAllo 163/71，UniEgoMotion 100/53。
- 本地 `continuation_v1/stage2_eval/72take_2400/report.json` 的 completed=true 完整结果，12 个共同 dev takes：Gaussian clean/freeze/drift = **200.449/200.457/203.137 mm**；History = **212.846/213.448/213.620 mm**；P-only = **751.582 mm**。此前 201.347/213.305 是三条件宏平均；大误差并非主要由这两种人工扰动抬高。
- 同协议 frozen E7 的 clean = **357.685 mm**，Gaussian clean 相比它下降约 44.0%。这是当前工程协议内的改善，不能替代与公开方法的正式比较；本地 E7 是上游 Flow 变体，不能当作论文官方 diffusion checkpoint 的直接复现。
- 协议差异：论文用官方 train/val、8 秒/80 帧片段、整段重建条件；当前是官方 train 内部划出的 12-take dev、2 秒公共模型启动后 18 秒因果闭环，模型只访问当前及历史观测、历史身体状态来自预测，体型固定为 beta_boot。当前微调 train 为 48 takes；开发集还用于选点。这些差异使 200.4 与 100 mm 只能作量级参照，尚不能据此计算正式方法退化率。我们目前的表是未对齐世界 SMPL22 FK MPJPE，不能与论文 PA-MPJPE 混比。
- 对已存逐帧误差作只读分段统计：按启动后 0–6/6–12/12–18 秒分箱，Gaussian clean = **205.838/198.463/197.044 mm**，History = **198.684/216.458/223.396 mm**，P-only = **410.687/728.356/1115.703 mm**。均为每 take 先平均再宏平均。Gaussian 的约 20 cm 误差不能全部归因于随时间累积的漂移；P-only 的增长尤为明显。该分段统计本身不能分离姿态、体型、根平移和朝向误差。
- 后续诊断优先项：同一模型/同一轨迹补充 root-relative、PA、根平移与头部误差；另设匹配论文数据划分、输入可见范围、80 帧采样和指标的评估。前者用于定位误差来源，后者用于正式公开比较。当前尚未执行这些新增评估，不改变既定四阶段队列或选模规则。
- 23:35 队列心跳正常、状态 running；阶段 2 仍五格完成，12-take/2400 训练收尾，其独立评估 pending；阶段 3/4 等待依赖。

## 2026-09-28 23:38（UTC+8）：E7 改动审计与 baseline 含义纠正

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

## 2026-09-28 23:58（UTC+8）：启动 G 损失消融，GPU 4–7

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

## 2026-09-29 00:17（UTC+8）：独立 GitHub 仓库发布准备

- 用户要求在 sxh-kk 下新建 EgoRecover，并明确选择公开仓库。将当前工作区代码、配置、文档及轻量验证报告制作成独立首次提交；排除 exp/、数据、模型权重、SMPL-X 资产、缓存和测试 XML，不携带旧 Git 历史中的实验轨迹。
- README 更新为 EgoRecover 总览；原始 E7 说明保留于 E7.md，SOURCE.md 保留上游仓库及提取提交。发布快照独立于训练工作区；原分支和运行中训练进程保持原状态。
- 上传前测试：`tests data_pipeline/tests` **113 passed, 5 warnings in 18.45s**。已检查常见凭据模式，待上传代码未发现匹配项。GitHub 创建和推送结果待认证完成后补记。
