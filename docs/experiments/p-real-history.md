# 新P对原E7预测历史的适应性验证

日期：2026-09-29。状态：已完成。clean新P Body213.535mm，对应GT历史34.099mm；输入E7历史自身Body187.205mm，需要先核查历史生成质量。[结果](../../exp/egorecover_prior_real_history/v1/RESULTS.md)。只评估D组新P，不重复旧P、CV、Hold；不训练P/G。实时状态见[队列](../../exp/egorecover_prior_real_history/v1/queue.json)。

[执行配置](../../config/egorecover_prior_real_history_v1.json) · [D组已有GT结果](../../exp/egorecover_prior_data_budget/v1/RESULTS.md) · [LOG](../../LOG.md)

## 1. 唯一核心比较

同一个新P，用E7预测历史的下一帧误差，减去用GT历史的已有下一帧误差。

| 输入 | 模型 | 数据来源 |
|---|---|---|
| 真实前20帧身体状态 | D组新P | 复用已完成D组的GT历史评估产物 |
| E7恢复的前20帧身体状态 | 完全相同的新P | 本轮新增评估 |

比较回答：换成实际恢复历史后，新P的误差增加多少，主要体现在哪些身体部位、运动状态或故障阶段。不根据该比较声称新P在E7历史上优于旧P/CV/Hold；它们之前的结果属于GT历史条件。

## 2. 冻结模型与已有对照

- 只使用D组CV残差、geometry=1、FK=0、无显式速度的三个模型。
- 已选checkpoint固定为seed62/9400、seed63/7500、seed64/8400，不重新选点。
- 已有GT结果：Body 34.099±0.025mm、PA 10.552mm、统一体型14.054mm。
- 逐帧参照为 `exp/egorecover_prior_data_budget/v1/train/D_s{62,63,64}/dev_prior.pt` 的single部分；逐seed与对应模型配对。配置中记录产物哈希，执行前校验。
- 原12个dev take、同一200帧episode、t=20…199，共2160个目标帧；模型启动体型、地面、坐标、SMPL资产和指标定义保持一致。
- GT逐帧结果已保存，直接复用；额外时间子集从原结果筛选，不重复GT推理，也不把整体34.099mm当成不同时间子集的基准。

## 3. E7历史如何生成


### 网络与采样

- 直接加载原 `model.uniegomotion.UniEgoMotion`，使用 `exp/e7/last.ckpt` 的EMA、`config/e7.yaml`、原统计量；冻结所有参数、eval模式。
- FlowMatching Euler10，Gaussian噪声，repaint关闭；不加入P、历史条件适配器或新的观测屏蔽策略。
- 采样draw seeds为1062/1063/1064；随机噪声由draw、take、时间索引确定，同一take不同输入变体共用对应噪声。与P训练seed分别命名、分别汇总。
- 原E7训练窗口为80帧。本轮采用截至当前帧的最多80帧观测，属于因果推理协议变更，不称为原官方离线benchmark。

### 提交历史的时间顺序

零起始帧索引，10Hz，每episode200帧，20帧模型启动。

1. 帧0…19：使用原72take模型启动缓存，三个新P种子、变体、E7 draw共用同一个已验证的startup。保持原beta_boot与启动估计地面，不重新用GT估计。
2. 对每个s=20…199：E7仅访问帧max(0,s−79)…s的观测，独立生成窗口，将最后一帧解码到统一世界坐标并提交为hat x_s。每次提交后不可被未来窗口改写。
3. 时刻t的P输入严格为hat x_(t−20)…hat x_(t−1)，预测x_t；P预测完成后才允许推进E7对t的恢复。
4. t<80时使用实际已有长度；不读取未来观测填满窗口。窗口锚点只来自窗口起点已可用的头部观测，使用同一个启动地面，保留原v4编码规则。
5. 窗口结果只取最后一帧；其余重生成帧不覆盖已提交历史。不能把整段200帧离线重建直接当作在线历史。

主评估t=20…199，每take180帧，与D组已存GT历史指标完全同帧。额外分别报告t=40…199（排除startup状态参与历史）和t=100…199（历史中的每个状态均来自完整80帧窗口）。两个子集均从已有GT逐帧结果筛选同帧对照。

保存预测dense身体位置、旋转、内部reference及auxiliary；不先把dense历史替换成SMPL FK身体，也不以GT坐标对齐历史。SMPL FK只用于离线评价，三个新P种子使用同一个模型startup beta_boot。保存每帧窗口锚点和观测请求范围，审计世界坐标转换。

### 现有代码复用边界

- 复用原E7加载、FlowMatching、MotionCodec、模型启动cache验证、CV残差predict和SMPL指标。
- `run/check_frozen_e7_smoke.py` 与 `run/evaluate_paired_development.py` 的frozen_e7分支使用HistoryUniEgoMotion单帧适配器；不能把其缓存作为本协议的原E7窗口历史。
- `run/collect_predicted_histories.py` 的缓存依赖已训练Gaussian/History G和旧P；不符合本轮P无关历史来源。
- 现有`prior_adaptation.predict`使用hold基准；本轮必须调用`prior_cv_residual.predict`恢复constant_velocity残差语义。

## 4. 数据与规模

保留当前12个dev take、每take原episode；holdout不使用。

- clean：主要结果。
- freeze_3s、drift_0p03mps：补充故障与恢复诊断，沿用原数据定义和事件窗口。
- E7采样draw seeds1062/1063/1064；每份历史供三个新P种子共享。P训练seed与E7采样seed是不同维度。
- 计划生成108条轨迹（12take×3变体×3draw），对应19,440个主评估历史/目标对；只评估三个新P。旧P/CV/Hold不加载、不推理。
- GT参照每个P seed只存一份；不能因为E7重复采样而把GT参照当作更多独立样本。

原E7预训练是否覆盖这些官方train take需核对来源记录。当前dev/holdout是P实验保留划分，不代表E7也未见过它们。

## 5. 指标与解释

主表按clean、冻结、漂移分别报告三个新P seed的Body、PA、统一体型MPJPE；同时列出已有GT历史结果、绝对差值和相对变化百分比。辅助记录根位置、根相对误差、局部旋转及每take帧误差P95。

离线记录E7历史自身的Body、根相对、旋转和末两帧速度误差，检查历史噪声与P退化的关系；这些诊断用GT量不进入模型输入。按历史Body误差三分位分组时只用共同历史确定阈值，写明样本数，不能按P结果挑选窗口。

统计以take为独立单位，先在同一take内平均帧、E7 draw、P seed，再对12个take做10,000次配对bootstrap，报告E7历史−GT历史的均值与95%区间。主比较为clean Body；其他条件、时间子集、指标为诊断。逐P seed和逐take结果全部保留。

解释原则：

- 差值及不确定性刻画新P对历史误差的敏感程度，不预设没有依据的毫米通过阈值。
- GT历史上的35/15mm项目目标不直接作为预测历史的门槛。
- 若退化较小且不同seed、take表现一致，可推进G集成验证；若明显退化，结合历史质量定位位置偏移、姿态或速度抖动，再决定是否设计GT/预测历史混合训练。
- 区间跨0不等于证明两种输入等效；有误差增加也不等于P结构失败。
- 本轮不比较预测历史上的替代算法，不证明闭环收益或独立泛化；P输出不写回E7，纯P长程自反馈不作为门槛。

## 6. 实施与记录

1. 实现原E7因果窗口缓存器和新P评估器；先用2个train take进行因果性、坐标与吞吐检查。
2. 校验未来观测扰动不改变已提交历史、P读不到当前观测、GT只用于离线评估、checkpoint与已有GT产物哈希一致。
3. 使用GPU4–7生成固定dev历史，按take/变体/draw断点恢复；只运行三个新P，复用已有GT逐帧指标。
4. 自动汇总误差变化、历史质量诊断与图表，写入LOG；不记录逐帧日志条目，不静默跳过失败样本。

计划产物根目录 `exp/egorecover_prior_real_history/v1/`：`plan.json`、`histories/`、`evaluation/frames.pt`、`summary.json`、`RESULTS.md`、`figures/`、`jobs/`与`queue.json`。当前队列已创建，最终结果在四个评估分片完成后自动生成。


## 运行与检查

- [控制器](../../run/prior_real_history_experiment.py)、[观测准备](../../run/prepare_prior_real_history.py)、[因果E7与P评估](../../run/evaluate_prior_real_history.py)、[因果窗口组件](../../egorecover/prior_real_history.py)。
- 新增6项检查与原7项CV检查通过：窗口不读取未来、80帧边界、世界坐标解码、P纯推理输入与原实现一致、按take配对统计不重复计算draw样本，以及完整CPU评估和部分历史断点恢复逐项一致。
- 正式分片启动前自动执行两个train take、clean、1个E7 draw、前85帧的GPU工程检查，覆盖短窗口、80帧窗口及滑窗起点变化；不以dev效果修改协议。
- 实际历史与断点保存在`work/shard*/histories.pt`、`history_resume.pt`，每20帧保存。每个`shards/shard*/frames.pt`保存新P逐帧结果；汇总至`evaluation/frames.pt`。以上为本版实际路径。

```bash
cd /gaozt-test1/sxh/EgoRecover
tail -n 20 -F exp/egorecover_prior_real_history/v1/controller.log
# 控制器意外退出后可用相同命令接管；失败任务需先排查原因，再加 --retry-failed。
/root/miniconda3/envs/egorecover/bin/python -u -m run.prior_real_history_experiment
```


## 与原E7评估流程的差异审计（2026-09-29）

本轮保留原E7网络、checkpoint EMA和Euler10，但不是原E7完整推理/评估流程的复现。下列差异已由代码确认，各自对误差的贡献尚未通过配对消融确认。

| 项目 | 原代码路径 | 本轮实际历史路径 |
|---|---|---|
| 观测窗口 | recon整窗条件，一次生成窗口，评价窗口内全部有效帧；短片段padding至80 | 每帧取最多80帧过去观测，短窗口使用真实长度，只提交末帧 |
| 时序采样 | 一段输出来自同一次联合窗口采样 | 每个时刻独立噪声重采样，只取末帧拼历史，未验证跨窗口连续性 |
| 地面 | 读取source floor_height；为0时根据GT脚部高度回退 | 固定前20帧头部高度中位数−训练集平均头高的启动估计 |
| 体型 | v4_beta使用该预测窗口的beta均值重建SMPL | 历史BodyState保留网络aux；指标中的SMPL固定最初20帧beta_boot |
| 坐标解码 | repre_to_full_sequence_v4累积网络预测的完整参考变换 | MotionCodec对参考变换逐步做planar投影，并用当前窗口起点传感器参考还原世界坐标 |
| 数据与指标 | 官方val抽样窗口；当前config/e7.yaml默认12个关键关节 | 官方train内的P-dev 12take，世界SMPL22、逐take平均；最终213.535mm是P下一帧而非E7重建分数 |

源码位置：原[data loader](../../dataset/ee4d_motion_dataset.py)、[表示与解码](../../dataset/representation_utils.py)、[预测入口](../../eval/save_uem_preds.py)、[E7配置](../../config/e7.yaml)；本轮[窗口构造](../../egorecover/prior_real_history.py)、[地面估计](../../egorecover/calibration.py)、[坐标codec](../../egorecover/codec.py)、[评估入口](../../run/evaluate_prior_real_history.py)。

普通MPJPE在原指标代码中也直接对关节位置求误差，不能笼统解释成“原E7做了PA而本轮没做”。原v4_beta也使用预测体型，不是原E7使用GT beta。

更正结论边界：213.535mm只证明当前修改后的历史生成流程配合新P的误差很大；输入历史187.205mm不能当成原E7官方指标，也不能单独归因于P适配失败。此前单元检查覆盖本轮自身因果性、解码往返及断点恢复，没有验证新旧E7完整流程数值等价。

优先核查顺序：在共同片段上先保留原E7预处理、原解码、原地面和体型处理复现原流程，再分别检查末帧/滑窗重采样、地面估计、体型固定和planar投影的影响。该诊断尚未启动；不根据当前结果直接开始P适配训练。
