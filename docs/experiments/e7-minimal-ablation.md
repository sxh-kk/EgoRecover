# 原 E7 最小适配：四组窗口与地面对照

日期：2026-09-29。状态：21:12全部完成，四个worker均正常退出。[结果与分析](../../exp/e7_minimal_ablation/v1/RESULTS.md)。按用户最终要求只保留四组，取代先前讨论的全因素矩阵。

结论：E0/E1/E2/E3的Body分别59.958/75.254/178.491/187.808mm。启动估计地面带来112.554–118.533mm的增量，是本轮主要误差来源；滑窗增加9.317–15.296mm，并使原地面组速度误差177.388→592.463mm/s。12个take、3个draw均已完成；尚未在修复后的历史上重测P。

## 要回答的问题

原 E7 权重在最近修改过的历史生成流程中误差很大。先检查两个主要变化：整窗输出改成逐帧因果滑窗，以及原地面处理改成启动估计。冻结原 E7，不训练 P/G，不用 P 的下一帧误差代替 E7 自身的重建误差。

| 组别 | 窗口 | 地面 | 比较意义 |
|---|---|---|---|
| E0 | 原整窗重建方式 | 原 source floor；缺失时原 GT 脚部回退 | 同片段原流程基线 |
| E1 | 因果滑窗，每次只提交末帧 | 同 E0 | E1−E0：在线窗口方式的影响 |
| E2 | 同 E0 | 固定前20帧头高估计 | E2−E0：地面估计的影响 |
| E3 | 同 E1 | 同 E2 | E3−E1：因果场景下地面影响；E3−E2：估计地面下窗口影响 |

交互量 `(E3−E1)−(E2−E0)` 检查两项变化是否互相放大。四组可以识别这两个因素及其交互；无法同时单独识别全部旧改动。体型固定、planar解码、改写观测编码、可变长度输入全部撤回，恢复原 E7 行为，不再作为本轮变化项。

E1 的“窗口方式”包括只用过去观测、末帧位置、不同长度有效上下文，以及相邻输出来自不同窗口采样。这四组不能进一步分离这些机制，不能把所有差值单独称为“未来信息收益”或“噪声抖动收益”。

## 保持一致的部分

- 原仓库 `/gaozt-test1/sxh/UEM-update-original`，固定提交 `156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08`；执行前检查工作树和提交。直接导入上游模型、Flow和SMPL解码代码，不改上游仓库。
- 同一 E7 checkpoint EMA，x0、Euler10、80帧输入。短窗口右侧补零到80，使用原有效帧 mask。原窗口预测 β 的有效帧均值用于 SMPL；不固定启动 β，不对网络预测的参考变换加 planar 投影。
- 保留上游头部 canonicalization 和18D轨迹编码。新增的适配只接受观测，避免为构造条件而引入完整人体GT。
- 每个窗口独立标准高斯噪声。种子由 take、draw、窗口末帧确定；同一窗口两种地面设置共享完全相同噪声。不同窗口的噪声不同，三个draw独立重复。
- 只用已有12个dev take的clean观测，各200帧、10Hz。三次采样种子1062/1063/1064。holdout不用；故障变体暂不扩展，避免混入第三个因素。
- 四组统一还原到原始数据世界坐标，地面偏移加回，再对同一个原始GT的SMPL22身体关节评分；不按各自地面重新定义评价坐标。

## 窗口与评价帧

E0/E2使用三个完整窗口 `[0,80)`、`[80,160)`、`[120,200)`。每个物理目标只评分一次：20…79取第一窗，80…159取第二窗，160…199取第三窗。这保留整窗联合重建方式，adaptation仅是匹配当前200帧片段的窗口调度，**不是完整官方val复跑**；90.73mm只作原仓库历史报告。

E1/E3对每个t取 `[max(0,t−79),t+1)`，不足80时补齐，解码后只保留t。每次窗口从真实起点观测重新确定世界坐标，未将上一帧预测作为当前E7条件，历史不回改。

主评价 `t=20…199`；另报20…78的短窗口段、79…199的完整窗口段，以及100…199的较晚阶段。79、159、199处两种窗口边界相同，应产生一致结果（容许批量数值舍入误差），可作为额外一致性诊断。

原地面：读取原序列 `floor_height`；等于0时使用当前有效窗口GT脚部平均高度−0.02m，严格复现原逻辑。此组包含标注辅助，只作基线诊断，不能称完全无GT部署。估计地面：复用前20帧实际头部z中位数−训练平均头高，四组共享观测，估计结果整个片段固定，不使用未来或身体GT。

## 指标与结论

主指标：世界坐标SMPL22 Body MPJPE（mm）。辅助：PA-MPJPE、根位置、根相对MPJPE、head关节15位置、dense关节位置、连续帧速度误差。head关节15不同于原论文常用leye23指标，不直接与论文Head列比较。

先在每个take内平均帧和三个draw，再对12个take等权平均。所有差值按take配对bootstrap 10,000次；区间是探索性区间，未作多重比较校正。输出上述四个单因素比较与交互量，并检查短窗口与完整窗口阶段是否一致。不能用80帧窗口中的帧数当成独立样本数。

- E1明显差、E2接近E0：优先检查/适配在线窗口任务。
- E2明显差、E1接近E0：优先修正启动地面估计。
- E3额外恶化且交互量为正：两项变化组合后误差更大。
- E0本身不正常：先解决基线复现问题，不能据此开始P适配或宣称需要训练G。

## 实现、验证和运行

配置：[e7_minimal_ablation_v1.json](../../config/e7_minimal_ablation_v1.json)。
入口：[run/e7_ablation.py](../../run/e7_ablation.py)；适配：[egorecover/e7_ablation.py](../../egorecover/e7_ablation.py)。正式结果目录 `exp/e7_minimal_ablation/v1/`。模型推理和解码均冻结，训练代码不改动。

从 EgoRecover 根目录执行：

```bash
PY=/root/miniconda3/envs/egorecover/bin/python
$PY -m run.e7_ablation plan
$PY -m run.e7_ablation prepare
$PY -m run.e7_ablation verify
# 两个train take的稀疏时间点检查；smoke不能作为正式成绩。
$PY -m run.e7_ablation evaluate --smoke --device cpu
```

`verify` 必须通过才允许执行模型评估：在两条train片段的短窗、满窗和移动窗口上，对照上游完整表示转换的输入，以及GT编码→dense关节解码→世界坐标还原。SMPL FK与保存关节标签的差异单独报告；不把这项差异直接等同于坐标适配失败。其通过只证明适配正确，不代表模型达到报告精度。

以下为手动复跑命令。当前已有控制器自动调度，运行期间不要再手动重复启动；四个shard分别使用GPU4…7。

```bash
CUDA_VISIBLE_DEVICES=4 $PY -m run.e7_ablation evaluate --shard 0 --shards 4
CUDA_VISIBLE_DEVICES=5 $PY -m run.e7_ablation evaluate --shard 1 --shards 4
CUDA_VISIBLE_DEVICES=6 $PY -m run.e7_ablation evaluate --shard 2 --shards 4
CUDA_VISIBLE_DEVICES=7 $PY -m run.e7_ablation evaluate --shard 3 --shards 4
# 全部完成后；缺少任何take/draw/组别会拒绝汇总。
$PY -m run.e7_ablation summarize
```

每个take/draw/组别独立缓存预测，完成的case可复用。断点发生在某一组中途时重跑该组；不宣称逐batch续跑。预测缓存锁定当前代码、配置、数据和模型身份。原始观测/标签缓存保留其生产代码身份与文件哈希，更新验证或推理代码可复用该数据，但必须重新验证与推理；配置或资产变化则拒绝复用。

产物：`matrix.json`、`data/report.json`、`verification.json`、`dev/<take>/<draw>/metrics.pt`、`summary.json`、`RESULTS.md`。汇总完成后自动向[LOG.md](../../LOG.md)写入配对结论和可点击链接，不记录逐点训练日志。本轮无训练。

控制器：[run/e7_ablation_queue.py](../../run/e7_ablation_queue.py)。状态与每个worker PID写入`queue.json`；日志位于`controller.log`及`jobs/verify.log`、`jobs/smoke.log`、`jobs/shard0.log`…`jobs/shard3.log`。工程检查失败会停止进入正式评估，并将原因写入LOG。
