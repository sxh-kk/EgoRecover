# P 常速度残差实验

日期：2026-09-29。状态：独立入口与指标已实现，15项测试及真实GPU资产检查通过；用户已要求GPU4–7执行，四组×三种子训练已完成；[结果](../../exp/egorecover_prior_cv_residual/v1/RESULTS.md)。

目标：在真实历史条件下，检验常速度基准加可学习残差能否改善 P 的下一帧预测，并检查收益是否伴随姿态或自反馈退化。上一轮结果见 [Two-Forward 结果](../../exp/egorecover_prior_two_forward/v1/RESULTS.md)，体型诊断见 [shape_reference.json](../../exp/egorecover_prior_two_forward/v1/diagnostics/shape_reference.json)。

## 1. 本轮问题与范围

需要分别回答：

1. 同样网络和损失下，常速度基准是否优于保持基准？
2. 学习残差是否优于无需训练的常速度，而非仅继承常速度的起点优势？
3. FK 监督在常速度基准下是否有效？改善是否同时体现在运动姿态上？
4. 下一帧收益能否延续到短时自反馈？

本轮使用 GT 历史，保持 20 帧输入、243D 输出、256 宽度、4 层/8 头 Transformer。暂不引入 Two-Forward、速度输入、额外角度/速度损失、体型一致监督或网络扩容。PA-MPJPE 作为新增评价，不改变训练损失。

## 2. 模型改动

保持现有残差结构，仅显式选择基准：

```text
预测 = 编码后的物理基准 + Transformer(历史) 输出的归一化表示修正量
base_mode = hold | constant_velocity
```

常速度基准复用 [MotionCodec.constant_velocity_prior](../../egorecover/codec.py)：

- 世界位置：p_next = 2 * p_last - p_previous。
- 世界旋转：R_next = (R_last * R_previous^T) * R_last，使用旋转矩阵复合。
- 身体参考系同样外推，随后按现有 planar 规则和编码方式处理。
- 手部、接触和体型通道保持上一帧。
- 最后一层继续零初始化，使 step0 严格等于所选物理基准。

这仍是归一化 243D 表示上的残差网络，没有将旋转残差改成另一种参数化。基准必须由物理状态计算再编码，不能直接对相邻归一化243D向量线性外推；参考坐标系增量必须正确处理。

训练、dev、纯 P rollout 和以后接 G 必须使用相同 base_mode。将模式写入 checkpoint、恢复身份和报告；旧 checkpoint 维持 hold 的原语义，不能在未训练的情况下替换其基准。网络输入仍只含历史，不读当前观测或 GT 目标。

## 3. 四组配对对照

| 配置 | 基准 | 历史 | geometry | FK | 作用 |
|---|---|---|---:|---:|---|
| H0_hold_dense | 保持 | GT | 1 | 0 | 重跑原 A 对照 |
| V0_cv_dense | 常速度 | GT | 1 | 0 | 与 H0 比较基准改动 |
| H1_hold_fk | 保持 | GT | 1 | 1 | 重跑原 C 对照 |
| V1_cv_fk | 常速度 | GT | 1 | 1 | 与 H1 比较基准改动；与 V0 比较 FK 监督 |

四组均训练 seeds 62、63、64，共 12 个训练任务。每个种子四组共享相同初始网络参数、目标帧采样顺序与训练预算，分别初始化优化器。记录初始权重哈希，以便核验配对条件。

另评估两个无训练基线：Hold、Constant Velocity。旧 A/C checkpoint 仅作复现参照，主要结论来自本轮配对训练。

主比较：V0−H0、V1−H1、V0−CV、V1−CV；辅助比较：H1−H0、V1−V0。均以当前项减参照项，负值表示位置误差改善。另报交互差值 (V1−H1)−(V0−H0)，检查 FK 是否改变基准改动的效果。

## 4. 数据与训练预算

- 复用上一轮已审计的 [数据身份](../../exp/egorecover_prior_two_forward/v1/data/report.json) 与 GT 序列缓存，不覆盖原文件。
- 48 train / 12 dev / 12 holdout takes；当前训练缓存仅包含 train/dev。每 take 200 帧，10 Hz。
- 输入过去20帧；训练目标 t=40…199，与上一轮目标范围一致。每次预测未来100 ms。
- 每组2400更新，batch32，AdamW lr=3e-4、weight_decay=0.01，梯度裁剪1.0，网络dropout=0.1。
- 保持现有表示损失、geometry和FK的定义及缩放。表示损失中参考系9通道权重8；位置平方误差按0.1 m尺度归一化。
- 每100步在同一dev上评估主指标，包含step0。按最低dev Body MPJPE选checkpoint；相同分数保留更早checkpoint。
- 报告选中点与最终2400步结果。若选中step0，必须明确写“学习残差未超过物理基准”，不能把起点优势算作训练收益。
- 第一轮不自适应追加预算。若有未收敛证据，需要另立统一预算的后续实验，不能只延长有利配置。

## 5. 指标定义与评价协议

所有位置指标用mm，角度用度；先按帧计算，再按take平均，最后对take宏平均。主要报告身体22关节，避免混入E7可选12关键关节口径。

| 指标 | 定义 | 用途 |
|---|---|---|
| Body MPJPE | 预测旋转、根位置、固定启动体型经SMPL FK得到22关节，与记录GT计算平均欧氏距离；不对齐 | 主指标和唯一选checkpoint指标；原名fk22_mm |
| Body PA-MPJPE | 对同一组FK关节逐帧做平移、旋转、统一尺度对齐，再计算MPJPE | 姿态辅助评价；复用eval.metrics.reconstruction_error |
| Root error | 预测根位置与GT根位置的欧氏距离 | 整体位置 |
| Root-relative MPJPE | 双方各自减根位置后计算22关节误差 | 去整体平移后的误差，仍包含整体朝向和体型影响 |
| Global joint rotation error | 22个关节世界旋转的SO(3)测地角均值 | 延续原rotation_deg；不称为UniEgoMotion头部旋转指标 |
| Local joint rotation error | 21个非根关节相对父关节的旋转测地角均值，根朝向另报 | 分离内部关节姿态和整体朝向 |
| Same-shape MPJPE | 预测FK与GT姿态FK均使用同一个beta_boot后比较；保留各自根位置和朝向 | 诊断运动预测；不用于替换正式GT或隐藏体型误差 |
| Dense MPJPE | 解码后直接输出的22关节位置与GT比较 | 诊断位置输出与FK输出是否一致改善 |

PA只用于预测后的离线评分，不反馈给模型。尺度对齐不能消除所有身体比例差异，因此PA与统一体型诊断不能互相替代。评估完整记录固定体型来源。

### 单步预测

每次都使用20帧GT历史，dev t=20…199，共2160帧；主指标采用完整集合，另保留t≥40的匹配子集。所有组与物理基线使用相同输入。

### 纯 P 自反馈

每个dev take固定t=20、40、…、180，共108个起点。只在起始提供20帧GT，之后连续预测10帧并反馈自身解码状态，不回填GT。沿用上一轮未额外FK投影历史的反馈定义，避免混入反馈方式改动。

报告100/200/400/800/1000 ms各端点的Body MPJPE、PA-MPJPE、Root error和Same-shape MPJPE。1000 ms指第10帧，不是1秒内平均。rollout起点采样与完整单步集合不同，因此两张表的100 ms不能当作相同样本的结果。

## 6. 统计、验收和结论边界

每个配置报告三个种子的均值、标准差及逐种子结果；不挑最好seed。对每项配对比较，先对同一take的三个seed差值取平均，再按take配对bootstrap 10000次，固定统计seed62，报告95%区间。该区间反映take间差异，种子变化另由逐种子结果和标准差体现。

三个判读层次：

1. **基准改动有效**：V0相对H0、V1相对H1分别判读；若只在某一种FK配置有效，就限定结论范围。
2. **残差学习有效**：候选在三个seed上均优于纯CV，平均改进的take配对区间上界低于0。若有均值改善但区间跨0，仅记录改善趋势。
3. **达到本项目单步目标**：三种子平均Body MPJPE≤35 mm且Same-shape MPJPE≤15 mm；Body MPJPE≤33 mm为更强目标。这些是当前数据协议下的工程目标，不是论文通用门槛。当前CV参照分别约36.75/16.84 mm；实际判断用本轮同协议重算值。

长时稳定性独立判读：报告各时长相对配对Hold残差组、纯Hold和纯CV的差值。单步达标但200/400 ms出现跨种子一致退化时，标为“单步有效，短时自反馈仍需改进”；1000 ms未优于最强物理基线时，不宣称长时预测有效。不能用PA变好替代世界位置变好。

dev同时用于选checkpoint和方法判断，不能称为独立测试。待方案冻结后，按三种子平均dev Body MPJPE选择V0或V1（相同分数优先V0），冻结配置和checkpoint，再对预留12-take holdout统一评估候选、对应H组及两个物理基线。holdout缓存须独立准备与审计，不改写train/dev身份；一次性报告全部三个seed，不按holdout挑模型。若之后根据holdout改方法，须将其标为已用于开发。

单步或holdout结果不能证明接入G有效。接G需后续固定G的独立对照；连续短段预测历史训练也留作后续实验。

## 7. 实现、资源与记录

计划实现顺序：

1. 在独立P的batch构造/预测路径加入显式base_mode，并绑定checkpoint与恢复身份。
2. 扩展统一评价器，增加PA、局部旋转和统一体型指标；保持旧fk22_mm语义，可增加标准名称映射。
3. 本轮独立调度入口为run/prior_cv_residual_experiment.py，避免覆盖旧队列。
4. 小规模核验：零残差等于选定物理基线；CV只读最后两帧历史；旋转外推和编码解码一致；训练/单步/rollout基准一致；恢复拒绝模式不匹配；PA已知相似变换不变性、角度单位和同体型零误差可验证。
5. 核验通过后按seed62→63→64运行，每轮H0/V0/H1/V1分别分配GPU4/5/6/7（用户最新指定，覆盖先前GPU0–3安排）。启动前读取现有任务和显存，延续共享用卡约束；不停止其他项目任务，不等待LingBot，不恢复已停止的G消融评估。

输出目录为 `exp/egorecover_prior_cv_residual/v1/`；队列与计划已创建，汇总结果将在12次训练结束后生成：

```text
queue.json / controller.log / jobs/
plan.json
train/<case>_s<seed>/
    initial.pt / prior.pt / resume.pt
    experiment.json / selection.json / report.json
    dev_prior.pt                  # 逐帧/逐take指标与预测；物理基线在H0_s62下
evaluation/holdout/              # 配置冻结后独立准备和评估
summary.json / RESULTS.md / figures/
```

记录同步训练耗时、评估耗时和总耗时；在共享GPU上不将耗时解释为独占性能。完成时间按首轮实测估算。

[LOG.md](../../LOG.md)只写设计/启动/异常摘要及最终结论，不写逐checkpoint选点。完成后结论必须回答四个研究问题，报告改善幅度、跨seed一致性、自反馈取舍与holdout状态，并链接实际生成的RESULTS.md、summary.json和图表。最终结果文件生成前，不能把预定路径当作已完成结果。

## 8. 参考与对应关系

- [UniEgoMotion §4.1](https://arxiv.org/html/2508.01126v1#S4.SS1)：使用身体22关节MPJPE及逐帧Procrustes对齐后的MPJPE-PA；本文档采用这些几何定义并换算为mm。输入条件、数据划分、固定体型及预测时长仍与论文不同。
- [原评价实现](../../eval/metrics.py)：提供身体MPJPE与PA计算，可复用几何定义。
- [当前P网络](../../egorecover/prior.py)、[物理基线与编码](../../egorecover/codec.py)、[损失](../../egorecover/losses.py)、[原P评估](../../egorecover/prior_two_forward.py)：本轮改动与对照的代码起点。

## 9. 本轮运行与监控

调度入口：[prior_cv_residual_experiment.py](../../run/prior_cv_residual_experiment.py)；训练入口：[train_prior_cv_residual.py](../../run/train_prior_cv_residual.py)；指标与基准：[prior_cv_residual.py](../../egorecover/prior_cv_residual.py)。

当前队列：[queue.json](../../exp/egorecover_prior_cv_residual/v1/queue.json)；固定方案：[plan.json](../../exp/egorecover_prior_cv_residual/v1/plan.json)；真实资产检查：[GPU smoke](../../verification/prior_cv_residual_gpu_smoke.json)。

```bash
cd /gaozt-test1/sxh/EgoRecover
# 总体任务状态
python -m json.tool exp/egorecover_prior_cv_residual/v1/queue.json
# 队列日志
tail -n 30 -F exp/egorecover_prior_cv_residual/v1/controller.log
# seed62四组日志；后续种子将文件名中的62换成63或64
tail -n 5 -F exp/egorecover_prior_cv_residual/v1/jobs/*_s62.1.log
```

调度器有进程锁。恢复失败队列使用原入口和 `--retry-failed`，不要重复启动第二套实验。12任务完成后自动生成RESULTS.md、summary.json和自反馈图，并在LOG写比较结论。当前调度范围为train/dev四组实验；holdout在配置冻结后另行执行，报告会明确尚未评估。
