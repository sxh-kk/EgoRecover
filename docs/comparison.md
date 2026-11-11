# EE4D-Motion 外部对比协议

2026-11-06 建立协议，2026-11-11 补充已实现的手部辅助评测。当前研究只使用 EE4D-Motion，Nymeria 留作后续扩展；主任务为在线身体重建，手PCA沿用Flow监督并在固定400片段上补充手指误差。本文件确定外部对比接口；内部结果见[实验报告](reports/ee4d_experiments_20261111.md)，不表示已经复现外部方法。

## 身体与手部的范围

[原始 EgoExo4D 的 EgoHandPose 基准](https://docs.ego-exo4d-data.org/benchmarks/ego_pose/ego_hand_pose/)提供独立的 3D 手部姿态标注。它与 EE4D-Motion 的 SMPL-X 拟合参数是不同的数据来源：后者通过多视图姿态估计、优化、时序平滑和质量筛选产生伪真值，见 [UniEgoMotion §3.6 / Appendix G](https://arxiv.org/html/2508.01126v1#S3.SS6)。不能将一个缓存内存在手部参数，解释为对应帧已经提供了独立人工手指真值。

[上游表示代码](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/dataset/representation_utils.py)把左右手姿态分别压成 12 维 PCA，同时仅用 SMPL-X 的前 22 个关节构造身体变换。本项目沿用该 243 维接口及辅助通道，以兼容 E7/P；当前变更限定身体输出与评估范围，不声称已经从模型内部移除了手部参数。

主精度评估覆盖 SMPL22 身体骨架。左右手腕为身体关节20/21，报告 `Wrist PE`。固定400片段的补充评测解码预测手PCA，并对SMPL-X关节25至54计算 `mpjpe_hand_mm`；同时记录左眼关节23误差及每帧最低网格点离地高度绝对值。补充手指标签来自官方SMPL-X拟合，不是独立人工手指标注。尚不报告手指PA-MPJPE或手部交互指标，手腕误差不能代替完整手指误差。

## 外部方法

| 方法 | 在当前实验中的位置 | 原论文对应指标与边界 |
|---|---|---|
| [EgoForce](https://arxiv.org/html/2605.13041v1)，Table 1 / §4.1 | 主要在线对手，EE4D、10 Hz、严格因果 | MPJPE、MPJRE-F、Head PE、Wrist PE、Semantic、FID、PJ、AUJ、单帧耗时、Online；PJ 按接近真实运动评价，不能将全静止当成最优 |
| [UniEgoMotion](https://arxiv.org/html/2508.01126v1)，Table 1 / §4.1 | 主要重建参照，原协议整段重建 | MPJPE、PA-MPJPE、MPJPE-H、头部旋转/平移误差、Foot Slide、Foot Contact、Semantic、FID；原论文位置指标单位为米 |
| [EgoPriMo v2](https://arxiv.org/html/2606.08495v2)，Table I / II | 补充方法，需新增 EE4D 重建复现后进入主表 | Table I 重建为 Nymeria、15 Hz、21 公共关节；其 MPJPE 在整段序列做一次相似变换对齐，PA-MPJPE 为逐帧对齐，另报 Root Mean/Final。Table II 的 EE4D 结果属于生成，不能直接与本项目在线重建比数值 |

EgoForce 另接收带可见性掩码的左右手腕位姿。统一输入对比必须让双方使用相同可获得的手腕估计，或显式关闭 EgoForce 的手腕输入并重新评测；不可直接将其额外稀疏控制与本项目仅头轨迹和图像特征混合为同设置。它的未来窗口槽位是待预测状态，不代表使用未来观测。

EgoPriMo 使用文本和异构训练数据；如纳入，需记录文本可用性、训练数据及采样预算。其整段重建不能直接标记为严格因果；因果适配需单列名称并检查信息访问。暂不为比较 EgoPriMo 而下载 Nymeria。

## 当前指标表

| 报告字段 | 定义与单位 | 主要参照 / 用途 |
|---|---|---|
| `mpjpe_world_mm` ↓ | 同一世界坐标下，22 个身体关节的欧氏位置误差均值；mm，不做根对齐或相似对齐 | EgoForce、UniEgoMotion 的身体重建精度；主精度指标 |
| `pa_mpjpe_mm` ↓ | 每帧分别估计含正尺度、旋转、平移的相似变换，再对 22 个关节计算 MPJPE；禁止镜像反射 | UniEgoMotion、EgoPriMo 的局部姿态比较 |
| `seq_sim_mpjpe_mm` ↓ | 整段全部帧共用一个相似变换后计算 MPJPE；mm | 对齐 EgoPriMo v2 的指标类型；关节数仍为本项目 22，不能直接搬用其 21 关节分数 |
| `wrist_pe_mm` ↓ | 左右手腕关节 20/21 的世界位置误差均值；mm | EgoForce 的 Wrist PE 类型；不涉及手指 |
| `head_joint_pe_mm` ↓ | 身体头关节 15 的世界位置误差均值；mm | 身体头部跟踪；明确区别于上游以左眼 23 为位置参考的 Head Translation |
| `mpjre_f` ↓ / `head_rot_f` ↓ | 22 个身体关节 / 头关节的世界旋转矩阵 Frobenius 差均值；无量纲 | EgoForce 的 MPJRE-F 和头部旋转误差类型；不能混用角度值 |
| `root_mean_initial_translation_mm` ↓ / `root_final_initial_translation_mm` ↓ | 仅平移匹配首个评价帧的骨盆后，计算根位置误差均值 / 末帧误差；mm | EgoPriMo Root Mean / Root Final 类型；观察轨迹漂移 |
| `foot_slide_uniegomotion_mm` ↓ | 按上游足点 7/10/8/11、踝/趾高度阈值 0.08/0.04 m，累计贴地水平位移的高度权重后，除序列帧数并对四点平均；mm | 复现 UniEgoMotion [Foot Slide 代码](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/eval/metrics.py#L203)；不是 mm/s |
| `velocity_error_mm_s` ↓ / `acceleration_error_mm_s2` ↓ | 用真实时间间隔差分预测和真值位置，计算一阶 / 二阶导数误差 | 时序稳定性的辅助诊断，避免只追求平滑而忽略真实动作 |
| `diagnostic_jerk_*` | 用明确的三阶差分与时间尺度，报告峰值、积分及对应 GT；峰值 mm/s³，积分 mm/s² | 当前诊断，不冒充 EgoForce 定义尚未逐项核实的 PJ/AUJ；需与 GT 对照 |
| 计算耗时 / P95 / NFE | 同设备同步计时的单帧计算耗时（ms）、尾延迟和求解调用数 | 在线精度与效率曲线；同时另报未来观测等待量与启动成本 |

建议主表先使用世界 MPJPE、PA-MPJPE、Wrist PE、MPJRE-F、Foot Slide、计算耗时和因果性；整段相似对齐、头部、根轨迹和导数诊断放附表。每种方法统一从当前帧首次输出计算分数，不能将后续修正的历史姿态代替在线输出。`same_shape_mm` 继续作为固定启动体型下的内部辅助指标，不能代替原始伪 GT 上的 `mpjpe_world_mm`。

TMR 的 Semantic/FID 保留为后续对齐项目：固定编码器 checkpoint、输入表示、采样率、窗口、归一化及评价样本后再加入。没有该依赖时报告为未计算，不能用关节误差或自行训练的特征距离同名替代。EgoForce 的 PJ/AUJ 需核实其差分、单位、归一化及序列聚合定义后才能与论文分数比较。完整网格的 Foot Contact、Air ratio 与脚部关节代理也不可混用，本轮不将代理指标写成论文同名指标。

## 比较规则

原论文发表数值仅作为文献参考。正式主表在同一 EE4D 留出集重新评测，固定关节映射、帧率、坐标、可用输入、启动条件、样本/片段范围和统计加权。当前采用官方 train/val，不把 val 改称 test。官方验证索引与提供的 4,405 条 GT key/长度一致，跳过 26 条仅 21 帧的尾段，保留 4,379 条；只评分每段第 20 帧起的首次输出，所以仍不同于论文整段 80 帧口径。外部方法须使用相同片段与评分 mask 后才能直接比较。缓存中随后修正的历史姿态另列；模型沿自身预测缓存连续运行。无未来观测时，观测等待延迟为零，但仍需实测计算耗时。

官方首轮以零 beta 固定体型和头部对齐中性模板启动，无 GT 体型或身体历史。Aria 原点与 SMPL15 的重合属于近似，需让对照方法采用共同启动条件或单独报告差异。官方地面常量为零时视为未知，不以 GT 足点补齐；该片段的 Foot Slide 置空，其余适用指标保留。汇总 JSON 报告每项指标实际可用的 episode/take 数，CSV 另列 `floor_calibrated_episodes`。

当前 GPU 耗时在相同设备同步计时，排除每个 episode 第一次被评估查询，将其余逐帧耗时汇总后计算均值和 nearest-rank P95；FPS 为 `1000 / 平均毫秒数`，不平均各段 FPS 或 P95。计时包含 P、源构造、Flow、缓存更新和身体关节解码，不包括视觉编码、SLAM、观测打包、噪声采样和指标计算。`first_frame_ms` 仅为启动历史之后的第 20 帧首次评价查询，不包含启动历史构造和模型/资产加载，不代表系统启动到首个姿态的耗时。`frame_budget_exceed_ratio` 是核心调用耗时超过源帧周期的比例，不包含排队延迟，不能称为端到端截止期违约率。

`metrics.csv` 为每个 NFE、history mode 和质量统计方式各写一行：`take_mean` 先在每个 take 内平均 episode，再对 take 等权；`episode_mean` 对所有 episode 等权。两种表中的时延均使用同一逐帧样本池，并通过 `timing_aggregation` 标明。外部方法转换到相同顺序的 `[T,22,3]` 米制身体位置后，可调用 `egorecover.metrics.comparison_metrics(prediction, truth, timestamps)` 使用完全相同的函数；若提供两组世界旋转矩阵，也会输出旋转误差。坐标转换和关节映射必须事先固定，不能按误差选取。

本方案成对反馈的 1/2/4/10 帧评价在每步有新观测到达的连续重建中进行，对应 0.1/0.2/0.4/1 秒；不是遮断未来观测的 motion forecasting。其结果用于内部反馈消融，不能与 EgoPriMo 的 forecasting 表直接比较。

EgoForce 和 UniEgoMotion 的位置数值以米报告，本项目统一转换为毫米。22 个身体关节、世界坐标、10 Hz、固定启动体型、相同有效评价帧和首次输出为本项目默认设置；这构成统一复评协议，不自动等于每篇论文的原始评测协议。

对同一基线记录绝对变化和相对误差下降：对于越低越好的正数指标，下降率为 `(基线 - 当前) / 基线 × 100%`；基线为零时只报告绝对变化。主精度改善、实测耗时与多个训练种子的均值/波动共同判断改进，不用单一对齐指标或平滑指标决定优劣。当前代码仍保留 `body_mm`、`same_shape_mm`、中心误差、连续/匹配缓存重建和成对多步诊断，本轮增加上述身体对比接口；外部方法成绩需在实际复评后填写。
