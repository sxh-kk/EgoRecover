# P 显式速度输入实验

日期：2026-09-29。状态：14项测试及真实GPU检查通过，GPU4–7上的6次配对训练与评价全部完成。产物目录：`exp/egorecover_prior_velocity/v1/`。

## 目标与对照

在用户选定的常速度＋残差、geometry=1/FK=0基础上，仅检验显式速度输入。上一轮该配置的三个种子平均Body MPJPE=35.939mm、PA-MPJPE=11.755mm、统一体型MPJPE=16.374mm，见[上一轮结果](../../exp/egorecover_prior_cv_residual/v1/RESULTS.md)。

| 配置 | 基准 | FK | 输入 |
|---|---|---:|---|
| control | 常速度＋残差 | 0 | 原20帧身体状态 |
| velocity | 常速度＋残差 | 0 | 原身体状态＋由相同历史计算的显式速度 |

每组seeds62/63/64，共6次训练，重跑配对对照；另评估纯保持与纯常速度。首先GPU4/5运行seed62的control/velocity，GPU6/7运行seed63的control/velocity，GPU4/5各自空闲后运行seed64。

## 速度定义与推理接口

每个历史帧提供70维特征：

- 根部平移速度3维，单位m/s，按1m/s固定尺度归一化。
- 根部空间角速度3维：Log(R_now R_previous^T)/dt，轴表达在共同窗口坐标系中。
- 21个非根关节局部角速度63维：先转成相对父关节旋转Q，再计算Log(Q_previous^T Q_now)/dt，轴表达在前一帧的关节局部坐标中。
- 相邻差分是否有效1维。窗口第一帧全部速度为0、有效位为0，其余19帧有效位为1。
- 固定dt=0.1秒；角速度按π rad/s归一化。尺度预先固定，不读取dev/holdout拟合统计量。

输入仍是已有归一化243D历史。先解码每帧局部关节变换和参考系增量，累积参考变换恢复共同窗口坐标，再求差分；不能直接对不同参考系下的243D编码相减。旋转通过SO(3)对数映射得到旋转向量，不对6D旋转编码直接差分。

推理同样只读取历史，GT、当前观测和未来帧均不进入速度函数。无需SMPL资产或新增传感器。本版要求完整有效的历史窗口，不支持含空洞的padding；固定20帧启动符合此约束。首版不加入加速度、平滑、速度损失或历史混合。

## 网络与配对初始化

```text
state_tokens = Linear(243, 256)(history_motion)
velocity_tokens = Linear(70, 256, bias=False)(history_velocity)
tokens = state_tokens + velocity_tokens
位置编码 → 原4层/8头Transformer → 原243D残差输出 → 加常速度基准
```

新增17920个参数。速度映射零初始化，原输出层继续零初始化。新增模块的初始化不消耗配对训练的后续随机序列；两组的共有网络参数、训练采样和起始dropout RNG保持相同。step0均等于纯CV，速度映射初始也不扰动状态token。

每次forward自动从history_motion计算速度。训练、单步评价和自反馈均使用同一模型接口。checkpoint绑定velocity_input、velocity_schema、base_mode、统计量和代码身份，恢复时拒绝特征配置变更。独立加载器检查归一化统计量；未来接G时仍须显式使用CV基准及新模型加载器，当前实验不修改已部署G链路。

实现：[速度特征与网络](../../egorecover/prior_velocity.py)、[训练入口](../../run/train_prior_velocity.py)、[调度与汇总](../../run/prior_velocity_experiment.py)。旧实验代码与产物保持原样。

## 固定训练协议

- 复用48 train / 12 dev的已审计GT序列缓存，12 holdout不参与。
- 20帧输入，10Hz，训练目标t=40…199。
- 每组2400步、batch32、AdamW lr3e-4/wd0.01、梯度裁剪1.0、dropout0.1。
- geometry=1/FK=0，其余原表示损失和缩放不变。
- 每100步全dev评价，包含step0；唯一checkpoint选择指标仍为100ms世界Body MPJPE，平分时保留更早点。
- 不按中途结果给某一组单独增加预算；保留最终2400步和选中checkpoint指标。

## 指标与结论规则

主指标：完整GT历史单步的Body MPJPE；辅助：PA-MPJPE、统一体型MPJPE、根位置、局部/全局关节旋转、dense位置。沿用上一轮定义，22关节、单位mm、take宏平均。

比较velocity−control和velocity−纯CV；先对同一take的三个seed差值取平均，再按take配对bootstrap10000次，统计seed62。报告三种子均值、标准差、逐seed差值及95%区间。共有初始网络权重必须逐张量一致。

- 各seed单步均改善且配对区间上界低于0，才称为稳定改善；区间跨0时只记录趋势。
- 检查PA和统一体型指标是否同向改善，避免只降低正式世界误差。
- 沿用Body≤35mm且统一体型≤15mm的项目目标；不是论文通用标准。
- 根据用户最新优先级，纯P自反馈100–1000ms只作辅助诊断，不用长程单项退化否决单步收益。实际G预测历史上的100ms精度和接入G后的收益仍需后续验证。
- dev参与选模，holdout尚未使用；本轮不根据holdout选择方法，不自动接G。

## 产物与日志

输出下保存plan.json、queue.json、controller.log、jobs，以及train/<case>_s<seed>/中的checkpoint、逐次选点曲线、report.json和dev预测。6个任务完成后自动生成RESULTS.md、summary.json和三指标配对图。

[LOG.md](../../LOG.md)记录开始/异常摘要和比较结论，提供可点击结果链接；不记录逐checkpoint选点。报告额外参数、同步训练耗时及总耗时，不将共享GPU上的时间当成独占基准。

```bash
cd /gaozt-test1/sxh/EgoRecover
python -m json.tool exp/egorecover_prior_velocity/v1/queue.json
tail -n 20 -F exp/egorecover_prior_velocity/v1/controller.log
tail -n 5 -F exp/egorecover_prior_velocity/v1/jobs/*.log
```

## 完成结果

[最终结果表与图](../../exp/egorecover_prior_velocity/v1/RESULTS.md)；[完整统计](../../exp/egorecover_prior_velocity/v1/summary.json)。

- 三种子平均Body MPJPE：35.939→35.733mm；差值−0.206mm，take配对95%区间[−0.457,+0.048]，尚未确认主指标稳定收益。三个seed方向均改善。
- PA-MPJPE：11.755→11.679mm，小幅改善且区间跨0。统一体型MPJPE：16.374→15.914mm；差值−0.461mm，配对95%区间[−0.753,−0.164]，三个seed均改善。相对而言，收益更明确地体现在统一体型的运动预测诊断上。
- 控制组完整复现上一轮V0三个种子的单步结果。速度分支只有17920个额外参数，改动有效运行，但目前提升幅度小，仍未达到35/15mm项目目标。
- 纯P1秒自反馈均值356.213→356.729mm，仅作为辅助诊断，不据此否决单步方法。以上仍是dev选模结果，未验证holdout、G历史适配或接入G收益。
