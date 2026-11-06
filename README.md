# EgoRecover

按 [plan.docx](plan.docx) 实现的第一视角在线全身运动重建：20 帧世界坐标身体历史、滚动窗口联合修正、预测中心 Flow 和完整后续响应的成对反馈监督。

当前仓库只维护这一套实现。清理前的代码、实验脚本和结果保留在 [归档分支](https://github.com/sxh-kk/EgoRecover/tree/archive/before-cleanup-20261106)。恢复方法及本次删除范围见 [重构记录](docs/refactor.md)。

## 安装

使用 Conda 管理环境；当前环境配置为 Python 3.12 / PyTorch 2.10 / CUDA 12.6：

```bash
conda env create -f environment.yml
conda activate egorecover
```

运行入口统一为 `egorecover`，也可使用 `python -m egorecover`。核心运行依赖只有 PyTorch；SMPL-X 和资产下载依赖按需安装。

## 最小验证

```bash
egorecover train --config configs/paired.json --synthetic --compact \
  --output exp/smoke --train-steps 2 --batch-size 1 --eval-every 1

egorecover evaluate --checkpoint exp/smoke/best.pt --synthetic \
  --output exp/smoke-evaluation --steps 1 2 4 10 --save-predictions

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

合成模式显式标记为诊断，用小网络和 dense 关节位置验证流程与梯度，不代表真实人体精度。

## EE4D 对比指标

当前只做 SMPL22 身体重建，包含左右手腕，不增加手指输出或 MPJPE-H。`evaluate` 自动生成 `report.json` 和可用于表格对比的 `metrics.csv`：世界坐标 MPJPE、逐帧 PA-MPJPE、手腕/头部误差、旋转矩阵误差、根轨迹、上游脚滑指标及单帧延迟。整段相似对齐误差另列，避免混淆不同论文的 MPJPE。CSV 分别输出 take 等权和 episode 等权统计，在线主结果使用 `rollout` 行；`matched` 是内部诊断。

TMR Semantic/FID 尚未接入；jerk 以明确的诊断名称输出，尚不作为 EgoForce 原论文 PJ/AUJ。字段定义、单位、对齐方式和排除项写入评估身份；详细口径见 [对比协议](docs/comparison.md)。

## 正式训练

当前只使用 EE4D-Motion，Nymeria 留待后续扩展。先做训练前核验；`preflight` 只恢复 manifest、P 及 train/dev 各一个 episode，不足以正式训练：

```bash
egorecover restore --preset preflight --root assets
# 将授权的 SMPL-X v1.1 SMPLX_NEUTRAL.npz 放在 body_models/smplx/
egorecover prepare --output exp/preparation --device cuda
```

`prepare` 检查划分、时间、坐标往返、真实 SMPL-X 解码、环境和资产身份，生成四轮配置、依赖及 Conda 命令，不启动实验。W=3 和默认训练预算均为待验证配置，窗口仍需在 dev 上选择。报告在 `exp/preparation/preparation.json`，缺少全量 train/dev 或 E7 时会明确列出。

再查看完整恢复空间并下载资产。`--dry-run` 只读取随包提供的索引，不联网、不创建目录。

```bash
egorecover restore --preset training --dry-run
egorecover restore --preset training --root assets
# 将自行授权下载的 SMPLX_NEUTRAL.npz 放在 body_models/smplx/
# 也可设置 SMPLX_MODEL_PATH 为该文件所在目录

egorecover experiments --window 3 --output exp/configs

egorecover train --config exp/configs/warmup/gaussian.json \
  --data-root assets/datasets/paired --prior-checkpoint assets/pretrained/prior.pt \
  --e7-checkpoint assets/pretrained/e7.ckpt --device cuda --seed 62 \
  --batch-size 64 --microbatch-size 8 \
  --output exp/seed62/warmup-gaussian

egorecover train --config exp/configs/feedback/paired.json \
  --data-root assets/datasets/paired --prior-checkpoint assets/pretrained/prior.pt \
  --initialize exp/seed62/warmup-gaussian/best.pt --device cuda --seed 62 \
  --output exp/seed62/paired

egorecover evaluate --checkpoint exp/seed62/paired/best.pt \
  --data-root assets/datasets/paired --prior-checkpoint assets/pretrained/prior.pt \
  --device cuda --output exp/seed62/evaluation --steps 1 2 4 10
```

`--resume` 用于同一实验的精确恢复；`--initialize` 从共同预热模型开始一轮新实验，两者不能混用。训练只在 dev 选模，test 评估需显式指定 `--split test`。窗口 3 是示例，正式实验应先在开发集完成窗口对照。

`--batch-size` 表示每次优化的有效样本数，`--microbatch-size` 控制单次并行的样本数，通过梯度累积保持有效预算。后续成对诊断在连续自主运行的缓存上，于多个时刻各写入一次参考标签，再评价后续 1/2/4/10 帧，不在每个时刻重置为 teacher 历史。

## 代码结构

```text
egorecover/
  models/          窗口重建器、冻结历史预测器、E7 注意力组件
  codec.py         243维表示及世界/窗口坐标变换
  state.py         身体缓存与有界观测包
  system.py        历史→P→中心/源→Flow→缓存更新
  flow.py          x0训练路径及可微Euler求解
  losses.py        当前监督、成对/直接后续监督、固定度量对照
  data.py          固定划分、缓存校验和训练样本
  timing.py        源帧网格与采样时间核验
  training.py      训练、开发集选模、恢复
  evaluation.py    连续运行与配对多步指标
  metrics.py       SMPL22 精度、时序、轨迹与脚滑指标
  evaluate.py      评估任务组织及预测导出
  checkpoint.py    E7/P导入与新版checkpoint身份
  assets.py        固定版本资产恢复
  preparation.py   训练前审计、预算与任务生成
  cli.py           唯一命令行入口
configs/           基础和成对反馈配置
tests/             当前方法、几何、时间、恢复和入口测试
docs/              方法、数据协议、重构与复核说明
```

方法细节见 [方法与实验设计](docs/method.md)，资产和输入格式见 [数据协议](docs/data.md)，外部方法与指标口径见 [EE4D 对比协议](docs/comparison.md)，实现边界与验证见 [复核记录](docs/review.md)。

正式框架默认保持 E7 的 768 维、12 层主干，可导入既有 E7 EMA 和校准 P。新原生 checkpoint 为 `egorecover_window_v2`，不直接续训清理前的 v1 诊断 checkpoint。重构前后完整尺寸网络加载相同 E7 权重后的窗口输出已经做数值比对。

当前已验证合成训练、梯度边界、精确恢复、权重迁移和命令行流程，并恢复了真实数据核验子集。授权 SMPL-X 的检查记录由 `prepare` 生成；正式训练和外部方法复现尚未完成，不能据此声称已有精度或 GPU 延迟收益。

代码来源及保留的第三方声明见 [SOURCE.md](SOURCE.md)。
