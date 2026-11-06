# EgoRecover

按 [plan.docx](plan.docx) 实现的第一视角在线全身运动重建：20 帧世界坐标身体历史、滚动窗口联合修正、预测中心 Flow 和完整后续响应的成对反馈监督。

当前仓库只维护这一套实现。清理前的代码、实验脚本和结果保留在 [归档分支](https://github.com/sxh-kk/EgoRecover/tree/archive/before-cleanup-20261106)。恢复方法及本次删除范围见 [重构记录](docs/refactor.md)。

## 安装

Python 3.11+，验证环境为 PyTorch 2.10 CPU。先安装适合机器的 PyTorch，再安装项目：

```bash
pip install -e '.[dev]'            # 核心代码和测试
pip install -e '.[body,assets]'    # 正式 SMPL-X 解码及 HF 资产恢复
# 或使用 environment.yml 建立 Python 3.11 / CUDA 12.8 环境
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

## 正式训练

先查看恢复所需的空间，再下载选定资产。`--dry-run` 只读取随包提供的索引，不联网、不创建目录。

```bash
egorecover restore --preset training --dry-run
egorecover restore --preset training --root assets
# 将自行授权下载的 SMPLX_NEUTRAL.npz 放在 body_models/smplx/
# 也可设置 SMPLX_MODEL_PATH 为该文件所在目录

egorecover experiments --window 3 --output exp/configs

egorecover train --config exp/configs/warmup/gaussian.json \
  --data-root assets/datasets/paired --prior-checkpoint assets/pretrained/prior.pt \
  --e7-checkpoint assets/pretrained/e7.ckpt --device cuda --seed 62 \
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
  evaluate.py      评估任务组织及预测导出
  checkpoint.py    E7/P导入与新版checkpoint身份
  assets.py        固定版本资产恢复
  cli.py           唯一命令行入口
configs/           基础和成对反馈配置
tests/             当前方法、几何、时间、恢复和入口测试
docs/              方法、数据协议、重构与复核说明
```

方法细节见 [方法与实验设计](docs/method.md)，资产和输入格式见 [数据协议](docs/data.md)，实现边界与验证见 [复核记录](docs/review.md)。

正式框架默认保持 E7 的 768 维、12 层主干，可导入既有 E7 EMA 和校准 P。新原生 checkpoint 为 `egorecover_window_v2`，不直接续训清理前的 v1 诊断 checkpoint。重构前后完整尺寸网络加载相同 E7 权重后的窗口输出已经做数值比对。

当前已验证合成训练、梯度边界、精确恢复、权重迁移和命令行流程；尚未在本环境恢复真实资产并进行训练，不能据此声称已有精度或 GPU 延迟收益。

代码来源及保留的第三方声明见 [SOURCE.md](SOURCE.md)。
