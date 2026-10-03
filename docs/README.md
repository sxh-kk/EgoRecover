# 文档索引

更新时间：2026-10-02。最新实验事实与运行状态统一记录在 [LOG.md](../LOG.md)。

## 当前实验

| 文档 | 内容与状态 |
|---|---|
| [G几何残差×实际历史四组设计](design/g-residual-history-four-way.md) | 已设计，尚未实现/启动；S0–S3检验G纠错结构及固定R0历史适配 |
| [G观测长度×设备参考](experiments/g-observation-reference.md) | 四组×三seed及最小评价完成；R0 131.91mm最好，80帧/设备参考未带来全身收益；[最终分析](../exp/egorecover_observation_reference/v2/analysis/final-conclusions.md) |
| [G重新训练四组对照](experiments/g-pretraining-four-way.md) | 09-30 14:05：12次训练及最小留出评估完成；混合历史有效，History源额外收益不稳定；T3 133.86mm，E7 115.02mm；[最终分析](../exp/egorecover_g_pretraining/v1/analysis/minimal-evaluation-conclusions.md) |
| [原E7最小适配四组对照](experiments/e7-minimal-ablation.md) | 已完成：Body59.96/75.25/178.49/187.81mm；地面估计为主因，滑窗速度误差明显增大 |
| [P 实际历史适应性](experiments/p-real-history.md) | 已完成：clean Body34.099→213.535mm；E7输入历史自身187.205mm，需核查历史质量 |
| [P 数据量与训练预算](experiments/p-data-budget.md) | D组已完成：192 take、562片段、9600步×三种子；Body 34.099mm，统一体型14.054mm |
| [P 统一体型FK监督](experiments/p-same-shape-fk.md) | 已完成：统一体型监督四权重×三个种子；[结果](../exp/egorecover_prior_same_shape/v1/RESULTS.md)；主线采用权重0 |
| [P 显式速度输入](experiments/p-explicit-velocity.md) | 已完成：常速度/FK=0，显式速度三种子配对；[结果](../exp/egorecover_prior_velocity/v1/RESULTS.md) |
| [P 常速度残差实验](experiments/p-cv-residual.md) | 已完成保持/常速度×FK四组、三种子对照；[结果](../exp/egorecover_prior_cv_residual/v1/RESULTS.md) |
| [P 的 MotionStreamer 改进](experiments/p-motionstreamer.md) | 上一轮方法：Two-Forward与FK四组对照；[已完成结果](../exp/egorecover_prior_two_forward/v1/RESULTS.md) |
| [P 实验运行说明](experiments/p-two-forward-runbook.md) | 上一轮入口、任务矩阵、输出与恢复；文内启动状态为历史记录 |
| [P 先独立验证的总顺序](experiments/p-first.md) | 独立验证 P，再决定是否接入 G |

## 设计与参考

| 目录 | 文档 |
|---|---|
| `design/` | [整体设计](design/blueprint.md)、[代码接口与坐标约定](design/interfaces.md)、[G重新训练的理论与文献核查](design/g-pretraining-review.md)、[G精度目标与下一步改进](design/g-targets-and-next-steps.md) |
| `data/` | [数据准备](data/dataset.md)；组件操作见 [data_pipeline](../data_pipeline/README.md) |
| `reference/` | [原始 E7](reference/e7.md) |
| `archive/` | [早期工程实验](archive/engineering-experiments.md)、[论文构想草稿](archive/iccv2027-draft.md)；保留历史语境，不代表当前结论 |

## 文件管理约定

发布文档放 `release/`：[实际迁移与复现](release/migration.md)、[大文件哈希与HF固定版本](release/artifacts.json)。[最初全量规划](release/github-huggingface-plan.md)保留盘点时的历史状态，实际发布范围与完成状态以迁移文档和 [发布状态](release/publication.json) 为准。

- 根目录保留 `README.md`（入口）、`LOG.md`（按时间追加实验事实）、`SOURCE.md`（来源）。
- 方法与复跑说明放 `docs/`，用小写英文和连字符命名；每份实验文档注明日期、状态、对应入口与输出位置。
- 同一方法更新原文档，实质改变实验协议时使用新的运行目录；不在根目录堆叠计划副本。
- 日志、checkpoint、缓存、图表、自动生成的运行报告放 `exp/<实验名>/<版本>/`，不混入方法文档。
- 组件自己的 README、数据交接和验证说明保留在组件目录。
- 本地链接相对当前文档定位；shell 命令均从仓库根目录执行。历史 LOG 中的旧路径保留，迁移对应关系见 2026-09-29 整理记录。

## 实验记录要求

标题统一使用 `## YYYY-MM-DD--HH：mm：实验设计概括`，例如 `## 2026-09-29--16：12：P 常速度残差扩量训练（192 take，FK=0）`。时间采用 Asia/Singapore（UTC+8），精确到分钟；标题概括实验设置或结论主题。历史记录原本未记时分的保留原日期，不补造时间。

[LOG.md](../LOG.md) 以实验结论为主，保留简短的配置、开始/完成/失败状态。每轮结束写清相对基线的改善或退化、消融因素的贡献、跨种子是否一致、短时自反馈是否退化，以及结论限制。

不逐次记录选点、step 或训练 loss。完整曲线、逐 take 指标和训练输出保存在各实验的 `selection.json`、`report.json` 和 `jobs/*.log`。LOG 中用可点击链接指向具体结果文件，不能只列无链接的目录名。未完成时明确说明结论尚未产生。
