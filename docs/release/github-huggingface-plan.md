# EgoRecover 全量盘点与 GitHub / Hugging Face 发布方案

> 这是最初仅盘点的历史规划；后续用户授权上传并指定HF公有。实际选择、恢复方法与发布状态见 [迁移与复现](migration.md)，不要将下文的“未上传”当作当前状态。

日期：2026-10-02，Asia/Singapore。状态：**已盘点、仅规划；未上传、未创建远程仓库、未构建压缩包。**

建议将当前代码及轻量实验记录发布到公开的 `sxh-kk/EgoRecover`，将自有推理权重发布到 Hugging Face 模型仓库，将续训状态作为可选归档。处理数据引用官方来源；派生缓存和含GT的预测产物先核对来源与分发条件。

## 1. 统计口径和逐文件清单

扫描根目录为 `/gaozt-test1/sxh/EgoRecover`，包含当前工作树、未跟踪文件、被Git忽略的实验及数据、`.git`；不沿软链接重复遍历，不计本次自动生成的 `exp/release_audit/20261002/` 报告。空目录无数据量，不进入逐文件统计。文件大小使用逻辑字节，GiB = 2³⁰ bytes，MiB = 2²⁰ bytes；这不是压缩后大小。

全目录约 **142.27 GiB，14,878个普通文件，4个目录软链接**；其中 `exp/` 为106.56 GiB，`data/` 为35.54 GiB。没有发现同inode的重复文件；尚未对所有大文件计算内容哈希，因此不能声称完成内容去重，也没有估算压缩收益。精确时间和字节数以扫描报告为准。

| 清单 | 用途 |
|---|---|
| [全量文件清单](../../exp/release_audit/20261002/files.csv) | 每一个文件的路径、分类、大小、mtime、inode；小型GitHub候选文本还记录SHA256 |
| [逐文件发布去向](../../exp/release_audit/20261002/release-manifest.csv) | 每个文件的建议平台、原相对路径、审核状态和用途；**是候选规划，不是可直接上传的许可清单** |
| [汇总统计](../../exp/release_audit/20261002/summary.json) | 分类、顶层目录、扩展名、最大文件、软链接和扫描限制 |
| [按实验统计](../../exp/release_audit/20261002/experiments.json) | 每个非空实验目录的文件量、总大小和各类产物大小 |
| [GitHub候选清单](../../exp/release_audit/20261002/github-candidates.csv) | 代码、实验文本、冻结源码、小型划分清单、图表及验证报告的候选集合 |
| [源码与验证文本](../../exp/release_audit/20261002/github-source.csv) / [实验元数据](../../exp/release_audit/20261002/experiment-metadata.csv) | 两类基础文本清单 |
| [权重清单](../../exp/release_audit/20261002/model-checkpoints.csv) / [权重发布索引](../../exp/release_audit/20261002/model-release-index.json) | 本项目202个模型文件，加1个原E7 checkpoint；标明核心推理与正式G消融 |
| [续训状态清单](../../exp/release_audit/20261002/training-resume.csv) | 77个resume文件及其实际大小 |
| [内容审核提示](../../exp/release_audit/20261002/content-review.json) | 机器绝对路径及强凭证模式扫描结果；不记录凭证值 |
| [checkpoint抽查](../../exp/release_audit/20261002/checkpoint-payloads.json) | 7个代表性文件的字段、张量大小、EMA形式和GT/预测行检查 |
| [软链接清单](../../exp/release_audit/20261002/symlinks.csv) / [恢复映射](../../exp/release_audit/20261002/symlink-restore.json) | 共享目标仅保存一次，迁移后建立相对软链接 |
| [外部E7参考仓库清单](../../exp/release_audit/20261002/upstream-reference-files.csv) / [统计](../../exp/release_audit/20261002/upstream-reference-summary.json) | 相邻原仓库另行盘点，不并入142.27 GiB |

## 2. 全量分类及建议去向

以下类别互斥，合计覆盖根目录内的所有普通文件。分类以路径、文件名和扩展名为依据，发布时还需检查内容；文件名带 `best` 不自动证明它是模型权重。

| 类别 | 文件数 | 实际体积 | 建议处理 |
|---|---:|---:|---|
| 当前代码、配置、依赖、文档 + 验证文本 | 243 | 约3.58 MiB | GitHub，核对当前工作树与文档 |
| 实验JSON/Markdown/CSV等元数据 | 992 | 54.85 MiB | GitHub轻量实验镜像，发布前检查绝对路径及内容 |
| 实验冻结源码 | 44 | 0.34 MiB | GitHub保留原路径；对应权重记录源码版本 |
| 本项目模型checkpoint | 202 | 37.69 GiB | 正式推理权重放HF模型仓库；初始/last/失败/smoke按需归档 |
| 续训状态 | 77 | 51.46 GiB | HF可选归档，包含优化器、EMA、RNG等 |
| 历史、bootstrap、episode等缓存 | 9,698 | 10.39 GiB | 派生数据审核；小型split/manifest/report候选单列GitHub |
| 预测、评估等张量产物 | 2,286 | 5.64 GiB | 可能含GT、观测和预测，审核后再确定归档范围 |
| 原E7 checkpoint | 1 | 1.32 GiB | 单独核对来源与发布权；专门导出EMA推理版本 |
| 原始处理数据及特征 | 8 | 35.53 GiB | 官方下载引用，不默认重复上传 |
| 本项目派生数据 | 671 | 6.79 MiB | 审核后选择生成脚本/清单或HF派生数据归档 |
| 实验运行日志 | 183 | 9.61 MiB | 可选归档；公开时检查路径及标识信息 |
| 实验图表 | 11 | 2.27 MiB | 小型结论图可放GitHub，姿态例图检查派生数据条件 |
| 验证二进制/图表/XML/smoke权重 | 9 | 18.38 MiB | 图表/XML候选GitHub；合成smoke权重无需作为正式模型发布 |
| SMPL-X模型资产 | 1 | 130.76 MiB | 排除GitHub/HF，提供授权下载及放置说明 |
| CV及图片附件 | 8 | 0.34 MiB | 人工筛选，默认不放进复现包 |
| `.git`历史与内部元数据 | 131 | 13.41 MiB | 不复制到新仓库；保留来源仓库与提交引用 |
| pyc、测试缓存、pid/lock等 | 313 | 2.98 MiB | 本地保留，发布包排除 |

基础轻量集合（代码+验证文本+实验元数据+冻结源码）约59 MiB。加上缓存目录中必要的轻量划分/清单以及小型图表，`github-candidates.csv`共1,318个候选文件、约64.14 MiB，精确字节数见 `model-release-index.json` 的 `github_candidate_bytes`。这些数字表示候选上限；公开版本应优先保留结论、协议、逐take指标、配置、身份和选点依据。

普通Git不适合提交上述模型和大数据。GitHub对超过100 MiB的文件拒绝普通Git推送，并建议仓库尽量低于1 GB。[GitHub大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

### 主要实验目录

| `exp/`内目录 | 普通文件数 | GiB | 发布时的角色 |
|---|---:|---:|---|
| `egorecover_observation_reference` | 430 | 42.095 | 最新R0–R3；v1失败记录与v2正式结果明确区分 |
| `egorecover_g_pretraining` | 10,047 | 40.542 | T0–T3、共享episode缓存、校准P |
| `egorecover_stages_1_4` | 1,544 | 10.492 | 早期四阶段实验与中断/重启历史 |
| `e7_official_full` | 6 | 2.670 | 原E7完整评估输出，含数据的二进制另审 |
| `egorecover_train_pilot_v1` | 128 | 1.725 | 初始训练、replay和预测历史试验 |
| `egorecover_loss_ablation` | 62 | 1.508 | geometry/FK与History/Gaussian消融 |
| `e7` | 2 | 1.320 | 原E7 checkpoint及引用元数据 |
| `egorecover_prior_cv_residual` | 144 | 1.140 | 保持/常速度×FK的P实验 |
| `egorecover_prior_same_shape` | 142 | 1.140 | 统一体型监督的P实验 |
| `e7_minimal_ablation` | 250 | 0.873 | 原E7滑窗、参考/地面等最小适配对照 |
| `egorecover_prior_two_forward` | 114 | 0.779 | Two-Forward/FK的P实验 |
| `egorecover_train_72take_v1` | 8 | 0.738 | 72 take工程训练 |
| `egorecover_prior_data_budget` | 431 | 0.636 | 扩大P训练数据/预算、D组三seed |
| `egorecover_prior_velocity` | 76 | 0.573 | P显式速度输入实验 |
| `egorecover_prior_real_history` | 64 | 0.164 | P对原E7实际历史的适应性 |
| `e7_official_key` | 6 | 0.153 | 原E7关键关节评估 |

其余小规模诊断及早期输出也已逐文件列入清单；完整39个非空实验目录统计见 `experiments.json`。目录大小含权重、resume和缓存，不能理解为最终模型的大小。

## 3. GitHub：当前代码及完整轻量实验记录

目标：公开仓库 `sxh-kk/EgoRecover`。**本地origin当前仍为 `sxh-kk/UEM-update`，未创建或推送新的EgoRecover仓库。** 使用当前工作树生成独立快照，并在 `SOURCE.md` 保留上游仓库、提交与文件来源。

应包含以下内容：

- `egorecover/`、`model/`、`mydiffusion/`、`module/`、`dataset/`、`utils/`：当前模型、表示、训练损失、采样、坐标、几何和指标实现。
- `run/`、`eval/`、`data_pipeline/`、`tools/`：P/G各轮训练、评估、队列、数据制作、统计和本次发布盘点工具。
- `config/`、`tests/`、依赖文件：全部已实现实验的配置和验证，以及尚未实现的新方案配置。明确新S0–S3仅设计，不能标成已实现功能。
- `README.md`、`LOG.md`、`SOURCE.md`、`docs/`：当前入口、项目来源、设计、各轮结论、状态与限制。
- `verification/`中的轻量验证文本及审核过的图表/XML。
- 轻量 `exp/<实验>/<版本>/` 镜像：保留原相对路径的 `RESULTS.md`、最终分析、summary/report、plan、selection、split/manifest、identity及源码快照等。

### 保证LOG结果链接可用

现有 `.gitignore` 忽略整个 `exp/`，普通 `git add` 会遗漏全部结果。因此发布准备时必须从逐文件白名单暂存轻量文件，保留 `exp/` 下原路径；不能直接对 `exp/` 全部强制添加。这样现有LOG和实验文档中的结论链接仍可跳转。

heavy checkpoint、完整运行日志或tensor产物用统一的artifact索引注明HF仓库、revision、下载相对路径、SHA256和恢复位置。链接只在实际仓库建立并上传后填写；本次不生成假下载地址。若某项仅本地留存，文档应明确“本地完整记录”，避免公开仓库中的链接指向缺失文件。

历史文档还有跳到相邻 `UEM-update-original` 的链接：公开版本应改为指定提交的GitHub链接，或保留必要的小型引用摘录并注明来源，不能要求用户拥有服务器上的同级目录。

### 发布前应修正文档与可移植性

1. README更新当前P常速度残差/FK=0、校准P、G的R0结论及最新复跑入口；旧Two-Forward和阶段1–4入口标明历史状态。
2. 来源与许可核对：本地未找到统一LICENSE；官方上游尚未核到可直接沿用的统一许可。保留文件内第三方许可头，不自行将继承代码和资产整体声明为MIT。
3. 本地有大量未跟踪的新模块和已有修改，`git archive HEAD` 会遗漏当前改进。按工作树候选清单建立独立发布暂存目录，再核对文件覆盖。
4. 绝对路径改为项目根目录、CLI参数或环境变量；历史证据和新可移植配置分别保留。

**不要原地改写已冻结的plan、manifest、源码快照和checkpoint identity。** 训练恢复会检查配置和源码/数据哈希；路径清洗也会改变哈希。发布版本应保存原身份与哈希，另建portable配置和显式迁移说明，证明加载/推理兼容；精确resume跨路径恢复需要另验证。

## 4. Hugging Face模型仓库：推理权重

建议仓库：`<HF_OWNER>/EgoRecover`，类型model。`HF_OWNER`和账号可用配额尚未检查，执行上传时再使用实际账号。模型卡列明架构、训练数据来源、协议、种子、对应源码、评估范围及失败/中断状态。

### 首选发布：当前可运行组合

| 文件 | 用途 |
|---|---|
| `exp/egorecover_observation_reference/v2/R0/seed62/best.pt` | R0/G，开发集选点18k |
| `exp/egorecover_observation_reference/v2/R0/seed63/best.pt` | R0/G，开发集选点22k |
| `exp/egorecover_observation_reference/v2/R0/seed64/best.pt` | R0/G，开发集选点24k |
| `exp/egorecover_g_pretraining/v1/prior/prior.pt` | 三个G共同依赖的已校准P |

这4个文件总共 **1.0959 GiB**，只是权重，不含代码、数据统计或SMPL-X依赖。三seed都发布，避免按测试成绩挑选最好的seed；可按固定seed62提供示例，不把它宣称为独立选出的最优模型。

**当前G使用的校准P与 `egorecover_prior_data_budget/.../D_s62/prior.pt` 不同。** 原D组三seed的P也应作为历史独立P实验发布，但不能直接替代校准P复现当前R0结果。v1/v2的 `prior/` 软链接均指向上述共享校准P。

### 完整模型对照与归档层级

- T0–T3三seed及R0–R3三seed：共24个正式 `best.pt`，合计 **8.6302 GiB**；已经包含核心3个R0，不能再累加一次。
- 各轮正式P、早期G及loss/replay实验模型：依据 `model-release-index.json` 核对实验完成状态后发布，记录原协议和指标，不能统称为当前推荐模型。
- 全部202个模型checkpoint合计37.69 GiB，其中有 `best.pt` 36个、`last.pt` 36个、`prior.pt` 61个、`g_gaussian.pt` 15个、`g_history.pt` 13个、`initial.pt` 41个。12个 `best.pt` 来自smoke/验证，不计入24个正式G。
- `initial.pt`、last、失败v1、smoke、中断模型作为可选实验归档。先保留原件，再依据内容SHA256判断是否能在归档索引中共享对象；本次未删除任何文件。
- 推理导出优先只包含state_dict及必要模型配置/格式版本，并清理公开副本中的服务器路径。原checkpoint保持不变；导出前后固定输入的数值对照以及原始文件SHA256应进入模型卡。

### 原E7的特殊处理

`exp/e7/last.ckpt`为1.32 GiB，其中含Lightning训练状态。其EMA在 `optimizer_states[0]['ema']`，原state_dict与EMA不是同一份权重。**不能简单去掉optimizer后发布并声称保留了原E7 EMA。** 应沿用 `egorecover/checkpoint.py` 和 `module/ema.py` 的已知格式，按原模型参数顺序应用EMA，再导出兼容加载格式并核对推理输出。

抽查原state_dict张量约0.357 GiB，纯推理导出体积预计在这个量级，尚未实际导出。原E7的权重来源及发布权须单独确认；官方Diffusion权重不能当作E7 checkpoint替代。

## 5. Hugging Face实验归档：按需保留续训与重算成本

建议仓库：`<HF_OWNER>/EgoRecover-experiment-archive`，类型dataset，用于文件归档；不是宣称这些产物构成可自由再分发的新数据集。

| 归档内容 | 大小 | 必要性与边界 |
|---|---:|---|
| 77个resume | 51.46 GiB | 继续现有训练需要；仅做推理无需下载 |
| 初始/last/smoke/失败权重 | 见逐文件清单 | 可审计训练过程与失败；与公开模型层去重 |
| 完整运行日志、原身份及冻结源码 | 日志9.61 MiB；其他见清单 | 审计/恢复用途，检查路径及标识；禁止悄悄替换冻结证据 |
| 缓存与历史 | 10.39 GiB | 可重算，数据条件允许才归档；共享缓存只存一次 |
| 预测/评估tensor | 5.64 GiB | 包含GT的文件按派生数据处理；轻量数值结论放GitHub |
| 派生数据 | 6.79 MiB | 默认提供生成脚本及清单，是否归档取决于来源条件 |

不能把全部实验产物默认为公开个人备份，也不能认为设置私有就解决数据分发条件。HF公开存储按平台政策提供；免费私有存储目前为100 GB，按账号整体计算。模型checkpoint+resume约 **95.72 GB（89.15 GiB）**，完整 `exp/` 约114.42 GB，单是后者已超过100 GB；既有仓库用量未检查。[HF存储限制](https://huggingface.co/docs/hub/storage-limits)

HF推荐每目录不超过10k文件。当前共享episode目录有9,630个文件，尚未超限但已接近；将来扩量应分目录或分片。归档可采用2–5 GiB分片，保留清单和哈希，按实验/用途可独立下载；不应生成一个143 GiB整体tar并盲目上传。分片大小是建议，压缩体积尚未测量。

### 数据和人体模型的处理

- `data/ee4d_motion_uniegomotion/` 的8个文件约35.53 GiB：原processed motion、DINOv2特征、GT评估文件、统计、takes与splits。优先写官方下载来源、预期路径和哈希，提供恢复检查，不重复占用自己的存储。[UniEgoMotion数据说明](https://github.com/chaitanya100100/UniEgoMotion/blob/main/DATASET.md)、[官方HF数据仓库](https://huggingface.co/datasets/chaitanya100100/uniegomotion)。
- EgoExo4D数据获取需要接受许可，派生GT/特征/缓存的公开范围须依实际适用条件确认；本次并未确认所有二进制产物可自由分发。[EgoExo4D获取说明](https://docs.ego-exo4d-data.org/getting-started/)
- `body_models/smplx/SMPLX_NEUTRAL.npz`：SMPL-X模型许可限制向第三方分享/分发，需事先许可，因此不放GitHub/HF发布包。发布模型文件哈希、版本及放置路径，让使用者自行授权下载。[SMPL-X模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)
- checkpoint样本未发现嵌入SMPL资产的字段名，但这只覆盖7个代表性文件，不能替代对全部发布权重的内容检查。

## 6. 复现依赖、目录及恢复策略

建议在HF保持原相对路径；若重排模型目录，必须在artifact索引中保留“HF路径→本地原路径”的显式映射。共享资产下载一次，4个别名根据 `symlink-restore.json` 恢复为相对软链接，不能打包时沿软链接把同一缓存复制三遍。

```text
GitHub sxh-kk/EgoRecover
  当前源码 + configs + 环境 + docs/LOG
  exp/<run>/<version>/  轻量结论/统计/身份/源码快照
  docs/release/          发布说明及精简artifact索引

HF <HF_OWNER>/EgoRecover                    model
  exp/.../best.pt, prior.pt, 历史正式G/P
  模型卡 + 文件SHA256 + 原始源码/协议标识

HF <HF_OWNER>/EgoRecover-experiment-archive  dataset，按需
  resume、归档权重、日志、审核允许的缓存/预测
  分片索引 + 哈希 + 软链接恢复映射

外部下载
  processed EE4D + 训练归一化统计
  经授权下载的SMPL-X
  原E7（发布权确认后可另提供EMA导出）
```

复现最新R0最低需要：当前匹配源码/配置、G的一个seed best、共享校准P、`v4_beta_ee_train_stats.pt`、对应观测/GT数据，以及需要FK解码和指标时的SMPL-X。不能仅下载G/P权重就声称可复现评估。

重新制作缓存/训练还需要原E7 EMA、原D_s62 P与校准过程、训练/开发/测试划分、bootstrap协议、体型/地面/参考坐标约定、固定dev monitor和采样种子。准确续训还依赖resume、冻结身份、共享episode缓存及原代码；异机/异路径的恢复尚未验证。

外部原仓库 `/gaozt-test1/sxh/UEM-update-original` 已额外统计：191个普通文件约20.02 MiB，其中非Git内容164个约6.73 MiB；工作树干净，提交 `156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08`。它用于原E7对照，计划记录指定提交的获取方法，不把整个副本重新塞入EgoRecover发布仓库。

## 7. 实际发布时的顺序与核验

1. 明确代码/权重/数据的适用许可，保留第三方声明；核对HF账号和可用存储。SMPL-X默认排除。
2. 从候选清单制作独立staging目录；更新README、portable配置和结果链接；保留原工作树和原身份文件。
3. 核验GitHub暂存清单覆盖当前改动/未跟踪模块，文件大小符合平台限制，没有凭证、缓存、GT或人体模型误入；所有公开的LOG结论链接可打开。
4. 对拟上传大文件计算SHA256、核对checkpoint内容；生成模型卡/下载恢复索引，验证EMA和校准P选择。需要归档时再进行内容去重和分片。
5. 在独立目录按依赖清单做最小加载/推理与结果读取验证。重新训练所有实验不属于发布核验；跨机器精确resume不能未经测试就承诺。
6. 最后执行创建仓库、上传、回读校验和HF链接替换，记录实际revision与最终字节数。

本次只完成统计、候选去向和规划文档，不执行以上发布动作，也不更改origin或训练进程。强凭证模式扫描未发现匹配项；它仅检查不超过16 MiB的文本，不覆盖Git对象历史和全部二进制。机器绝对路径文件清单见 `content-review.json`，不能把“未发现模式”当作完整安全证明。

重新盘点命令（只写本地报告，无上传）：

```bash
cd /gaozt-test1/sxh/EgoRecover
/root/miniconda3/envs/egorecover/bin/python tools/audit_release_inventory.py \
  --output exp/release_audit/20261002
```

该命令刷新根目录清单与去向建议；外部参考仓库统计和checkpoint抽查为本次另外完成的证据，不会由此命令自动重跑。
