# 代码、实验记录与资产迁移

日期：2026-10-02开始准备，2026-10-03完成本地复现核验。按项目所有者指令直接提交GitHub main，HF数据与模型均公有。全部选中资产已完成传输及远端对象核验；远程revision和核验记录见 [发布状态](publication.json)。每个恢复文件的路径、大小、SHA256和下载分组见 [artifact索引](artifacts.json)。

## 发布位置与范围

| 位置 | 内容 |
|---|---|
| [GitHub sxh-kk/EgoRecover，main](https://github.com/sxh-kk/EgoRecover/tree/main) | 当前工作树代码、配置、依赖、测试、文档、LOG、CV、轻量实验结论/统计、运行日志和冻结源码 |
| [HF sxhkk/EgoRecover，model](https://huggingface.co/sxhkk/EgoRecover) | 有效P/G权重、原E7初始化checkpoint、配对控制需要的P初始权重、当前R0与D组P的续训状态 |
| [HF sxhkk/EgoRecover-data，dataset](https://huggingface.co/datasets/sxhkk/EgoRecover-data) | 实际使用的processed EE4D、派生错配数据、冻结缓存、完整合并历史、最终评估证据 |

本次选中恢复文件10,582个，合计73.311 GiB：模型158个、25.625 GiB；数据/缓存/评估产物10,424个、47.686 GiB。数据以官方源归档和17个无损缓存/指标ZIP分片加轻量文件存储，HF数据有效载荷约40.983 GiB，恢复后仍是47.686 GiB的原字节。模型文件中有43个额外的同内容副本，15组重复SHA256。大权重和缓存ZIP使用最大64 MiB的内容寻址字节分片：模型368片、22.168 GiB；缓存142片、8.238 GiB。恢复器按顺序合并，验证分片和完整原文件SHA256，不改变模型张量。大小为逻辑大小，不能当作实测网络上传量。

所有选中大文件均重新计算完整SHA256，并检查读取过程中大小/mtime未变。权重和冻结数据直接保留原内容，未清洗身份、替换EMA或重新生成缓存。

官方35,067,467,786-byte ZIP从原HF仓库指定revision服务端复制，源归档SHA256已核对；本地8个processed数据成员与固定官方归档的大小/CRC全部一致。恢复时再检查归档和解压文件的完整SHA256。缓存分片写入时再次验证每个成员原SHA256，保留内容与路径；避免数千个小二进制独立上传的网络开销。Xet与原始大文件LFS上传都出现长时间未完成批次，部分请求遇到代理连接失败。已停止本次上传器，改为基础LFS上传64 MiB分片；所有选中分片已完成上传及远端哈希核验。没有调整GPU训练进程。

### 为什么选择这些checkpoint

- **当前可用组合**：R0三个seed的best及校准P，共1.096 GiB。校准P位于 `exp/egorecover_g_pretraining/v1/prior/prior.pt`，与原D_s62 P不同。
- **正式G消融**：T0–T3、R0–R3，各三个seed，共24个开发集选定best；复现全部已完成的四组对照。
- **P与早期G结论依据**：常速度残差、Two-Forward、FK/统一体型、显式速度、扩量、loss/replay及早期History/Gaussian等有效模型，保留对应报告与协议。历史实验模型不代表推荐部署配置。
- **41个小型初始权重**：已有汇总程序会比较同seed初始张量，验证公平对照，因此有复现用途。很多文件内容相同，由哈希辨认。
- **原E7完整checkpoint**：原EMA存于 `optimizer_states[0]['ema']`，重新制作缓存、初始化训练和原基线复现需要它。保留原文件，不把普通state_dict冒充EMA。
- **6个当前续训状态**：R0和D组P的seed62/63/64。包含优化器/EMA或采样与RNG状态，可保留当前工作；其余71个旧resume默认留在原机器，不上传。

不上传失败v1模型、smoke权重、中断/restart模型、大量旧resume及各模型last副本；失败过程的轻量配置、日志与结论仍保留。需要重新得到last时可重跑该协议；本次不承诺已保留全部旧训练状态。[排除清单](excluded-files.csv)说明本地剩余产物的类别及原因。

### 为什么保留这些数据产物

- 原processed数据的8个文件，35.531 GiB；来源为 [官方UniEgoMotion数据](https://huggingface.co/datasets/chaitanya100100/uniegomotion)。保留训练/验证运动、DINOv2、评估GT、统计和take/split身份，避免仅迁移模型却缺输入。
- 4,815个G paired episode及哈希标记：1,608个train take/4,369片段、224 dev和222 test。`g-evaluation`只恢复222个test episode与必要清单，不要求下载全部训练缓存。
- P原48take和扩量192take/562片段缓存、固定dev张量、模型启动体型和split身份；默认避免重算原E7产生的启动状态。
- 72take错配数据、bootstrap、最终合并的train/dev历史和最终预测证据，保留早期接口和实验的复现依赖。
- 原E7整窗预测、关键关节预测、正式G best最终输出与独立P最终dev输出，支持重算指标/查证结论。

不保留已经合并的history shard、P逐take构建的重复shard、所有中间G选点预测和合成smoke张量。相关小型选点曲线/统计仍在GitHub。原始视频、本地Git对象、pyc/pid/lock/测试缓存不上传。

## 在新机器恢复

```bash
git clone https://github.com/sxh-kk/EgoRecover.git
cd EgoRecover
conda env create -f environment.yml
conda activate egorecover
pip install -r requirements-reproduction-lock.txt
pip install -r requirements-release.txt
```

环境文件固定Python3.11、PyTorch2.10/CUDA12.8；CUDA不同则安装兼容PyTorch并记录版本。已记录本次验证环境的全部101个已安装分发包，运行/测试依赖的精确版本见根目录 `requirements-reproduction-lock.txt`；[环境快照](../../verification/migration_environment.json)保留Python及完整版本表。新机器从零完整安装尚未执行。仅查看文档无需训练环境。

| preset | 下载范围 |
|---|---|
| `core` | R0三seed best、校准P、归一化统计；约1.096 GiB。没有完整输入，不足以直接评估 |
| `g-smoke` | R0 seed62 best、校准P、统计和一个固定test episode；实际下载约0.537 GiB，含所在test分片 |
| `g-evaluation` | 24个正式G best、校准P、222take test缓存与清单；约8.957 GiB |
| `p-evaluation` | 各轮P权重及P原/扩量缓存；复现D组固定dev需要的原cache一起恢复 |
| `training` | 原E7、已选模型/初始权重、6个当前resume、原数据和训练/开发缓存 |
| `data` | 全部选中数据、缓存和最终预测/指标 |
| `all` | 全部发布大文件，约73.311 GiB |

```bash
python tools/restore_release.py --preset g-evaluation --dry-run
python tools/restore_release.py --preset g-evaluation
python tools/restore_release.py --preset g-evaluation --verify-only
```

下载器使用索引中固定的HF revision，按原相对路径恢复并验证每个文件大小/SHA256。对已存在但哈希错误的本地文件会明确失败，避免覆盖新的实验结果。删除/转移冲突副本后可再次恢复。公共仓库通常可直接下载，若上游访问条件变化需按HF提示登录。

字节分片下载到被Git忽略的 `data/.release-parts/`，合并后的归档保存在 `data/.release-archives/`，解压只处理所选文件；一个归档只下载/哈希一次，保留ZIP成员CRC及最终文件SHA256验证。恢复原processed source时需要下载完整官方ZIP；G评估/小规模检查只取对应prepared分片，不要求下载原始特征全集。空间需容纳归档和解压后的文件，可在恢复完成并校验后自行清理归档缓存。

### SMPL-X与共享目录

人体模型资产没有上传。取得授权的 `SMPLX_NEUTRAL.npz` 后置于 `body_models/smplx/`，或设置 `SMPLX_MODEL_PATH` 指向包含它的目录；需要与原实验资产的身份哈希一致。原文件版本/哈希见P报告和数据identity。[SMPL-X获取与条款](https://smpl-x.is.tue.mpg.de/modellicense.html)

恢复脚本将v1/v2的 `data/` 和 `prior/` 建为指向G-pretraining共享目录的相对软链接。一个episode缓存和一个校准P只下载一次，不重复复制。

## 验证已训练模型

```bash
# 小规模加载/前向检查：1个take，生成两帧，不是完整成绩
python tools/restore_release.py --preset g-smoke
python -m run.reproduce_release --model G --group R0 --seed 62 --smoke --device cuda \
  --output exp/reproduction/R0_s62_smoke.json

# 222take、t40…199、Euler10、draw1062
python -m run.reproduce_release --model G --group R0 --seed 62 --device cuda \
  --output exp/reproduction/R0_s62_full.json

# D组P：原12take固定dev、独立下一帧
python tools/restore_release.py --preset p-evaluation
python -m run.reproduce_release --model P --seed 62 --device cuda \
  --output exp/reproduction/P_D_s62.json
```

新入口调用现有模型、加载器和评价函数，比较复现Body数值与原报告；输出到新目录，不改已发布的指标。G的smoke只检查加载与两帧推理，不能与完整131.913mm均值比较。GPU/驱动/数值实现的变化可能产生浮点差异，报告会记录torch版本及偏差；不承诺不同机器上逐比特一致。

扩量P的 `original_data` 是旧机器绝对路径。新入口只在构造dev加载器时临时映射这个已知指针，仍执行原cache SHA、split、dev tensor和体型一致性检查。原 `egorecover/prior_expanded.py`、checkpoint、report和sequences内容未修改，历史身份哈希保持有效。

## 重跑训练与恢复的边界

源码、原协议、训练/开发/测试划分、已选权重、必要输入和当前resume已保存，可进行新输出目录的训练/评价。原调度器的plan可能包含历史机器绝对路径或依赖原运行状态，不应直接在已有输出目录套用路径替换后 `--resume`。

- 重新训练使用新的输出目录，将 `source_root`、`parent_run`、checkpoint/缓存路径定位到本地；冻结配置和identity原件只用于溯源。
- 原E7最小适配对照需要额外克隆 `sxh-kk/UEM-update`，固定提交 `156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08`；本仓库不重复包含整个上游副本。
- 严格续训会验证源码哈希、配置、缓存和RNG；模型架构文件原样保留。当前resume跨机器/路径的完整更新轨迹尚未验证，需显式迁移方案，不把“存在resume”当成续训已验证。
- S0–S3残差×R0历史方案仍仅设计，没有新训练或伪造的checkpoint。

## 发布核验记录

基础测试：`tests`和`data_pipeline/tests`共183项通过，63.65秒；5条框架/硬件提示。完整结果见 [迁移测试](../../verification/migration_tests.json)。新增恢复工具还检查了下载路径、分组和哈希；移位目录下的模型加载与P原dev指针验证见发布状态。

移位目录实际核验：G strict加载及两帧CPU推理通过；D组P seed62在完整原12take dev上为34.071924941mm，相对原报告−0.000001659mm。真实G缓存从无损分片解压后逐字节哈希一致，错误SHA256被拒绝并未留下目标文件。[G检查](../../verification/migration_g_smoke.json)、[P完整dev](../../verification/migration_p_dev.json)、[分片恢复](../../verification/migration_bundle_restore.json)、[源归档核对](../../verification/migration_archive_validation.json)、[字节分片恢复](../../verification/migration_parts_restore.json)。另外从固定公开HF revision实际下载5个原始文件、9个传输分片，在干净目录还原后校验全部SHA256，并完成G两帧CPU推理（4.65秒）：[远端下载复现](../../verification/migration_remote_g_smoke.json)。这是加载/评价/内容恢复核验，尚未验证另一台机器上的完整训练轨迹。

GitHub原有main历史保留，通过独立clone准备提交，不强推，不改训练模型源文件或运行中的进程。源码和轻量实验文件按白名单暂存，即使 `exp/` 被.gitignore忽略，也保留原相对路径，使LOG结论链接可点击。引用大文件的链接改为对应HF文件或分片索引；未选产物注明排除原因，不伪装成已上传。

代码沿用来源/文件内已有声明。数据、预测GT及模型资产保留各自来源条款，HF公有不会重新授权第三方数据；本项目没有给整个继承代码或数据添加未经核对的MIT许可。

最终发布检查：模型与数据仓库保持公有，所有选中远端对象的大小与LFS SHA256/普通Git blob ID均匹配；见[远端对象核验](../../verification/migration_remote_objects.json)。GitHub main保留原历史，并同步完成记录、复现入口和固定HF revision。
