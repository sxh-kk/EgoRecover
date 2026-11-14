# 训练服务器迁移清单：共享盘不计入迁移量

记录日期：2026-11-14，北京时间。适用范围为当前正式P02 A/B主线：复用选定的W3 G10k与既有G历史库，在新服务器从随机初始化训练P。

**需要传输的本地资产共172个文件、3,120,959,001字节，约2.9066 GiB。共享存储盘不扫描、不复制、不计入大小或校验文件。** 项目代码从GitHub取得，Conda安装及后续训练输出另计。

当前P02 A/B队列已按用户要求取消，两组正式训练均0步。迁移清单用于准备资产，后续开训需新的明确指令；既有代码、模型和缓存保留。本次只将清单、路径和校验值上传GitHub，模型及历史库通过服务器间文件传输。

配套文件：

- [机器可读清单、实际字节数及SHA256](reports/data/server_migration_manifest_20261114.json)
- [rsync逐文件列表](reports/data/server_migration_files_20261114.txt)
- [目标端SHA256校验文件](reports/data/server_migration_20261114.sha256)
- [当前实验主线及取消状态](formal_experiment_plan_20261112.md)

## 1. 先确认共享盘可复用

- [ ] 目标服务器能读取同一份官方数据目录：`/gaozt-test1/datasets/EE4D-Motion/ee4d_motion_uniegomotion/`。
- [ ] 目录内存在`takes.json`、`annotations/splits.json`，以及`uniegomotion/`下的动作、DINO特征、训练统计和官方验证参考文件。
- [ ] 若目标服务器的共享盘挂载路径不同，在新的运行计划中更新`data_root`；原数据文件保持不变。

官方数据保留在共享盘，不进入本地迁移包。纯P的2000步训练直接读取已合并历史库；后续G训练、D400/long50建库和完整评测仍需访问共享盘。如果目标服务器不能访问该共享盘，应先解决挂载，不能按本清单假定数据已经可用。

## 2. 必需迁移的本地资产

以下路径相对于项目根目录`/root/workspace/songxuhua/`。配套逐文件列表已经包含表中资产及元数据，无需手动选择整个实验目录。

| 资产 | 项目内位置 | 必要内容 | 实测大小 |
| --- | --- | --- | ---: |
| P历史库 | `exp/formal-p-w3-g10k-20261113/banks/` | train、D128、preflight-train、preflight-D128四份`bank.pt` | 2.4538 GiB |
| 历史库元数据 | 上述四个目录 | 各自`identity.json`、`selection.json`、`complete.json` | 约6 MiB |
| 选定W3 G权重 | `exp/formal-v3-10k-20261112/V3_G_CV_W3_10k_s62/` | `best.pt`及`identity.json` | 约351 MiB |
| G生产者冻结源码 | `exp/formal-v3-10k-20261112/source/` | 源码和资源，排除字节码 | 约1.1 MiB |
| P02 r1冻结源码 | `exp/formal-p02-ab-r1-20261113/source/` | 源码和资源，排除字节码 | 约1.4 MiB |
| SMPL-X模型 | `body_models/smplx/SMPLX_NEUTRAL.npz` | 当前使用的同一中性模型文件 | 约104 MiB |
| 固定评测清单 | `exp/formal-v1-20261112/configs/` | `dev128.json`、`evaluation400.json`、`long50.json` | 约155 KiB |
| 几何系数来源 | `exp/formal-p-w3-g10k-20261113/P01_W3G10k_s62/identity.json` | 记录固定几何系数的来源身份 | 约9 KiB |

### 四份合并库

```text
exp/formal-p-w3-g10k-20261113/banks/train/bank.pt
exp/formal-p-w3-g10k-20261113/banks/D128/bank.pt
exp/formal-p-w3-g10k-20261113/banks/preflight-train/bank.pt
exp/formal-p-w3-g10k-20261113/banks/preflight-D128/bank.pt
```

这些库保存官方数据上冻结G产生的历史及标签，已经足够复用，不需要再次生成train/D128库。`shard*`、`chunks`等重复生成中间文件不在清单内；迁移后也不需要重新执行这些旧目录的merge。

G生产者源码需要保留原版本：建库程序会对照G检查点核验代码身份，不能用当前工作区代码替换它。P02冻结源码用于保留本次修复版的准确实现，其中包含原工作区当时的辅助脚本；GitHub代码和冻结快照分别保留。

### 不需要搬的内容

- 共享盘数据、压缩包或整个`/gaozt-test1/`。
- 整个`exp/`、旧E7、失败P权重以及历次完整`resume.pt`。
- 重复历史库分片、建库chunk和CPU验证产生的模型/优化器包。
- 旧控制器PID、进程日志状态、锁文件或SIMPLE任务。
- Conda环境实体目录、Python字节码及缓存；环境在目标机重建。

本清单覆盖P02重新随机初始化、G评估及后续从best开始的适配准备。如果要继续某个已经训练中的G，须另行加入该运行的完整`resume.pt`、身份、日志及冻结版本，不能用`best.pt`代替原生恢复。

## 3. 取得代码并传输资产

训练代码基线为`9b7d64cc78473b025a729d8b8d70a0368c436b31`；包含本清单的文档提交没有修改训练实现。目标端应取得包含本清单的仓库提交。旧生产者的精确代码来自上表冻结快照。

### 目标服务器：取得项目

在专用项目目录执行，示例路径保持与源服务器一致：

```bash
mkdir -p /root/workspace
git clone git@github.com:sxh-kk/EgoRecover.git /root/workspace/songxuhua
cd /root/workspace/songxuhua
```

如果使用其他项目路径，随后建立目标服务器运行计划时统一重绑定。当前未提交的研究草稿不会随Git clone出现，按需单独备份；本清单覆盖正式主线所需资产。

### 源服务器：按白名单复制

执行前，源端和目标端都需要安装`rsync`。清单最初生成时源端尚未安装rsync，以下是通用迁移示例；本次实际通过共享盘完成传输，结果见第7节。以下变量需要替换成实际目标地址、端口和项目路径：

```bash
cd /root/workspace/songxuhua
migration_host='root@NEW_SERVER'
migration_port=22
migration_project='/root/workspace/songxuhua'

rsync -a --partial --info=progress2 \
  -e "ssh -p ${migration_port}" \
  --files-from=docs/reports/data/server_migration_files_20261114.txt \
  ./ "${migration_host}:${migration_project}/"
```

命令只传逐文件列表中的本地资产，目标项目目录由前一步clone建立。未使用`--delete`；无需打包或搬运共享数据。传输中断后可重跑同一命令。

### 目标服务器：逐文件校验

在目标项目根目录执行：

```bash
sha256sum -c docs/reports/data/server_migration_20261114.sha256
```

- [ ] 172个文件均显示`OK`。
- [ ] 四份bank与W3检查点的SHA同原协议一致；SMPL-X同原生产者资产一致。
- [ ] 固定128/400/50清单的内容未变化。

清单SHA已经在源服务器实际计算，原协议已声明的bank、G、SMPL-X、系数来源和评测清单SHA也逐项核对。没有读取或散列共享盘大文件。传输工具应保留时间戳；官方数据身份包含文件大小和`mtime_ns`，共享盘挂载后的可见身份仍须核对。

## 4. 在目标机用Conda重建环境

```bash
cd /root/workspace/songxuhua
conda env create -f environment.yml
conda activate egorecover
export SMPLX_MODEL_PATH="$PWD/body_models/smplx"
python -m pip check
```

`SMPLX_MODEL_PATH`必须指向模型所在目录。`environment.yml`固定PyTorch构建及Python 3.12系列；其他包不全部锁定，严格复现时需核对下表的源端实装版本，而不是只确认环境创建成功。

| 项目 | 源端实装版本 |
| --- | --- |
| Python | 3.12.1 |
| PyTorch | 2.10.0+cu126 |
| NumPy | 1.26.4 |
| SMPL-X | 0.1.28 |
| joblib | 1.6.0 |
| SciPy | 1.17.1 |
| pytest | 9.2.0 |

- [ ] 核对新机GPU、驱动与当前PyTorch CUDA构建兼容。
- [ ] 需要同时运行原A/B两组时，确认有8张可用GPU，每组4张。
- [ ] 复制后项目输出盘仍有至少10 GiB空闲，满足当前控制器的保护条件；Conda安装和未来输出空间另行预留，不计入2.9066 GiB迁移量。

## 5. 建立新运行计划，保留原科学设置

不要直接运行旧服务器已取消的`plan.json`。在新机建立独立运行目录，核对：

| 需重绑定的字段 | 处理方式 |
| --- | --- |
| 项目根、Python解释器、输出目录 | 使用目标服务器实际路径 |
| `producer.checkpoint`和`producer.source_snapshot` | 指向复制后的W3 best及原冻结源码 |
| `existing_banks.*.path` | 指向四份合并库；内容SHA不变 |
| SMPL-X目录和P01系数来源路径 | 使用目标路径；文件SHA不变 |
| 128/400/50清单路径 | 指向复制后的三个JSON；内容及selection ID不变 |
| `data_root` | 指向新机实际可读的共享盘挂载路径 |
| GPU编号、分布式端口 | 使用新机可用编号和端口 |
| 资源等待条件 | 不继承旧服务器SIMPLE PID；只针对目标机实际任务设定 |

已有`.pt`和固定清单保留原字节，不对内部历史路径进行批量替换；只修改新机协议中的定位路径。旧路径可以作为来源记录保留。对新配置重新生成身份和源码快照，不覆盖旧取消记录及其来源快照。

方法和预算保持：W3 G10k生产者、seed62、随机主干/零残差头、共同因果CV参考系目标、A预测beta/B继承beta、几何系数`0.008869197811349056`、global64/micro8、48成熟/16启动、原2000步门槛与10000步上限、8k后降学习率，以及固定评测片段。当前没有可续训的P02正式断点；CPU检查包不作为正式初始化。

## 6. 目标机验收与后续启动

可先在目标机做不占GPU的相关检查：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q \
  tests/test_rotation_validity.py \
  tests/test_body_fk.py \
  tests/test_p02_objectives.py \
  tests/test_formal_prior_training.py \
  tests/test_formal_prior_processes.py
```

- [ ] 共享盘挂载、资产SHA、Conda关键版本及相关CPU测试通过。
- [ ] 新机重新进行真实4卡预检：连续0→2对照0→1＋恢复1→2，跨一次学习率变化，比较模型、EMA、Adam、调度、采样及每rank CPU/CUDA随机状态。
- [ ] 两组匹配初始化、首次损失和查询trace通过；源端599项测试及4进程CPU证据不替代新机GPU预检。
- [ ] 获得后续明确开训指令后，再启动A/B各2000步，按原门槛决定续训。

本清单最初生成时仅完成源端清单和校验值；后续已按用户指令完成目标资产传输，实际结果见第7节。环境创建、GPU验收及训练尚未执行。取消状态见[2026-11-14取消记录](reports/data/formal_p02_queue_cancelled_20261114.json)。

## 7. 实际传输完成记录

2026-11-14 20:47（北京时间），172个文件已复制到目标服务器4953的`/root/workspace/songxuhua/`，逐文件目标端SHA256、大小及修改时间全部一致，总计2.9066 GiB。目标主机名为`di-20261106193215-2f6lr`。实际回执见[迁移完成记录](reports/data/server_transfer_4953_20261114.json)。

公网SSH路径超时后，使用已确认的内网SSH进行控制，通过共享盘中转本地资产；原共享EE4D数据没有复制，目标端可以访问它。目标机直接GitHub clone同样超时，因此代码通过校验后的Git bundle安装为已发布提交`b78ac41`，`origin`保留为GitHub仓库。该方法交付了同一Git版本，不表示目标机直连GitHub已恢复。

临时SSH公钥已从目标机删除，其他授权公钥保持原内容，源端临时密钥文件也已删除。目标机保留`exp/server-transfer-20261114/`下的代码、资产和清理回执。共享中转副本保留在`/gaozt-test1/datasets/EgoRecover-transfer/`，具体子目录记录在JSON回执中。

本次交付代码和资产，没有创建Conda环境、运行GPU预检或启动训练；源端P02队列仍为取消状态。后续从第4至6节继续准备，开训须新的明确指令。
