# 主线清理与恢复

日期：2026-11-06。用户要求删除不再使用的文件，并把重构后的唯一主线更新至 `main`。

## 清理前版本

已在 GitHub 创建 [archive/before-cleanup-20261106](https://github.com/sxh-kk/EgoRecover/tree/archive/before-cleanup-20261106)，对应提交 `0b9a3b552e72cb6477b095fd4cfbccdafac152d1`。归档包含清理前完整代码、实验记录、配置、文档和首次窗口实现。

本次只创建正常后继提交，不修改旧提交、不强推。可以随时在 GitHub 浏览或下载归档，也可恢复具体文件：

```bash
git fetch origin archive/before-cleanup-20261106
git restore --source=origin/archive/before-cleanup-20261106 -- path/to/file
```

如需运行旧版本，应另行检出归档，避免把旧入口与新版包混合。

本轮将跟踪文件从 1,604 个收敛为 48 个；跟踪内容由约 94 MB 降至约 1.54 MB。索引压缩和历史产物删除均计入该变化，Git 历史对象与归档分支不在这个工作版本大小统计内。

## 删除范围与理由

| 原目录/文件 | 处理 |
|---|---|
| `exp/` 中已跟踪的历史实验报告、队列、统计与结果 | 从 main 删除；不再把旧协议的输出当成当前源码 |
| `run/` 全部旧实验、训练、诊断、调度和发布复现脚本 | 删除；当前命令统一进入 `egorecover` CLI |
| `motionstreamer_flow/` | 删除独立潜变量模型/训练/导出/测试；仍需要的纯时间验证移到 `egorecover/timing.py` |
| `data_pipeline/`、`dataset/` | 删除故障构造和旧训练数据管线；保留并独立实现当前缓存读取、时间协议及授权 SMPL-X 加载 |
| `model/`、`module/`、`mydiffusion/`、`eval/`、`utils/` | 删除旧单帧/多接口网络、Lightning/EMA 框架、旧采样和渲染/评估工具；必要注意力、连续时间、旋转及 FK 收进当前包 |
| `egorecover` 中 P/Q、replay、单帧、旧消融、调度等模块 | 删除；将 `window/` 收进包主层，移除对所有上述旧目录的依赖 |
| `config/` 及 portable 副本 | 删除；仅保留 `configs/base.json`、`configs/paired.json` 和可生成的四阶段矩阵 |
| 历史 `docs/`、`verification/`、`LOG.md`、`cv.md`、`图片和附件/` | 从 main 删除；新文档只描述当前方法、数据、重构及验证，旧资料在归档完整保留 |
| 旧发布工具、可视化/复现依赖清单 | 删除；按功能拆分为包的 body/assets/dev 可选依赖 |
| 原始方案、来源说明 | 保留 `plan.docx`；更新 `SOURCE.md`，保留仍使用代码的第三方声明 |

只删除了仓库已跟踪且已归档的历史产物；未自动删除用户未跟踪的数据或外部模型资产。

## 实质重构

1. 窗口网络直接继承 `torch.nn.Module`，通过 `ModelConfig` 管理尺寸，完整与诊断网络共用同一构造/前向逻辑。移除旧 `HistoryUniEgoMotion → UniEgoMotion` 继承、未使用文本分支及 YACS/Lightning/loguru 等依赖。
2. `checkpoint.py` 独立实现 E7 raw/EMA 与校准 P 导入。E7 三个旧冻结文本参数被显式识别并忽略；其余键/形状和 EMA 顺序严格核验后再写入模型，错误输入不会留下部分加载状态。
3. `codec/state/system/flow/losses` 成为唯一在线与训练实现。删除只供历史 SE(3)、保持姿态、全网格/55 关节渲染等旧分支；保留当前方案需要的平面参考、常速度 P 残差和 SMPL22 固定体型梯度路径。
4. `io.py` 统一原子保存与代码身份；`fk.py` 统一授权资产路径和解码；评估不再导入训练脚本获得辅助函数。
5. 合成 episode 与 dense 诊断解码从生产数据/损失模块分离至 `diagnostics.py`。生产数据加载、在线观测白名单与离线监督边界保持明确。
6. 统一 `egorecover train/evaluate/experiments/restore` 及 `python -m egorecover`。项目可通过 pyproject 安装、构建 wheel，离开源码工作目录仍可运行。
7. 资产清单只保留当前训练输入，压缩后约 520 KiB；HF 原 revision 与字节哈希不变，输出改为统一的 `assets/pretrained` 和 `assets/datasets/paired`，不重建旧实验目录树。
8. 新原生 checkpoint 格式为 `egorecover_window_v2`，包含明确模型尺寸；旧 v1/单帧/潜变量 checkpoint 不混作原生恢复。E7/P 是单独、受检查的初始化接口。

验证和局限见 [复核记录](review.md)。
