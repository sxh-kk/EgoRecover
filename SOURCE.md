# 来源与保留范围

- 原始方法：UniEgoMotion，ICCV 2025，Chaitanya Patel 等；[上游仓库](https://github.com/chaitanya100100/UniEgoMotion)，参考提交 `580c92c6d70a91672c4106bab30ca82cdd80f379`。
- E7 来源：[sxh-kk/UEM-update](https://github.com/sxh-kk/UEM-update)，提取提交 `156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08`。
- 旋转函数继承原项目采用的 PyTorch3D 实现，版权/许可声明保留在 `egorecover/rotations.py`。
- 主干注意力和连续时间嵌入源于 E7。新版直接实现窗口网络，移除旧类继承、文本条件、Lightning 训练层及单帧接口；checkpoint 导入显式忽略旧的三个冻结文本参数，保持其余参数名称及 EMA 的学习参数顺序。
- 配对缓存来自项目已发布的 EE4D-Motion/DINOv2 数据及模型生成的启动状态。SMPL-X 资产仍需自行授权下载；本仓库不重新授权上游代码、人体模型或数据。
- 对比指标用 PyTorch 单独实现；脚滑定义核对了上游固定提交的 `eval/metrics.py`，保留其足点、阈值、按序列帧数归一化和毫米单位。论文指标来源及尚未对齐的口径见 [对比协议](docs/comparison.md)。

清理前完整工作版本保存在 [archive/before-cleanup-20261106](https://github.com/sxh-kk/EgoRecover/tree/archive/before-cleanup-20261106)，对应提交 `0b9a3b552e72cb6477b095fd4cfbccdafac152d1`。本轮只向 main 增加正常提交，不改写旧 Git 历史。
