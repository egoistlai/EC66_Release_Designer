# EC66 释放动作设计工作台（第一版）

先在 GUI 中设计 release 起始状态和释放阶段动作，观察理想携料初始化下的结果，再从固定起点搜索接近轨迹，比较真实接近后产生的颗粒分布。

本仓库已包含运行所需的运动学、轨迹规划、Newton 颗粒场景、机器人/铲子模型及597粒装料快照。克隆后安装本项目环境即可使用，不需要其他 EC66 项目、历史轨迹库或预生成缓存。

## 运行环境

以下是已验证的环境版本，完整 Python 包版本见 [`requirements-locked.txt`](requirements-locked.txt)：

| 项目 | 已验证配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS，x86_64；Windows 上通过 WSL2 + WSLg 运行 |
| Python | 3.12.15，由安装脚本下载到本项目 `.python/` |
| Newton / Warp | `newton==1.6.1` / `warp-lang==1.18.0` |
| 数值与几何 | NumPy 2.3.1、SciPy 1.18.1、trimesh 5.1.1 |
| GUI | pyglet 2.1.16、imgui-bundle 1.92.900、PyOpenGL 3.1.10 |
| 模型读取 | usd-core 26.8 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存 |
| NVIDIA 驱动 | 本机验证版本 610.60；此项是测试记录，不是最低版本声明 |
| 安装工具 | uv 0.13.0；Git 用于克隆仓库 |

图形界面需要可用的 OpenGL 桌面会话；WSL 环境需要 WSLg 和可用的 Windows NVIDIA 驱动。建议先确认 `nvidia-smi` 能识别 GPU。颗粒仿真使用 Newton XPBD / Warp CUDA；无 GUI 命令仍会执行物理仿真。当前未对原生 Windows 或 macOS 环境做验证。

## 安装与启动

在 Ubuntu / WSL 的 Bash 终端中执行。系统依赖和 uv 只需安装一次：

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip libgl1 libegl1 libx11-6 libxcursor1 libxi6 libxrandr2 libxinerama1
python3 -m pip install --user uv==0.13.0
export PATH="$HOME/.local/bin:$PATH"
```

克隆私有仓库时使用你自己的 GitLab 登录凭据，然后安装和启动：

```bash
git clone https://gitlab.dacheng.ai/Egoist-py/EC66_Release_Designer.git
cd EC66_Release_Designer
./INSTALL.sh
./CHECK.sh
./START.sh
```

`INSTALL.sh` 创建项目专用 `.venv/`，并按锁定清单同步依赖；首次安装需要联网下载 Python 和软件包。已有 `uv` 时可直接执行安装脚本，也可用 `UV=/path/to/uv ./INSTALL.sh` 指定工具路径。无需手动激活虚拟环境，所有启动脚本都使用本仓库 `.venv/bin/python`。`.python/`、`.venv/` 和 `.uv-cache/` 均不进入 Git。

启动脚本在 WSLg 可用、但终端未继承 `DISPLAY` 时自动选择 `:0`。没有桌面的机器可以通过 `./RUN.sh` 执行生成和仿真，或用 `./START.sh --viewer null --test --num-frames 3` 检查初始化。首次颗粒仿真可能需要编译 GPU 内核。

右侧 **Release Workbench** 是主操作窗口，可以移动、调整大小和滚动。左侧保留原 Newton 相机/渲染设置。界面使用英文标签以兼容原 Newton 字体。

## 推荐第一次使用

1. 打开 **Library**，选择 `03_large_pitch_dump`，点 **Load selected scheme**。这个低位65°翻铲示例已完成理想携料仿真，适合观察颗粒释放；`02_pitch_dump`是较温和的35°翻铲对照，默认`01_default`主要保料。
2. 在 **State** 中拖动位置、姿态、线速度和角速度滑条，默认开启 **Live solve while dragging**，机械臂随编辑实时更新。Ctrl+单击滑条数值可精确输入。可达性、位姿误差和速度误差同步显示；这一步只检查该状态。
3. 在 **Action** 中调整总时长、制动开始时间、翻铲角度和翻转窗口。两条曲线即时显示速度比例和附加翻转角。
4. 点击 **1 Build action**，检查完整动作机械可行性。到 **Results** 播放不含颗粒的机械臂动作。
5. 点击 **2 Simulate release**，从理想携料状态进行 Newton 颗粒仿真。完成后在 **Results** 播放、拖动时间，并查看分布热力图。
6. 满意后在 **Library** 命名并 **Save scheme as a new version**。保存只记录设计条件；仿真运行另有独立快照。
7. 点击 **3 Plan approach + compare full motion**，自动搜索固定起点到所选 release 的接近轨迹，再比较完整回放和理想预览。

后台任务不阻塞参数编辑。任务始终使用启动时的不可变快照；继续修改后，GUI 会显示 **EDITED**，明确当前显示的是旧条件的结果。失败会显示原因，不会偷偷降低速度、放宽限制或替换成历史成功轨迹。可以取消任务；取消留下的部分文件不算完成结果。

## 固定起点和坐标含义

接近规划使用以下固定的**机械臂基座坐标系法兰起点**：

```text
位置 m:        [-0.540661, -0.142302, 0.111460]
四元数 xyzw:  [0.559687, 0.660870, -0.376476, -0.329041]
关节角 deg:   [181.08705686, -63.00297723, 97.84592210,
               -4.63034441, -96.48870137, 175.34229846]
```

固定起点保存在 [`runtime/scene_config.json`](runtime/scene_config.json)，规划参数保存在 [`runtime/emcp_phase1_config.json`](runtime/emcp_phase1_config.json)。

GUI 的 `Position / RPY / velocity` 编辑的是**世界坐标系中的铲子工具点**；工具点在铲子坐标中的位置是 `[0.3, 0, 0.08] m`。它不同于法兰位置。RPY 为固定轴 XYZ 欧拉角；角速度为世界坐标 rad/s。锅内分布使用 Bucket-local XY 坐标，蓝/绿/红参考轴沿用原场景。青色点标记固定起点对应的工具点。

State 滑条范围：位置 X `[-1.35, 0.10]`、Y `[-1.70, 0.35]`、Z `[-0.25, 0.75]` m，RPY `[-180, 180]`°，线/角速度分量范围读取 `runtime/full_state_config.json`。保留输入的目标速度，不自动缩小到可行值；速度受限时显示 **LIMITED**，不可达位姿显示 **UNREACHABLE**，机械臂显示求解器找到的近似姿态。

黄色箭头表示期望线速度，紫色表示期望角速度；绿色/青色分别表示当前机械臂解算后的线/角速度。线速度箭头长度为速度乘0.3秒，角速度为角速度乘0.1 m/(rad/s)，零速度不画箭头。两组可分别关闭；完全匹配时箭头重叠。RGB轴标记目标姿态，绿色小点标记实际工具点。

编辑任意 State 滑条会退出轨迹回放，以前一次 release 解为种子实时求解。后台旧方案完成也不会覆盖新编辑的姿态。需要恢复旧回放时，在 **Results** 点击 **Show saved replay**；回放中的解算速度箭头使用当前帧的关节速度。关闭实时求解时，未求解的修改会显示 **UNSOLVED EDIT**，重新开启或点击 **Solve / show release state** 即可更新。

## 如何描述整个释放阶段

`t=0` 是释放动作开始时刻，不代表所有颗粒同时离铲。第一版提供“继续平移 + 附加翻转 + 平滑停止”动作：

| 控件 | 含义 |
|---|---|
| Duration | 从 release 到停止的时长，单位秒 |
| Brake begins | 制动开始时间占总时长的比例 |
| Brake shape | `early / balanced / late`：减速变化偏早、均衡或偏晚 |
| Axis | 工具局部 X/Y/Z 翻转轴 |
| Angle | 附加翻转总角度，度 |
| Flip begins / ends | 翻转时间窗，占总时长的比例 |
| Flip shape | 翻转进度偏早、均衡或偏晚 |

平移沿**起始线速度方向**，并按平滑速度曲线减速；移动距离由积分计算。不会同时强制指定时间、速度曲线和终点位移。如果希望沿初始法向移动，在 State 中把世界速度设为该法向乘目标速度。

非零起始角速度不会突然被清零：它在整个动作中平滑衰减，用户设置的工具轴翻转叠加在这个旋转上。因此 `Angle` 是附加角，不一定等于最终姿态与初始姿态之间的总夹角。翻转窗开始之前，初始角速度仍会造成旋转。

后端逐点连续 IK，再构造五次样条；初始速度严格接入，初始关节加速度由零工具加速度边界求得，终点关节速度/加速度为零。不能满足跟踪误差、碰撞或动态限制的动作会拒绝。

## 机械检查

- 关节位置、速度、加速度和 **绝对 jerk 上限**：对每一段多项式解析求极值，包含段内极值和两侧边界。
- 所有拼接点的 `q / qdot / qddot` 连续性。
- 锅体、自碰撞、工作空间、路径长度、绕行率、总关节行程、TCP 线/角速度和加速度检查。
- 期望笛卡尔动作与实际 FK 的最大位置/姿态误差。

碰撞检查使用采样几何近似；TCP 检查也是采样检查。Newton 使用规定的关节运动，尚未验证实际电机力矩、跟踪误差或硬件控制器。机械通过表示通过这些明确的模型与检查。

## 颗粒预览及结果含义

使用仓库内置的 Newton XPBD 场景、铲子几何和597粒装料快照：`runtime/scene_config_particles.json`、`runtime/assets/`、`runtime/settled_layered_particles_2cm.json`。

理想初始化直接放在 release 姿态，不插入额外静止等待，颗粒初速度为：

```text
v_particle = v_tool + omega_tool × (x_particle - p_tool)
```

`Initial load` 可以选择附加相对速度噪声；噪声标准差和种子可编辑。这仅是敏感性试验，不是现实装料分布的完整模型，也不声称是鲁棒性验证。改变初始姿态可能使该理想快照不可经前段实际获得。

结果同时显示：

- 实际速度和角速度曲线、瞬时在铲比例；
- 释放前最低保料率、release保料率、在铲颗粒相对刚体速度场的RMS；
- 24×24沉降质量热力图、有效沉降质量、锅外质量、铲内残留；
- 沉降稳定性判据；不稳定时可以增加 `Observe after stop` 再运行；
- 完整接近与理想初始化之间的质量栅格TV距离，包括未进入栅格的质量差异。

栅格按**初始总颗粒质量**归一化，不把落锅外或残留铲内的质量隐藏掉。锅外和铲内残留类别可能重叠，不可相加。保料判据是铲体局部mesh AABB外扩20mm，属于可与原数据对齐的几何代理。

## 固定起点接近搜索

第一版搜索：

- 不同时长的直接五次连接；
- 自动抬高越过锅沿的路径：在较高位置保持起点携料姿态，随后进入锁定的release状态；
- 可选的反向预备：沿释放速度反向0.15/0.30/0.45m，分别尝试固定起点姿态和释放姿态；
- 不同预备时刻和总时长；通过单个样条连接中间点，不把每个中间点强制设成零速。

机械检查通过后，先用现有效加速度/滑移模型排序，再对指定数量的候选做实际Newton仿真。最终按释放前最低保料率、相对滑动速度选择。搜索是有界的初版多初值搜索，不穷尽整个预备区域；没有找到候选不意味着数学上不可达。

完整仿真从固定起点加载颗粒，从起点一直运行到释放、停止和沉降；**在release处不重置颗粒**。记录器插入精确release帧。前段末端匹配后段 `q/qdot/qddot`，避免在接缝处突变。

低于保料目标的最佳结果仍保存并明确标记 **FAIL**，方便诊断。单次保料通过也不代表分布满足你的需求，或在其他装料情况下仍然通过。

## 保存与复现

`schemes/` 保存命名方案，每次保存创建新版本。`outputs/<时间>_<任务>_<唯一ID>/` 保存独立实验：

```text
design.json                  启动任务时的设计快照
provenance.json              设计哈希、源代码及关键配置哈希
status.json / worker.log     状态与后台日志（GUI启动的任务有worker.log）
release.npz                 已通过检查的释放轨迹，release事件在t=0
release_audit.json           实际release状态、完整机械检查
release_samples.npz         关节及工具点采样曲线
ideal/                      理想释放：replay、departure、distribution、release_particles
approach_search.json        所有接近尝试及拒绝原因
approach_XX/                各接近候选的完整轨迹、检查和颗粒数据
approach_results.json       实测排序和选择
result.json                 理想/完整结果及分布差异
error.txt                   失败时的详细错误
```

`distribution.npz` 包含630维分布标签。`release_particles.npz` 保存位置、速度、相对速度、在铲标记及关节状态。轨迹、设计条件和结果可以直接按运行目录配对。

方案、代码和模型资源纳入 Git；`outputs/`、录屏、环境目录和 GUI 布局缓存不上传。新克隆的仓库需要自行运行生成/仿真，才能在 Library 中看到本机历史结果。

## 仓库结构

```text
gui.py / cli.py               GUI 与命令行入口
design.py / motion.py         方案定义、释放轨迹与机械检查
approach.py / physics.py      前段搜索、连续颗粒仿真与分布记录
worker.py                     后台任务与运行快照
runtime/                      内置运动学、规划、场景及颗粒求解模块
runtime/assets/               EC66 URDF、连杆网格、铲子 STL
runtime/*.json                场景、运动限制、装料快照和分布配置
schemes/                      可直接加载的示例和保存方案
tests/                        数值、编辑器与独立运行检查
requirements-locked.txt       已验证的完整 Python 环境版本
INSTALL.sh                    安装项目专用 Python 和依赖
START.sh / RUN.sh / CHECK.sh   图形启动、命令行任务、回归检查
```

`runtime/SOURCE_MANIFEST.json` 记录内置代码和资产的来源哈希，仅用于追溯，不用于运行时定位。第三方声明见 `runtime/NOTICE.md`。

## 命令行与回放

无GUI运行：

```bash
./RUN.sh build --scheme schemes/02_pitch_dump.json
./RUN.sh simulate --scheme schemes/02_pitch_dump.json
./RUN.sh approach --scheme schemes/02_pitch_dump.json
# 更明显的颗粒释放示例
./RUN.sh simulate --scheme schemes/03_large_pitch_dump.json
```

打开某次已保存结果，无需重新仿真：

```bash
./START.sh --open-run /绝对路径/到/某次outputs运行目录 --tab Results
```

## 检查

```bash
./CHECK.sh
./START.sh --viewer null --test --num-frames 3
```

单元/数值测试覆盖速度曲线积分、非零起始角速度、工具局部转轴、连续路点、解析极值、加速度跳变、绝对jerk、固定起点和release事件序列化。实际Newton验证记录保存在本项目 `outputs/` 中。

当前已通过20项测试，包括4项真实IK/FK驱动的编辑器回归，以及内置资源完整性、运行模块定位和无相邻项目时移目录生成轨迹的检查。

独立版本已在无相邻项目的临时目录重新安装 Python 和依赖，并通过实际 OpenGL GUI、颗粒释放和完整接近流程验证。记录见 [`docs/standalone_validation.json`](docs/standalone_validation.json)。

低位65°翻铲的Newton8验证结果与迁移前一致：自动抬高0.15m越过锅沿，4秒接近，2秒释放动作，5秒沉降观察；释放前最低保料率99.665%，release相对速度RMS约0.00119m/s，最终有效沉降99.497%。这些是固定装料与当次求解器设置下的观测，不是对其他方案/材料的保证；生成的 `outputs/` 文件不随仓库分发。

生成完整结果后，将实际运行目录传给回放入口：

```bash
./START.sh --open-run outputs/实际运行目录 --tab Results --result-kind approach
```

这一验证运行关闭了反向预备、只筛选一个候选，但自动锅沿避让保持启用。方案库中的`03_large_pitch_dump`仍默认允许更广的反向预备搜索。
