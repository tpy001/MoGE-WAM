# A100 RoboTwin 评测

项目：`/data3/tangpeiyuan/latentforceflow_0908`。更新日期：2026-10-28。
本文对应当前 A100 代码，不是 5090 的启动说明。

## 环境与依赖

- Policy：项目 `.venv/bin/python`，JAX/OpenPI 环境。
- 仿真：`/data1/user10/robotwin/env/bin/python`，已有 RoboTwin/CuRobo/SAPIEN 环境。
- RoboTwin：`/data1/user10/robotwin/RoboTwin`，需要机器人、物体资产及任务/指令配置。
- FastWAM：单 client 入口的 `--fastwam-root` 可指定已有安装；具体默认见该入口参数定义。
- CUDA：调度器使用 `/usr/local/cuda-12.4`；Vulkan ICD 为 `/usr/share/vulkan/icd.d/nvidia_icd.json`。
- 仿真环境需要 `websockets==15.0.1`、`msgpack`、`numpy`、`Pillow`、`PyYAML` 和 ffmpeg。
- Policy/调度环境还需项目依赖及 `pynvml`。环境、数据集、权重、外部 RoboTwin/FastWAM 不随这组评测代码提交。
- 当前外部 RoboTwin 相机配置中的 D435 为宽320、高240；换环境时需核对该 YAML，项目内脚本读取该文件，不硬编码此尺寸。

主要文件：

| 文件 | 作用 |
|---|---|
| `scripts/eval_robotwin_pi05_parallel.py` | 持久 server、任务队列、多 client、显存/线程检查、恢复 |
| `scripts/eval_robotwin_pi05.py` | 专用 policy server 和仿真 client 入口 |
| `scripts/robotwin_pi05_policy.py` | RGB/state/action 适配、chunk执行、client独立RNG |
| `scripts/robotwin_eval_loop.py` | 在线专家检查及原任务成功判定 |
| `scripts/robotwin_render_runtime.py` | 仅取观测时渲染及场景引用修复 |
| `scripts/robotwin_seed_cache.py` | 现仅保留MPLib引用修复和精简输出辅助函数，非seed缓存实现 |

RoboTwin输入适配、模型、增强和K=4配置已在仓库的 `src/openpi/` 中；无需使用通用 `scripts/serve_policy.py` 启动本评测。

## 批量评测

先看显卡是否空闲，再显式选择物理 GPU 编号0–7。一个调度器支持最多两张卡，每卡一个server、最多指定数量的client。
默认建议每卡2个client；多卡独立批次必须分配不重复的任务、不同输出目录和不同端口。

```bash
cd /data3/tangpeiyuan/latentforceflow_0908
nvidia-smi
CONFIG=pi05_robotwin_rgb
CHECKPOINT="$PWD/checkpoints/pi05_robotwin_rgb/robotwin_rgb_restored_20261019_201221/29999"
RUN_DIR="$PWD/evaluation_results/${CONFIG}_20ep_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python scripts/eval_robotwin_pi05_parallel.py \
  --gpus 0 --workers-per-gpu 2 \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --tasks adjust_bottle open_microwave move_can_pot \
  --phases clean random --episodes 20 --seed 42 \
  --replan-steps 24 --denoise-steps 10 --clear-cache-freq 1 \
  --client-reserve-mib 10240 --min-free-gpu-mib 6144 \
  --base-port 22780 --output-dir "$RUN_DIR"
```

- `--tasks all`：使用外部 RoboTwin 的官方50任务列表。
- `--episodes 20`：每任务每条件20次；全50任务clean/random共2000次。完整100次设置需显式传100。
- `--gpus 6 7`：使用两张空闲卡，不要同时占用其他批次正在使用的卡。
- `--video`：保存视频。默认不保存；未取新观测的chunk内复用图像。
- 长时间后台运行可在完整命令前加 `nohup`，末尾加 `> work/<独立日志名>.log 2>&1 < /dev/null &`，并记录 `$!`。
- `eval_robotwin_pi05_fastwam20.sh`、`eval_robotwin_pi05_fastwam50.sh` 是已有RGB批量包装器；任意Flow模型建议显式使用上述调度器并指定config/checkpoint。

Flow推理配置可选择：

- `pi05_latent_flow_joint_denoise_robotwin_latent005`：原Flow。
- `pi05_latent_flow_joint_denoise_robotwin_unclean_patchwam_latent005`：未清洗数据+PatchWAM增强，K=1。
- `pi05_latent_flow_joint_denoise_robotwin_unclean_patchwam_multinoise4_latent005`：同配方K=4训练。

`CHECKPOINT`必须替换为对应实验的步数目录，内含`params/`和`assets/`；模型配置和权重必须匹配。
使用checkpoint保存的归一化统计。权重路径是部署信息，不应把权重加入Git。

## 协议与输入

- 在线专家检查默认开启；先`play_once()`，再在关闭专家场景前检查`plan_success`与`check_success()`，不修改任务判定函数。
- 不使用专家seed缓存；当前批量入口拒绝`--precheck-only`、`--seed-cache-dir`和`--lightweight-rollout`。
- `clean` / `random` 对应 `demo_clean` / `demo_randomized`。
- 默认`unseen`指令。seed42对应首个候选seed4300000，专家不可执行的候选不计入20次。
- 三路RGB经输入适配：全局独立，左右腕上下拼接，然后resize/pad。
- state与输出action均为14维，顺序是左6关节、左夹爪、右6关节、右夹爪；action为绝对目标，不再做delta转换。
- 推理每次生成一组32步动作和内部latent，执行前24步再规划；K=4是训练的噪声分组数，不是推理时输出四组动作。
- Flow student推理不需要真实未来光流。
- 渲染默认只在模型取观测时触发；物理、成功判定继续按原流程执行。`--legacy-render-sync`是旧渲染同步参考模式。
- 显存分配比例、`clear-cache-freq=1`不能保证全部Vulkan/场景内存立即释放，需结合实际设备剩余显存观察。
- 任务/条件切换重建client，server保留权重和编译；首个模型调用可能需要较长编译时间。

## 单独启动server和client

一般让批量调度器自动启动即可。手动模式需分别使用项目Python与仿真Python，并配置相同GPU、PYTHONPATH、CUDA/Vulkan环境及端口；客户端需能找到ffmpeg。
在两个终端均先设置上面的`CONFIG`、`CHECKPOINT`及：

```bash
cd /data3/tangpeiyuan/latentforceflow_0908
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:/data1/user10/robotwin/env/bin:$PATH"
```

终端A：

```bash
.venv/bin/python scripts/eval_robotwin_pi05.py --serve \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --port 22780 --seed 42 --denoise-steps 10
```

等`server listening on 127.0.0.1:22780`后，终端B：

```bash
/data1/user10/robotwin/env/bin/python scripts/eval_robotwin_pi05.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --port 22780 \
  --tasks adjust_bottle --task-config demo_clean --episodes 20 \
  --instruction-type unseen --seed 42 --replan-steps 24 --denoise-steps 10 \
  --clear-cache-freq 1 --output-dir "$PWD/evaluation_results/manual_adjust_bottle_clean"
```

手动client退出不会自动停止手动server；需在server终端Ctrl+C。旧`eval_robotwin_pi05.sh`是固定RGB的单client包装器，仍有历史GPU0–3限制，不用于当前多卡Flow批次。

## 成功率、停止与恢复

输出结构：`run_config.json`、`summary.json/summary.csv`、`server_gpuN.log`、`TASK_PHASE/eval.log`、`TASK_PHASE/TASK/result.json`；启用视频时还包含MP4。
`summary.json`的完成数只含完整20次的任务/条件；未满20次进度从`eval.log`读取。

现有历史对照查看器：

```bash
watch -n 5 -t 'python3 scripts/show_robotwin_patchwam_k4_results.py'
```

显示RGB、增强Flow K=1、K=4。它依赖本机`work/robotwin_patchwam_k4_eval_active.json`和`work/robotwin_patchwam_eval_active.json`，这些运行记录不随Git提交，新机器需提供自己的对应记录。新批次不会自动注册到该表。
其他已存在的历史表是`show_robotwin_patchwam_results.py`和`show_robotwin_clean_flow_results.py`；`show_robotwin_eval_progress.py --manifest <路径>`接受其定义的多模型manifest格式。

前台Ctrl+C；后台先核实调度器PID，再`kill -TERM <本批调度器PID>`，调度器清理自己启动的进程。不使用全局`pkill python`。
用原输出目录和相同完整命令加`--resume`恢复：保留已完成task/phase，未完成的移到`incomplete_attempts/`并从头重测。
恢复必须保持checkpoint、任务/条件列表、次数、seed、replan、denoise和渲染协议相同；可改空闲GPU、端口和client并发数。

## 代码提交范围

评测入口、runtime辅助、shell包装器、结果查看器及本文需纳入Git。
不要提交`.venv`、外部仿真环境、`checkpoints`、`assets`链接、`work`、`evaluation_results`、数据集或模型文件。
旧`eval_robotwin_pi05_cached50.sh`及专家缓存测试不属于当前在线专家检查路径。
