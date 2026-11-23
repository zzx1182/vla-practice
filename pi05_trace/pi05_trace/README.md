# pi05_trace：看一帧数据怎么走进 π₀.₅

对应课程「π₀ 与 π₀.₅：数据在模型里怎么走」。要 NVIDIA 显卡和 openpi 原版环境（提交 215abfb）。

- trace_pi05.py：拿一帧 LIBERO 观测走一遍 pi05_libero 推理，打印变换链每一步的形状、
  前缀和后缀的 token 数、注意力掩码、推理耗时，把 10 步去噪逐步记下来，写成 trace_pi05.json。
- libero_frame.npz：现成的一帧（libero_spatial 任务 0、初始状态 0），两张 224×224 图、8 维状态、指令。
- dump_libero_frame.py：这一帧是怎么截的；装了 LIBERO 客户端环境的，可以换任务、换初始状态自己截。

用法：在 openpi 仓库根目录解压，得到 pi05_trace/ 文件夹，还在仓库根目录运行
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python pi05_trace/trace_pi05.py \
      --frame pi05_trace/libero_frame.npz --checkpoint-dir <pi05_libero 权重目录>

课程内部资料，版权所有，请勿外传。
