"""π₀ 与 π₀.₅：数据在模型里怎么走。从 LIBERO 仿真里截一帧观测，存成 trace_pi05.py 的输入。

和 openpi 官方评测客户端 examples/libero/main.py（commit 215abfb）的做法逐项一致：
256×256 渲染，先空跑 10 步等物体落稳，两路图像旋转 180 度、等比缩放补边到 224×224，
state 是末端位置 3 维 + 四元数转轴角 3 维 + 夹爪 2 维，prompt 是任务原文。

在 openpi 仓库根目录、激活 examples/libero/.venv（Python 3.8）后运行：
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
    MUJOCO_GL=egl python dump_libero_frame.py --out libero_frame.npz
课程附件里已经带了一帧（libero_spatial 任务 0、初始状态 0），没有 LIBERO 环境也能直接用 trace_pi05.py。

课程内部资料，版权所有，请勿外传。
"""
import argparse
import math
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESOLUTION = 256
NUM_STEPS_WAIT = 10


def quat2axisangle(quat):
    """robosuite 的写法（main.py 同款）：角度 2·acos(w)，不规范化到 [0, π]。"""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0, help="第几个固定初始状态")
    ap.add_argument("--out", default="libero_frame.npz")
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RESOLUTION, camera_widths=RESOLUTION)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task)[args.init])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    np.savez_compressed(args.out, image=img, wrist_image=wrist, state=state, prompt=np.array(task.language))
    env.close()
    print(f"{args.suite} 任务 {args.task}「{task.language}」初始状态 {args.init}")
    print(f"image {img.shape} {img.dtype}  wrist_image {wrist.shape} {wrist.dtype}  state {state.shape} {np.round(state, 3)}")
    print(f"写到 {args.out}")


if __name__ == "__main__":
    main()
