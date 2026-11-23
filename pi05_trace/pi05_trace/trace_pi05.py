"""π₀ 与 π₀.₅：数据在模型里怎么走。拿一帧 LIBERO 观测走一遍 pi05_libero 的推理，把每一步打出来。

要 NVIDIA 显卡和 openpi 原版环境（commit 215abfb，GIT_LFS_SKIP_SMUDGE=1 uv sync 装好的 .venv）。
在 openpi 仓库根目录运行：
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python pi05_trace/trace_pi05.py \
        --frame pi05_trace/libero_frame.npz --checkpoint-dir <pi05_libero 权重目录>
不给 --checkpoint-dir 时用 gs://openpi-assets/checkpoints/pi05_libero，第一次会下载约 12.4 GB。

打印五段：
1. 输入变换链：Policy 里的每个变换依次作用在这一帧上，每步之后各字段的形状、类型、取值范围；
2. 进网络的 token：前缀（三路图像 + 指令）和后缀（动作）各多少个、哪些被掩码关掉；
3. 注意力掩码：几类 token 各能看到多少个 token，并把整张掩码按块压缩存成 JSON 给网页画图；
4. 推理：policy.infer 整体的耗时，以及只走 1 步去噪和走 10 步的耗时差（拆出前缀和每步去噪各花多少）；
5. 把 10 步去噪手动走一遍，逐步记下动作块，和 sample_actions 的结果对一下。

课程内部资料，版权所有，请勿外传。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config


def describe(x) -> str:
    if isinstance(x, str):
        return f"str {x!r}" if len(x) <= 60 else f"str {x[:57]!r}…"
    a = np.asarray(x)
    if a.dtype == np.bool_:
        return f"bool {tuple(a.shape)}，True {int(a.sum())} 个"
    if a.size == 0:
        return f"{a.dtype} {tuple(a.shape)}"
    return f"{a.dtype} {tuple(a.shape)} [{a.min():.3g}, {a.max():.3g}]"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        else:
            out[key] = v
    return out


def show(data: dict) -> dict:
    flat = flatten(data)
    for k, v in flat.items():
        print(f"    {k:<34} {describe(v)}")
    return {k: describe(v) for k, v in flat.items()}


def timed(fn, *args, repeat: int = 20, **kwargs) -> float:
    """先调一次触发编译，再测 repeat 次的中位数（毫秒）。"""
    jax.block_until_ready(fn(*args, **kwargs))
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args, **kwargs))
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def denoise_by_hand(model: _pi0.Pi0, obs_pre: _model.Observation, noise) -> list[np.ndarray]:
    """照抄 pi0.py 的 sample_actions（215abfb L217-279），只是不进 jit、每一步把 x 记下来。"""
    dt = -1.0 / 10
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs_pre)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=_pi0.make_attn_mask(prefix_mask, prefix_ar),
                                      positions=positions)        # 前缀只算这一次，K、V 进缓存
    x, t = noise, 1.0
    traj = [np.asarray(x[0], dtype=np.float32)]
    for _ in range(10):
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs_pre, x, jnp.full((1,), t))
        suffix_attn = _pi0.make_attn_mask(suffix_mask, suffix_ar)
        prefix_attn = jnp.repeat(prefix_mask[:, None, :], suffix_tokens.shape[1], axis=1)
        full = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm([None, suffix_tokens], mask=full, positions=pos, kv_cache=kv_cache,
                                                 adarms_cond=[None, adarms])
        v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        x, t = x + dt * v, t + dt
        traj.append(np.asarray(x[0], dtype=np.float32))
    return traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="libero_frame.npz")
    ap.add_argument("--checkpoint-dir", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--out", default="trace_pi05.json")
    args = ap.parse_args()
    record: dict = {"config": "pi05_libero", "steps": []}

    f = np.load(args.frame)
    obs = {
        "observation/image": f["image"],
        "observation/wrist_image": f["wrist_image"],
        "observation/state": f["state"],
        "prompt": str(f["prompt"]),
    }
    config = _config.get_config("pi05_libero")
    t = time.perf_counter()
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    print(f"加载权重 {time.perf_counter() - t:.1f} s")

    print("\n== 1. 输入变换链 ==")
    print("  客户端送来的一帧：")
    record["steps"].append({"name": "客户端送来的一帧", "fields": show(obs)})
    data = jax.tree.map(lambda x: x, obs)
    for tf in policy._input_transform.transforms:
        data = tf(data)
        name = type(tf).__name__
        print(f"  经过 {name}：")
        record["steps"].append({"name": name, "fields": show(data)})
    batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], data)
    observation = _model.Observation.from_dict(batched)
    print("  加 batch 维、转成 Observation（图像变成 [-1, 1] 的浮点）：")
    fields = {f"images/{k}": v for k, v in observation.images.items()}
    fields |= {f"image_masks/{k}": v for k, v in observation.image_masks.items()}
    fields |= {"state": observation.state, "tokenized_prompt": observation.tokenized_prompt,
               "tokenized_prompt_mask": observation.tokenized_prompt_mask}
    record["steps"].append({"name": "Observation", "fields": show(fields)})

    print("\n== 2. 进网络的 token ==")
    model: _pi0.Pi0 = policy._model
    obs_pre = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs_pre)
    noise = jax.random.normal(jax.random.key(0), (1, model.action_horizon, model.action_dim))
    suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs_pre, noise, jnp.ones(1))
    n_img = int(prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]) // len(observation.images)
    n_text = int(observation.tokenized_prompt_mask.sum())
    tok = {
        "prefix_shape": list(prefix_tokens.shape), "suffix_shape": list(suffix_tokens.shape),
        "tokens_per_image": n_img, "image_slots": list(observation.images),
        "image_valid": [bool(observation.image_masks[k][0]) for k in observation.images],
        "text_slots": int(observation.tokenized_prompt.shape[1]), "text_valid": n_text,
        "prefix_valid": int(prefix_mask.sum()), "suffix_len": int(suffix_tokens.shape[1]),
        "adarms_cond_shape": None if adarms is None else list(adarms.shape),
    }
    record["tokens"] = tok
    print(f"  前缀 {tuple(prefix_tokens.shape)}：{len(observation.images)} 个图像槽位 × {n_img} + 指令 {tok['text_slots']}，"
          f"其中真正参与的 {tok['prefix_valid']} 个（图像槽位有效 {tok['image_valid']}，指令实际 {n_text} 个 token）")
    print(f"  后缀 {tuple(suffix_tokens.shape)}：{tok['suffix_len']} 个动作 token；时间步不占 token，"
          f"走 adaRMSNorm 条件 {tok['adarms_cond_shape']}")

    print("\n== 3. 注意力掩码 ==")
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    attn = np.asarray(_pi0.make_attn_mask(input_mask, ar)[0])
    n_pre = prefix_tokens.shape[1]
    rows = {"主相机图像块": 0, "左腕图像块": n_img, "右腕槽位（空）": 2 * n_img,
            "指令第 1 个 token": 3 * n_img, "指令补齐位": 3 * n_img + n_text, "动作 token": n_pre}
    seen = {}
    for label, i in rows.items():
        row = attn[i]
        seen[label] = {"prefix": int(row[:n_pre].sum()), "suffix": int(row[n_pre:].sum())}
        print(f"  {label:<14} 能看到前缀 {seen[label]['prefix']:>3} 个、动作 {seen[label]['suffix']:>2} 个")
    record["mask_rows"] = seen
    bounds = [0, n_img, 2 * n_img, 3 * n_img, 3 * n_img + n_text, n_pre, n_pre + tok["suffix_len"]]
    names = ["主相机", "左腕", "右腕（空）", "指令", "指令补齐", "动作"]
    blocks = [[bool(attn[bounds[r]:bounds[r + 1], bounds[c]:bounds[c + 1]].any()) for c in range(6)] for r in range(6)]
    record["mask_blocks"] = {"names": names, "sizes": [bounds[i + 1] - bounds[i] for i in range(6)], "visible": blocks}

    print("\n== 4. 推理 ==")
    t = time.perf_counter()
    out = policy.infer(obs)
    print(f"  第一次 infer（含 XLA 编译）{time.perf_counter() - t:.1f} s")
    wall_ms = []
    for _ in range(20):                        # infer 返回前把结果转成 numpy，会等 GPU 算完，墙钟时间可信
        t = time.perf_counter()
        policy.infer(obs)
        wall_ms.append((time.perf_counter() - t) * 1000)
    acts = out["actions"]
    print(f"  输出 actions {acts.shape}，第一步 {np.round(acts[0], 3)}")
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    rng = jax.random.key(1)
    t1 = timed(sample, rng, observation, num_steps=1)
    t10 = timed(sample, rng, observation, num_steps=10)
    per_step = (t10 - t1) / 9
    record["infer"] = {"actions_shape": list(acts.shape), "first_action": np.round(acts[0], 4).tolist(),
                       "policy_infer_wall_ms_p50": float(np.median(wall_ms)), "sample_1_step_ms": t1, "sample_10_steps_ms": t10,
                       "per_denoise_step_ms": per_step, "prefix_ms_est": t1 - per_step,
                       "device": str(jax.devices()[0].device_kind)}
    print(f"  policy.infer 整体（含前后的变换）中位数 {np.median(wall_ms):.1f} ms（20 次）")
    print(f"  只走 1 步去噪 {t1:.1f} ms，走 10 步 {t10:.1f} ms → 每步去噪约 {per_step:.1f} ms，"
          f"前缀一次（含 SigLIP 编码图像）约 {t1 - per_step:.1f} ms")

    print("\n== 5. 把 10 步去噪一步步走出来（和 sample_actions 同一套计算，逐步记下 x） ==")
    traj = denoise_by_hand(model, obs_pre, noise)
    ref = np.asarray(sample(rng, observation, num_steps=10, noise=noise))[0]
    diff = float(np.abs(traj[-1] - ref).max())
    final = policy._output_transform({"state": np.asarray(batched["state"][0]), "actions": traj[-1]})["actions"]
    print(f"  手动走完和 sample_actions 的最大差 {diff:.4f}（bfloat16 计算，千分之几以内算一致）")
    print(f"  第 0 步（纯噪声）第一行前 7 维 {np.round(traj[0][0, :7], 2)}")
    print(f"  第 10 步第一行前 7 维 {np.round(traj[-1][0, :7], 2)} → 反归一化后 {np.round(final[0], 3)}")
    record["denoise"] = {"frames": [np.round(x[:, :7], 3).tolist() for x in traj], "max_abs_diff_vs_sample_actions": diff,
                         "final_actions": np.round(final, 4).tolist()}
    pathlib.Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"\n记录写到 {args.out}")


if __name__ == "__main__":
    main()
