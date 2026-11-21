"""仿真评测之后：把失败局按任务归类，排出下一轮补采的优先级（学员动手，CPU 秒级）。

读一份逐局评测记录（每局：任务、局号、成败、步数），对每个任务算成功率和 Wilson 95% 区间，
按失败数从多到少排，列出失败局号供看录像归因，再按「每个失败局补 N 条演示」给出建议补采条数。
记录里只能看出哪一局超时，失败原因（抓空、放错、卡住……）要看录像人工填。

用法（在解压出来的 recollect_sample/ 里）：python recollect_list.py libero10_raw_s43_episodes.json [--per-failure 2] [--csv 补采清单.csv]
只依赖标准库。

课程内部资料，版权所有，请勿外传。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项成功率的 Wilson 区间。n 小或 k 贴近 0、n 时比「p ± 1.96·标准误」靠谱。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def build_list(episodes: list[dict], per_failure: int) -> list[dict]:
    by_task: dict[int, dict] = defaultdict(lambda: {"task": "", "n": 0, "k": 0, "failed": [], "fail_steps": []})
    for ep in episodes:
        row = by_task[ep["task_id"]]
        row["task"] = ep["task"]
        row["n"] += 1
        if ep["success"]:
            row["k"] += 1
        else:
            row["failed"].append(ep["episode"])
            row["fail_steps"].append(ep["steps"])
    rows = []
    for task_id, r in by_task.items():
        lo, hi = wilson(r["k"], r["n"])
        rows.append({
            "task_id": task_id, "task": r["task"], "success": r["k"], "trials": r["n"],
            "rate": r["k"] / r["n"], "ci_low": lo, "ci_high": hi,
            "failures": len(r["failed"]), "failed_episodes": r["failed"],
            "all_failures_hit_step_limit": bool(r["fail_steps"]) and len(set(r["fail_steps"])) == 1,
            "recollect": per_failure * len(r["failed"]),
        })
    return sorted(rows, key=lambda r: (-r["failures"], r["task_id"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--per-failure", type=int, default=2, help="每个失败局建议补采几条演示")
    ap.add_argument("--csv", help="把清单另存成 CSV")
    args = ap.parse_args()
    data = json.loads(Path(args.records).read_text(encoding="utf-8"))
    episodes = data["episodes"]
    rows = build_list(episodes, args.per_failure)
    k, n = sum(r["success"] for r in rows), sum(r["trials"] for r in rows)
    lo, hi = wilson(k, n)
    print(f"{data.get('task_suite', '?')}：{len(rows)} 个任务，共 {n} 局，成功 {k} 局（{k / n:.1%}，95% 区间 {lo:.1%}–{hi:.1%}）")
    print(f"{'任务':<4}{'成功/局数':>9}{'成功率':>8}{'95% 区间':>16}{'失败':>5}{'建议补采':>8}  任务描述")
    for r in rows:
        print(f"{r['task_id']:<4}{r['success']:>5}/{r['trials']:<4}{r['rate']:>8.0%}   {r['ci_low']:>5.0%}–{r['ci_high']:<5.0%}  {r['failures']:>5}{r['recollect']:>8}  {r['task'][:48]}")
    top = rows[0]
    print(f"\n失败最多的任务 {top['task_id']}：失败局号 {top['failed_episodes'][:12]}{' …' if len(top['failed_episodes']) > 12 else ''}")
    if all(r["all_failures_hit_step_limit"] for r in rows if r["failures"]):
        print("所有失败局都跑满了步数上限：记录只说明「超时没完成」，具体败在哪一步要按局号看录像归类。")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task_id", "task", "success", "trials", "rate", "ci_low", "ci_high", "failures", "recollect", "failed_episodes", "failure_reason"])
            for r in rows:
                w.writerow([r["task_id"], r["task"], r["success"], r["trials"], f"{r['rate']:.3f}", f"{r['ci_low']:.3f}", f"{r['ci_high']:.3f}",
                            r["failures"], r["recollect"], " ".join(map(str, r["failed_episodes"])), ""])
        print(f"清单另存为 {args.csv}（最后一列 failure_reason 留空，看完录像自己填）")


if __name__ == "__main__":
    main()
