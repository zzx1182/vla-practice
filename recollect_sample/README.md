# recollect_sample：从评测失败局排出补采清单

对应课程「仿真评测：LIBERO 复现与测试逻辑」。只用 Python 标准库，普通电脑秒级跑完。

- recollect_list.py：读逐局评测记录，按任务算成功率和 Wilson 95% 区间，按失败局数排序，
  给出建议补采条数，可另存 CSV（最后一列 failure_reason 留给你看完录像自己填）。
- libero10_raw_s43_episodes.json：一次 LIBERO-10 的 500 局逐局记录（任务、局号、成败、步数）。
  模型是我在 libero_10 上用每任务 10 条演示微调的 π₀.₅，不是 openpi 官方权重，
  评测协议同 openpi examples/libero（replan 5 步、步数上限 520、seed 7）。

用法：进到解压出来的 recollect_sample/ 文件夹，运行
    python recollect_list.py libero10_raw_s43_episodes.json --per-failure 2 --csv 补采清单.csv

课程内部资料，版权所有，请勿外传。
