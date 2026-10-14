# Polar Route Planning

北极航线规划研究代码，包含固定环境底图、S-IACO、SAG-PPO、对照算法和论文实验结果数据。

## 目录结构

```text
data/environments/projected_nsr_2025/  固定的 12 个环境栅格与场景清单
src/polar_route/                       算法、场景读取、指标和训练入口
results/paper/                         论文实验结果 CSV
DATA_SOURCES.md                        环境数据来源
```

## 安装与运行

需要 Python 3.10 或更高版本：

```bash
python -m pip install -e .
```

在一个场景上使用 5 个随机种子运行 S-IACO 和 SAG-PPO：

```bash
polar-train --algorithm sagppo --scene 2025-08-05 --seeds 0 1 2 3 4 --episodes 300
```

运行五种对照算法：

```bash
polar-train --algorithm baselines --scene 2025-08-05 --seeds 0 1 2 3 4
```

`--scene all` 可用于全部 12 个场景。结果汇总、路径、训练记录、模型权重和搜索日志保存在 `results/` 下；论文结果数据保存在 `results/paper/`。

## 算法模块

- `siaco.py`：S-IACO、ACO 与 Zhou-IACO。
- `sag_ppo.py`、`ppo.py`：SAG-PPO 及 PPO 策略训练。
- `baselines.py`、`search.py`：A*、CA*、LoS-A* 和网格搜索工具。
- `metrics.py`：路线与环境指标。
- `train.py`：命令行训练及结果导出。

论文结果文件的内容见 [`results/paper/README.md`](results/paper/README.md)。
