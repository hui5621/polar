# Polar route planning: S-IACO and SAG-PPO

北极航线规划的精简研究代码包，包含固定的 12 个 NSR 环境底图、S-IACO 全局路径搜索、SAG-PPO 局部路径优化训练，以及用于对照的传统规划算法。

仓库只保存已经处理好的栅格环境，不包含 NetCDF、海图原始文件、AIS 轨迹、参考航线或本机数据路径。数据来源与署名说明见 [`DATA_SOURCES.md`](DATA_SOURCES.md)。

## 目录结构

```text
data/environments/projected_nsr_2025/
  manifest.json                 # 固定场景清单、网格大小和起终点
  YYYYMMDD.npz                  # 每日处理后的环境栅格
src/polar_route/
  scene.py                      # 场景与栅格读取
  search.py                     # A*、局部栅格搜索和路径平滑
  siaco.py                      # ACO、Zhou-IACO、S-IACO
  sag_ppo.py                    # 六类局部算子、SAG-PPO 训练过程
  ppo.py                        # 离散动作 PPO Actor-Critic
  baselines.py                  # A*、CA*、LoS-A*、ACO、Zhou-IACO 对照
  metrics.py                    # 路径与环境指标
  train.py                      # 命令行训练与批量运行入口
results/                        # 运行后生成的结果目录
```

## 环境底图

底图来自论文实验固定的 12 个每日环境快照，日期为 2025-08-05、08-11、08-20、08-29、09-05、09-11、09-20、09-29、09-30、10-01、10-05 和 10-12。每张栅格为 325×209，网格间距 18 km，采用北极极射赤面投影。起点和终点以从上方起算的 `(row, col)` 保存。

每个 `.npz` 包含冰型、海冰浓度与厚度、海流、海图风险与阻塞掩膜、陆地和冰区阻塞掩膜、水深、预处理的航线走廊栅格，以及起终点。`manifest.json` 记录每个文件对应的日期和网格信息。

注意：2025-10-05 场景的保存环境中 `current_u`、`current_v` 和 `current_speed` 全为零；该日原环境记录没有 CMEMS 输入，因此这三个零值表示缺失数据，不代表观测到无流场。

## 安装与运行

需要 Python 3.10 或更高版本。安装包会自动安装 NumPy 和 PyTorch：

```bash
python -m pip install -e .
```

在单个场景上运行 S-IACO 并训练 SAG-PPO：

```bash
polar-train --algorithm sagppo --scene 2025-08-05 --seed 0 --episodes 300
```

运行所有传统对照算法：

```bash
polar-train --algorithm baselines --scene 2025-08-05 --seed 0
```

在全部 12 个场景上运行 S-IACO、SAG-PPO 和传统对照：

```bash
polar-train --algorithm all --scene all --seed 0 --episodes 300
```

`--episodes` 设置每个场景的 SAG-PPO 训练轮数；`--ants` 和 `--iterations` 设置蚁群规模与搜索轮数；`--device` 可设为 `cpu` 或 `cuda`；`--out` 可改结果目录。单独运行 S-IACO：

```bash
polar-train --algorithm siaco --scene 2025-08-05 --seed 0
```

## 算法

- **S-IACO**：根据当前场景栅格生成搜索参数，使用探索/开发双信息素构造候选路线，经局部安全捷径修复后按多目标非支配关系存档并选择先验路线。
- **SAG-PPO**：固定 S-IACO 全局路线，在局部区段上训练六种路径编辑算子的离散策略：保持先验、缩短重连、风险绕行、顺流偏移、转弯平滑和安全重连。策略使用有效动作掩码与有限幅度的候选评分偏置；硬安全检查未通过时回退至先验区段，Actor 只接收实际执行的提议动作的策略梯度。
- **传统对照**：A*、海流代价 A*（CA*）、视线平滑 A*（LoS-A*）、经典 ACO 和方向引导 Zhou-IACO。

实现使用同一套栅格、起终点、阻塞规则和路径指标；未加入按日期分支或训练后人工挑选结果。

## 运行结果

每次运行在 `results/<日期>/` 下保存结果：

- `summary.csv`：每种算法的成功标记、路径长度、综合代价、平均海冰浓度/厚度、海图风险、海流助益、冰区暴露距离、转弯复杂度和航线走廊得分。
- `routes/*.csv`：最终路径的 `(row, col)` 栅格坐标；SAG-PPO 运行也会保存对应的 S-IACO 先验路径。
- `sagppo_training_seed*.csv`：逐训练轮的奖励、路线指标、Actor/Critic 损失与熵。
- `checkpoints/sagppo_seed*.pt`：SAG-PPO Actor 与 Critic 权重。
- `search_logs/*.csv`：S-IACO/ACO 每轮的可行蚂蚁数、非支配档案规模和当前最优代价。
- `run.json`：本次场景、随机种子和训练参数。

已完成一次单场景短跑：2025-08-05、seed 0、3 个蚂蚁、2 轮搜索、3 轮 SAG-PPO。两种方法都生成了到达终点的路线；短跑的路线长度均为 5212.67 km，S-IACO 综合代价为 1097.62，SAG-PPO 为 1088.40。该小预算运行只用于展示输出格式和训练入口，不作为算法性能结论。

论文原实验报告中，S-IACO 在 12 个场景的聚合路线指标上优于 Zhou-IACO；SAG-PPO 的八项指标平均排名为 1.99，PPO 为 3.94，并在 96 组场景-指标比较中有 92 组不劣于 PPO。论文还报告了相对 PPO 的海冰浓度/厚度、转弯复杂度和冰区暴露距离改善。上述数字是论文结果摘要；本仓库提供的是便于直接运行的核心实现，默认训练不会自动复现论文全部配置、随机种子或表格数值。

本项目用于离线研究和算法比较，不是经认证的航海导航系统；环境栅格和规划输出不得用于实际航行决策。
