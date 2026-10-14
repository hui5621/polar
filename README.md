# Polar Route Planning

Polar route-planning research code with twelve fixed environment maps, S-IACO, SAG-PPO, comparison methods, and experiment results.

## Repository structure

```text
data/environments/projected_nsr_2025/  Twelve fixed projected environment grids and scene manifest
src/polar_route/                       Algorithms, scene loading, metrics, and training entry point
results/paper/                         Experiment result CSV files
DATA_SOURCES.md                        Environmental data attribution
```

## Setup and run

Requires Python 3.10 or later:

```bash
python -m pip install -e .
```

Run S-IACO and SAG-PPO on one scene with five random seeds:

```bash
polar-train --algorithm sagppo --scene 2025-08-05 --seeds 0 1 2 3 4
```

Run the complete algorithm set, including S-IACO, SAG-PPO, and comparison methods:

```bash
polar-train --algorithm all --scene 2025-08-05 --seeds 0 1 2 3 4
```

Run the global-planning comparison methods only:

```bash
polar-train --algorithm baselines --scene 2025-08-05 --seeds 0 1 2 3 4
```

Use `--scene all` to run all twelve scenes. Training defaults to at most 500 episodes and 420 decision steps per episode. Summaries, routes, training logs, model checkpoints, and search logs are written to `results/runs/`. The supplied result tables are stored in `results/paper/`.

## Algorithm modules

- `siaco.py`: S-IACO, ACO, and Zhou-IACO.
- `sag_ppo.py`, `ppo.py`: SAG-PPO and PPO policy training.
- `baselines.py`, `search.py`: A*, CA*, LoS-A*, and grid-search utilities.
- `metrics.py`: Route and environment metrics.
- `train.py`: Command-line training and result export.

See [`results/paper/README.md`](results/paper/README.md) for descriptions of the result files.
