# Polar Route Planning

Research code for polar route planning, with fixed environment maps, S-IACO, SAG-PPO, comparison methods, and experiment results.

## Structure

- `data/environments/projected_nsr_2025/`: Fixed environment maps.
- `src/polar_route/`: Algorithms and training tools.
- `results/paper/`: Experiment result tables.
- `DATA_SOURCES.md`: Environmental data acknowledgments.

## Run

Requires Python 3.10 or later.

```bash
python -m pip install -e .
polar-train --algorithm all --scene 2025-08-05
```

Use `--scene all` to run the available scenes. New run outputs are written to `results/runs/`.
