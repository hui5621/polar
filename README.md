# Polar Route Planning

Research code for polar route planning with fixed environment maps, S-IACO, and SAG-PPO.

## Structure

- `data/environments/projected_nsr_2025/`: Fixed environment maps.
- `src/polar_route/`: Algorithm, vessel model, and training tools.
- `results/paper/`: Experiment result tables.
- `DATA_SOURCES.md`: Environmental data acknowledgments.

## Run

Requires Python 3.12.6.

```bash
python -m pip install -e .
polar-train --algorithm all --scene all
```

The command runs both algorithms on the available scenes. New routes, training logs, checkpoints, and summary tables are written to `results/runs/`.
