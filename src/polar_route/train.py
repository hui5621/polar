from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .baselines import BASELINE_NAMES, run_baseline
from .metrics import route_metrics
from .sag_ppo import train_sag_ppo
from .scene import ROOT, list_scenes, load_scene
from .siaco import siaco


METRIC_FIELDS = (
    "success", "steps", "path_length_km", "total_cost", "mean_ice_concentration",
    "mean_ice_thickness_m", "mean_chart_risk", "current_assist_m_s",
    "ice_exposure_km", "turn_complexity_deg", "mean_turn_angle_deg",
    "mean_ais_corridor_score",
)


def _write_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_route(path: Path, route: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(route, dtype=np.int32), fmt="%d", delimiter=",", header="row,col", comments="")


def _scene_list(scene_arg: str) -> list[str]:
    if scene_arg.lower() == "all":
        return list_scenes()
    return [load_scene(scene_arg).date]


def _record(
    scene_name: str,
    algorithm: str,
    seed: int,
    route: list[tuple[int, int]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    scene = load_scene(scene_name)
    metrics = route_metrics(scene, route)
    return {
        "date": scene_name,
        "algorithm": algorithm,
        "seed": seed,
        "elapsed_seconds": round(elapsed_seconds, 3),
        **{key: metrics.get(key) for key in METRIC_FIELDS},
    }


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.out)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    dates = _scene_list(args.scene)
    summaries: list[dict[str, Any]] = []

    for date in dates:
        scene = load_scene(date)
        scene_dir = output_root / date
        route_dir = scene_dir / "routes"
        search_dir = scene_dir / "search_logs"
        checkpoint_dir = scene_dir / "checkpoints"
        selected = args.algorithm

        if selected in {"baselines", "all"}:
            for name in BASELINE_NAMES:
                started = time.perf_counter()
                route, history = run_baseline(
                    scene, name, seed=args.seed, ants=args.ants, iterations=args.iterations
                )
                elapsed = time.perf_counter() - started
                _save_route(route_dir / f"{name}_seed{args.seed}.csv", route)
                summaries.append(_record(date, name, args.seed, route, elapsed_seconds=elapsed))
                if history:
                    _write_dicts(search_dir / f"{name}_seed{args.seed}.csv", history)

        prior: list[tuple[int, int]] | None = None
        if selected in {"siaco", "sagppo", "all"}:
            started = time.perf_counter()
            prior, history = siaco(scene, ants=args.ants, iterations=args.iterations, seed=args.seed)
            elapsed = time.perf_counter() - started
            _save_route(route_dir / f"siaco_seed{args.seed}.csv", prior)
            _write_dicts(search_dir / f"siaco_seed{args.seed}.csv", history)
            summaries.append(_record(date, "siaco", args.seed, prior, elapsed_seconds=elapsed))

        if selected in {"sagppo", "all"}:
            assert prior is not None
            started = time.perf_counter()
            route, training_log, agent = train_sag_ppo(
                scene,
                prior,
                episodes=args.episodes,
                seed=args.seed,
                device=args.device,
                window=args.window,
            )
            elapsed = time.perf_counter() - started
            _save_route(route_dir / f"sagppo_seed{args.seed}.csv", route)
            _write_dicts(scene_dir / f"sagppo_training_seed{args.seed}.csv", training_log)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(checkpoint_dir / f"sagppo_seed{args.seed}.pt"))
            summaries.append(_record(date, "sagppo", args.seed, route, elapsed_seconds=elapsed))

    summary_path = output_root / "summary.csv"
    _write_dicts(summary_path, summaries)
    run_info = {
        "algorithm": args.algorithm,
        "scenes": dates,
        "seed": args.seed,
        "episodes": args.episodes if args.algorithm in {"sagppo", "all"} else None,
        "siaco_ants": args.ants,
        "siaco_iterations": args.iterations,
        "sagppo_window": args.window,
        "device": args.device,
        "summary_csv": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
    }
    (output_root / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the included polar route planners on fixed scene maps.")
    parser.add_argument("--algorithm", choices=("siaco", "sagppo", "baselines", "all"), default="all")
    parser.add_argument("--scene", default="2025-08-05", help="One scene date or 'all'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=300, help="SAG-PPO training episodes per scene.")
    parser.add_argument("--ants", type=int, default=30, help="Ants per S-IACO/ACO iteration.")
    parser.add_argument("--iterations", type=int, default=80, help="S-IACO/ACO search iterations.")
    parser.add_argument("--window", type=int, default=12, help="Prior route points per SAG-PPO local decision.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda.")
    parser.add_argument("--out", default="results", help="Output directory, relative to this repository by default.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run(args)
    print(f"Results written to {summary}")


if __name__ == "__main__":
    main()
