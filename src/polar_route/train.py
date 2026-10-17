from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import CONFIG, ALGORITHM_SETTINGS
from .metrics import route_metrics
from .sag_ppo import train_sag_ppo
from .scene import ROOT, list_scenes, load_scene
from .siaco import siaco


SUMMARY_METRICS = (
    "PathLength", "TotalCost", "MeanIce", "MeanThick", "CurrentAssist",
    "TurnComplex", "TravelTime", "IceExposureDistance", "FuelTons",
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
    np.savetxt(path, np.asarray(route, dtype=np.int32), fmt="%d", delimiter=",",
               header="row,col", comments="")


def _scene_list(scene_arg: str) -> list[str]:
    if scene_arg.lower() == "all":
        return list_scenes()
    return [load_scene(scene_arg).date]


def _record(scene: Any, algorithm: str, seed: int, route: list[tuple[int, int]], *, elapsed: float) -> dict[str, Any]:
    metrics = route_metrics(scene, route)
    return {
        "date": scene.date,
        "method": {"siaco": "S-IACO", "sagppo": "SAG-PPO"}.get(algorithm, algorithm),
        "seed": int(seed),
        "elapsed_seconds": round(float(elapsed), 3),
        "success": bool(metrics["success"]),
        **{metric: metrics[metric] for metric in SUMMARY_METRICS},
    }


def _save_training_outputs(scene_dir: Path, method: str, seed: int, route: list[tuple[int, int]], history: list[dict[str, float]], agent: Any | None = None) -> None:
    slug = method.lower().replace("-", "_")
    _save_route(scene_dir / "routes" / f"{slug}_seed{seed}.csv", route)
    if history:
        _write_dicts(scene_dir / f"{slug}_training_seed{seed}.csv", history)
    if agent is not None:
        checkpoint = scene_dir / "checkpoints" / f"{slug}_seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint))


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.out)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    dates = _scene_list(args.scene)
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("At least one random seed is required")
    summaries: list[dict[str, Any]] = []

    for date in dates:
        scene = load_scene(date)
        scene_dir = output_root / date
        route_dir = scene_dir / "routes"
        log_dir = scene_dir / "search_logs"
        for seed in seeds:
            need_prior = args.algorithm in {"siaco", "sagppo", "all"}
            prior: list[tuple[int, int]] | None = None
            if need_prior:
                started = time.perf_counter()
                prior, history = siaco(scene, ants=args.ants, iterations=args.iterations, seed=seed)
                elapsed = time.perf_counter() - started
                _save_route(route_dir / f"siaco_seed{seed}.csv", prior)
                _write_dicts(log_dir / f"siaco_seed{seed}.csv", history)
                summaries.append(_record(scene, "siaco", seed, prior, elapsed=elapsed))

            if args.algorithm in {"sagppo", "all"}:
                assert prior is not None
                started = time.perf_counter()
                route, training_log, agent = train_sag_ppo(
                    scene, prior, episodes=args.episodes, seed=seed, device=args.device,
                    window=args.window, max_decision_steps=args.max_decision_steps,
                )
                elapsed = time.perf_counter() - started
                _save_training_outputs(scene_dir, "sagppo", seed, route, training_log, agent)
                summaries.append(_record(scene, "sagppo", seed, route, elapsed=elapsed))

    summary_path = output_root / "summary.csv"
    _write_dicts(summary_path, summaries)
    run_info = {
        "algorithm": args.algorithm,
        "scenes": dates,
        "seeds": seeds,
        "episodes": args.episodes if args.algorithm in {"sagppo", "all"} else None,
        "siaco_ants": args.ants,
        "siaco_iterations": args.iterations,
        "max_decision_steps": args.max_decision_steps,
        "sagppo_candidate_window": args.window,
        "device": args.device,
        "summary_csv": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
    }
    (output_root / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate polar route-planning algorithms.")
    parser.add_argument("--algorithm", choices=("siaco", "sagppo", "all"), default="all")
    parser.add_argument("--scene", default="2025-08-05", help="One fixed scene date or 'all'.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIG.seeds),
                        help="Five random seeds by default: 0 1 2 3 4.")
    parser.add_argument("--episodes", type=int, default=CONFIG.max_episodes,
                        help="Maximum training episodes per scene and seed.")
    parser.add_argument("--ants", type=int, default=CONFIG.ants, help="Ants per S-IACO iteration.")
    parser.add_argument("--iterations", type=int, default=CONFIG.siaco_iterations,
                        help="S-IACO search iterations.")
    parser.add_argument("--window", type=int, default=int(ALGORITHM_SETTINGS["candidate_horizon_cells"]),
                        help="Local candidate-search window in grid cells.")
    parser.add_argument("--max-decision-steps", type=int, default=CONFIG.max_decision_steps,
                        help="Maximum local decision steps per episode.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda.")
    parser.add_argument("--out", default="results/runs", help="Run output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"Results written to {summary}")


if __name__ == "__main__":
    main()
