from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import UNSPECIFIED
from .geometry import segment_samples, turn_angles_deg
from .scene import Scene
from .vessel import evaluate_route_vessel


def route_metrics(scene: Scene, route: list[tuple[int, int]]) -> dict[str, Any]:
    """Compute the nine paper metrics from segment samples and vessel SOG."""
    if len(route) < 2:
        return {"success": False, "steps": max(len(route) - 1, 0), "path_length_km": 0.0,
                "total_cost": 0.0, "mean_ice_concentration": 0.0, "mean_ice_thickness_m": 0.0,
                "mean_chart_risk": 0.0, "current_assist_m_s": 0.0, "turn_complexity_deg": 0.0,
                "travel_time_h": None, "ice_exposure_km": 0.0, "fuel_tons": None,
                "dynamic_feasible": False}

    length = 0.0
    sums = {name: 0.0 for name in ("ice", "thick", "risk", "assist")}
    exposure_km = 0.0
    integrated_cost = 0.0
    cost_cfg = UNSPECIFIED["route_cost_weights"]
    blocked_samples = 0
    for a, b in zip(route[:-1], route[1:]):
        dr, dc = float(b[0] - a[0]), float(b[1] - a[1])
        grid_distance = math.hypot(dr, dc)
        distance_km = grid_distance * scene.cell_km
        if distance_km <= 1e-12:
            continue
        samples = segment_samples(a, b, rows=scene.rows, cols=scene.cols)
        if not samples:
            continue
        east, north = dc / grid_distance, -dr / grid_distance
        values = {name: [] for name in sums}
        exposed = 0
        for row, col, _ in samples:
            ice = float(scene.ice_concentration[row, col])
            thick = float(scene.ice_thickness[row, col])
            risk = float(scene.chart_risk[row, col])
            assist = east * float(scene.current_u[row, col]) + north * float(scene.current_v[row, col])
            values["ice"].append(ice)
            values["thick"].append(thick)
            values["risk"].append(risk)
            values["assist"].append(assist)
            exposed += int(ice > float(UNSPECIFIED["ice_exposure_threshold"]))
            blocked_samples += int(scene.blocked[row, col])
        means = {name: float(np.mean(v)) if v else 0.0 for name, v in values.items()}
        for name, value in means.items():
            sums[name] += distance_km * value
        exposure_km += distance_km * exposed / len(samples)

        # Equation (4): integrate the segment cost along physical route length.
        normalized_thick = float(np.clip(means["thick"] / 2.0, 0.0, 1.0))
        soft_risk = float(np.clip(
            cost_cfg["ice"] * means["ice"]
            + cost_cfg["thickness"] * normalized_thick
            + cost_cfg["risk"] * means["risk"], 0.0, 1.0
        ))
        multiplier = (
            1.0
            + cost_cfg["ice"] * means["ice"]
            + cost_cfg["thickness"] * normalized_thick
            + cost_cfg["risk"] * means["risk"]
            + cost_cfg["soft"] * soft_risk
            - cost_cfg["current"] * means["assist"]
        )
        integrated_cost += distance_km * multiplier
        length += distance_km

    denominator = max(length, 1e-12)
    means = {name: value / denominator for name, value in sums.items()}
    turns = turn_angles_deg(route)
    turn_threshold = float(UNSPECIFIED["turn_complexity_threshold_deg"])
    turn_complexity = float(sum(angle for angle in turns if angle >= turn_threshold))
    vessel = evaluate_route_vessel(scene, route)
    success = bool(
        tuple(route[0]) == scene.start
        and tuple(route[-1]) == scene.goal
        and blocked_samples == 0
        and vessel["dynamic_feasible"]
    )
    metrics: dict[str, Any] = {
        "success": success,
        "steps": len(route) - 1,
        "path_length_km": float(length),
        "total_cost": float(integrated_cost),
        "mean_ice_concentration": means["ice"],
        "mean_ice_thickness_m": means["thick"],
        "mean_chart_risk": means["risk"],
        "current_assist_m_s": means["assist"],
        "turn_complexity_deg": turn_complexity,
        "mean_turn_angle_deg": float(np.mean(turns)) if turns else 0.0,
        "travel_time_h": vessel["travel_time_h"],
        "ice_exposure_km": float(exposure_km),
        "fuel_tons": vessel["fuel_tons"],
        "dynamic_feasible": vessel["dynamic_feasible"],
        "mean_safe_speed_kmh": vessel["mean_safe_speed_kmh"],
        "mean_sog_speed_kmh": vessel["mean_sog_speed_kmh"],
        "min_sog_speed_kmh": vessel["min_sog_speed_kmh"],
        "mean_required_shaft_power_kw": vessel["mean_required_shaft_power_kw"],
        "propulsion_margin_min": vessel["propulsion_margin_min"],
        "ground_speed_infeasible_samples": vessel["ground_speed_infeasible_samples"],
        "icebreaker_infeasible_samples": vessel["propulsion_infeasible_samples"],
    }
    # Paper-table labels are included for convenient CSV export.
    metrics.update({
        "PathLength": metrics["path_length_km"], "TotalCost": metrics["total_cost"],
        "MeanIce": metrics["mean_ice_concentration"], "MeanThick": metrics["mean_ice_thickness_m"],
        "CurrentAssist": metrics["current_assist_m_s"], "TurnComplex": metrics["turn_complexity_deg"],
        "TravelTime": metrics["travel_time_h"], "IceExposureDistance": metrics["ice_exposure_km"],
        "FuelTons": metrics["fuel_tons"],
    })
    return metrics
