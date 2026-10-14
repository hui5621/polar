from __future__ import annotations

import math
from typing import Any

import numpy as np

from .scene import Scene


def route_metrics(scene: Scene, route: list[tuple[int, int]]) -> dict[str, Any]:
    if not route:
        return {"success": False, "steps": 0}
    cells = np.asarray(route, dtype=np.int32)
    lengths = np.asarray(
        [math.hypot(b[0] - a[0], b[1] - a[1]) * scene.cell_km for a, b in zip(route, route[1:])],
        dtype=np.float64,
    )
    assists = np.asarray([scene.current_assist(a, b) for a, b in zip(route, route[1:])], dtype=np.float64)
    total_cost = sum(scene.step_cost(a, b) for a, b in zip(route, route[1:]))
    turn_angles: list[float] = []
    for a, b, c in zip(route, route[1:], route[2:]):
        first = np.asarray([b[1] - a[1], -(b[0] - a[0])], dtype=np.float64)
        second = np.asarray([c[1] - b[1], -(c[0] - b[0])], dtype=np.float64)
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator:
            turn_angles.append(math.degrees(math.acos(float(np.clip(first @ second / denominator, -1.0, 1.0)))))

    ice = scene.ice_concentration[cells[:, 0], cells[:, 1]].astype(np.float64)
    thickness = scene.ice_thickness[cells[:, 0], cells[:, 1]].astype(np.float64)
    risk = scene.chart_risk[cells[:, 0], cells[:, 1]].astype(np.float64)
    return {
        "success": bool(tuple(route[-1]) == scene.goal and all(not scene.blocked[p] for p in route)),
        "steps": int(max(len(route) - 1, 0)),
        "path_length_km": float(lengths.sum()),
        "total_cost": float(total_cost),
        "mean_ice_concentration": float(ice.mean()),
        "mean_ice_thickness_m": float(thickness.mean()),
        "mean_chart_risk": float(risk.mean()),
        "current_assist_m_s": float(assists.mean()) if len(assists) else 0.0,
        "ice_exposure_km": float(np.dot(lengths, 0.5 * (ice[:-1] + ice[1:]))) if len(lengths) else 0.0,
        "turn_complexity_deg": float(sum(turn_angles)),
        "mean_turn_angle_deg": float(np.mean(turn_angles)) if turn_angles else 0.0,
        "mean_ais_corridor_score": float(scene.ais_corridor[cells[:, 0], cells[:, 1]].mean()),
    }
