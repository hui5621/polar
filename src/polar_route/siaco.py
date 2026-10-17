from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import CONFIG, ALGORITHM_SETTINGS
from .geometry import turn_angles_deg
from .metrics import route_metrics
from .scene import NEIGHBOR_STEPS, Scene, list_scenes, load_scene
from .search import astar, line_cells, line_is_free


@dataclass(frozen=True)
class SceneProfile:
    descriptor: tuple[float, ...]
    alpha_explore: float
    beta_explore: float
    alpha_exploit: float
    beta_exploit: float
    evaporation_start: tuple[float, float]
    evaporation_end: tuple[float, float]
    exploration_share: float
    risk_ice: float
    risk_thickness: float
    risk_chart: float


def _scene_descriptor(scene: Scene) -> np.ndarray:
    water = ~scene.blocked
    if not np.any(water):
        water = np.ones(scene.shape, dtype=bool)
    ice = scene.ice_concentration[water].astype(np.float64)
    thick = scene.ice_thickness[water].astype(np.float64)
    risk = scene.chart_risk[water].astype(np.float64)
    current = scene.current_speed[water].astype(np.float64)
    cost = ALGORITHM_SETTINGS["route_cost_weights"]
    soft = np.clip(
        cost["ice"] * scene.ice_concentration
        + cost["thickness"] * np.clip(scene.ice_thickness / 2.0, 0.0, 1.0)
        + cost["risk"] * scene.chart_risk,
        0.0, 1.0,
    )[water]
    route_scale = math.dist(scene.start, scene.goal) / max(math.hypot(scene.rows, scene.cols), 1.0)
    values = [
        np.mean(ice), np.std(ice), np.quantile(ice, 0.75),
        np.mean(thick), np.std(thick), np.quantile(thick, 0.75),
        np.mean(current), np.std(current), np.mean(risk), np.std(risk),
        np.mean(scene.blocked), np.mean(soft > float(ALGORITHM_SETTINGS["soft_risk_ratio_threshold"])), route_scale,
    ]
    return np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=1)
def _descriptor_bounds() -> tuple[np.ndarray, np.ndarray]:
    samples = np.stack([_scene_descriptor(load_scene(date)) for date in list_scenes()])
    return samples.min(axis=0), samples.max(axis=0)


def scene_profile(scene: Scene) -> SceneProfile:
    """Build a deterministic profile from thirteen scene descriptors."""
    raw = _scene_descriptor(scene)
    low, high = _descriptor_bounds()
    d = np.divide(raw - low, np.maximum(high - low, 1e-9)).clip(0.0, 1.0)
    # Piecewise-affine map: high ice/risk increases exploitation and local
    # risk evaporation; current variability increases exploration diversity.
    ice_level = float(np.clip(0.55 * d[0] + 0.25 * d[2] + 0.20 * d[10], 0.0, 1.0))
    risk_level = float(np.clip(0.40 * d[8] + 0.25 * d[9] + 0.20 * d[10] + 0.15 * d[11], 0.0, 1.0))
    current_level = float(np.clip(0.5 * d[6] + 0.5 * d[7], 0.0, 1.0))
    scale_level = float(np.clip(d[12], 0.0, 1.0))
    return SceneProfile(
        descriptor=tuple(float(x) for x in d),
        alpha_explore=0.65 + 0.35 * (1.0 - risk_level),
        beta_explore=1.8 + 0.7 * scale_level,
        alpha_exploit=1.1 + 0.5 * ice_level,
        beta_exploit=2.5 + 1.0 * (0.5 * ice_level + 0.5 * risk_level),
        evaporation_start=(0.10 + 0.06 * risk_level, 0.18 + 0.06 * risk_level),
        evaporation_end=(0.06 + 0.04 * risk_level, 0.10 + 0.04 * risk_level),
        exploration_share=float(np.clip(0.55 - 0.15 * risk_level + 0.10 * current_level, 0.35, 0.65)),
        risk_ice=0.10 + 0.10 * d[0],
        risk_thickness=0.07 + 0.08 * d[3],
        risk_chart=0.05 + 0.05 * d[8],
    )


@lru_cache(maxsize=12)
def _distance_to_goal(date: str) -> np.ndarray:
    """Hard-safe grid distance used as the global-distance guidance field."""
    scene = load_scene(date)
    distances = np.full(scene.shape, np.inf, dtype=np.float64)
    distances[scene.goal] = 0.0
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, scene.goal)]
    weights = ALGORITHM_SETTINGS["ant_guidance_weights"]
    while queue:
        distance, point = heapq.heappop(queue)
        if distance > distances[point] + 1e-12:
            continue
        for nxt in scene.neighbors(point):
            # The search is reversed from the goal, but edge cost follows the
            # eventual sailing direction from predecessor ``nxt`` to ``point``.
            edge_cost = scene.step_cost(
                nxt, point, ice_weight=float(weights["ice"]),
                thickness_weight=float(weights["thickness"]),
                chart_weight=float(weights["chart"]),
                current_weight=float(weights["current"]), ais_weight=0.0,
            )
            candidate = distance + edge_cost
            if candidate + 1e-12 >= distances[nxt]:
                continue
            distances[nxt] = candidate
            heapq.heappush(queue, (candidate, nxt))
    return distances


def _soft_risk(scene: Scene) -> np.ndarray:
    cost = ALGORITHM_SETTINGS["route_cost_weights"]
    return np.clip(
        cost["ice"] * scene.ice_concentration
        + cost["thickness"] * np.clip(scene.ice_thickness / 2.0, 0.0, 1.0)
        + cost["risk"] * scene.chart_risk,
        0.0, 1.0,
    )


def _soft_budget(scene: Scene, route_length_km: float) -> float:
    soft = _soft_risk(scene)
    water = ~scene.blocked
    soft_ratio = float(np.mean(soft[water] > float(ALGORITHM_SETTINGS["soft_risk_ratio_threshold"]))) if np.any(water) else 0.0
    per_km = float(ALGORITHM_SETTINGS["soft_budget_excess_per_km"]) * (1.0 + soft_ratio)
    return per_km * max(float(route_length_km), 1.0)


def _route_soft_excess(scene: Scene, route: list[tuple[int, int]], threshold: float) -> float:
    soft = _soft_risk(scene)
    total = 0.0
    for a, b in zip(route[:-1], route[1:]):
        distance = math.dist(a, b) * scene.cell_km
        steps = max(1, int(math.ceil(math.dist(a, b) / float(ALGORITHM_SETTINGS["route_sample_step_cells"]))))
        for index in range(1, steps + 1):
            t = index / steps
            r = int(np.clip(math.floor(a[0] + t * (b[0] - a[0]) + 0.5), 0, scene.rows - 1))
            c = int(np.clip(math.floor(a[1] + t * (b[1] - a[1]) + 0.5), 0, scene.cols - 1))
            total += max(float(soft[r, c]) - threshold, 0.0) * distance / steps
    return total


@lru_cache(maxsize=4096)
def _cached_metrics(date: str, route_key: tuple[tuple[int, int], ...]) -> dict[str, float | bool | None]:
    return route_metrics(load_scene(date), list(route_key))


def _metrics(scene: Scene, route: list[tuple[int, int]]) -> dict[str, float | bool | None]:
    return _cached_metrics(scene.date, tuple(route))


def _hard_safe(scene: Scene, route: list[tuple[int, int]]) -> bool:
    return bool(route and all(scene.in_bounds(p) and not scene.blocked[p] for p in route)
                and all(line_is_free(scene, a, b) for a, b in zip(route[:-1], route[1:])))


def local_repair(scene: Scene, route: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Repair excess soft-risk locally while retaining each window's anchors."""
    if len(route) < 3:
        return list(route)
    risk = _soft_risk(scene)
    threshold = float(np.quantile(risk[~scene.blocked], float(ALGORITHM_SETTINGS["soft_risk_threshold_quantile"])))
    lower, upper = ALGORITHM_SETTINGS["repair_radius_cells"]
    budget = _soft_budget(scene, float(_metrics(scene, route)["path_length_km"]))
    repaired = list(route)
    if _route_soft_excess(scene, repaired, threshold) <= budget:
        return repaired

    # Each iteration addresses the highest-excess local edge, searches only
    # inside its bounded window, and keeps the original entry/exit anchors.
    attempted: set[int] = set()
    max_windows = int(ALGORITHM_SETTINGS["repair_max_windows_per_route"])
    for _ in range(max_windows):
        edge_excess = [
            _route_soft_excess(scene, [a, b], threshold)
            for a, b in zip(repaired[:-1], repaired[1:])
        ]
        eligible = [i for i, excess in enumerate(edge_excess) if i not in attempted and excess > 1e-9]
        if not eligible:
            break
        edge_index = max(eligible, key=edge_excess.__getitem__)
        attempted.add(edge_index)
        point = repaired[edge_index]
        local_window = risk[max(0, point[0] - upper):min(scene.rows, point[0] + upper + 1),
                            max(0, point[1] - upper):min(scene.cols, point[1] + upper + 1)]
        local_mean = float(np.mean(local_window)) if local_window.size else 0.0
        radius = int(np.clip(round(lower + (upper - lower) * local_mean), lower, upper))
        entry_index = max(0, edge_index - radius // 2)
        exit_index = min(len(repaired) - 1, entry_index + radius)
        if exit_index <= entry_index + 1:
            continue
        entry, exit_point = repaired[entry_index], repaired[exit_index]
        old_segment = repaired[entry_index:exit_index + 1]
        old_excess = _route_soft_excess(scene, old_segment, threshold)
        bounds = (
            max(0, min(entry[0], exit_point[0]) - radius),
            min(scene.rows - 1, max(entry[0], exit_point[0]) + radius),
            max(0, min(entry[1], exit_point[1]) - radius),
            min(scene.cols - 1, max(entry[1], exit_point[1]) + radius),
        )
        weights = ALGORITHM_SETTINGS["route_cost_weights"]
        replacement = astar(
            scene, entry, exit_point, bounds=bounds,
            ice_weight=float(weights["ice"]),
            thickness_weight=float(weights["thickness"]),
            chart_weight=float(weights["risk"]),
            current_weight=0.0, ais_weight=0.0,
        )
        if replacement is None or not _hard_safe(scene, replacement):
            continue
        new_excess = _route_soft_excess(scene, replacement, threshold)
        if new_excess + 1e-9 < old_excess:
            repaired = repaired[:entry_index] + replacement + repaired[exit_index + 1:]
            attempted.clear()
    return repaired


def _objectives(scene: Scene, route: list[tuple[int, int]]) -> np.ndarray:
    values = _metrics(scene, route)
    return np.asarray([
        values["path_length_km"], values["total_cost"], values["mean_ice_concentration"],
        values["mean_ice_thickness_m"], -values["current_assist_m_s"],
        values["turn_complexity_deg"],
        float(values["travel_time_h"] if values["travel_time_h"] is not None else 1e12),
        values["ice_exposure_km"],
    ], dtype=np.float64)


def _epsilon(scene: Scene) -> np.ndarray:
    scales = np.asarray([
        math.dist(scene.start, scene.goal) * scene.cell_km,
        math.dist(scene.start, scene.goal) * scene.cell_km,
        1.0, 2.0, 1.0, 360.0, 240.0,
        math.dist(scene.start, scene.goal) * scene.cell_km,
    ])
    return scales * float(ALGORITHM_SETTINGS["pareto_tolerance_normalized"])


def _dominates(a: np.ndarray, b: np.ndarray, eps: np.ndarray) -> bool:
    return bool(np.all(a <= b + eps) and np.any(a < b - eps))


def _archive_add(scene: Scene, archive: list[list[tuple[int, int]]], candidate: list[tuple[int, int]], limit: int) -> list[list[tuple[int, int]]]:
    if not candidate or tuple(candidate) in {tuple(p) for p in archive}:
        return archive
    if not _hard_safe(scene, candidate):
        return archive
    if not _metrics(scene, candidate)["dynamic_feasible"]:
        return archive
    soft = _soft_risk(scene)
    threshold = float(np.quantile(soft[~scene.blocked], float(ALGORITHM_SETTINGS["soft_risk_threshold_quantile"])))
    budget = _soft_budget(scene, float(_objectives(scene, candidate)[0]))
    if _route_soft_excess(scene, candidate, threshold) > budget:
        return archive
    eps = _epsilon(scene)
    candidate_obj = _objectives(scene, candidate)
    survivors = [p for p in archive if not _dominates(candidate_obj, _objectives(scene, p), eps)]
    if any(_dominates(_objectives(scene, p), candidate_obj, eps) for p in survivors):
        return archive
    survivors.append(candidate)
    if len(survivors) > limit:
        matrix = np.stack([_objectives(scene, p) for p in survivors])
        scale = np.maximum(np.ptp(matrix, axis=0), np.maximum(np.abs(matrix).mean(axis=0) * 0.02, 1e-8))
        normalized = (matrix - matrix.min(axis=0)) / scale
        keep = np.argsort(normalized.mean(axis=1), kind="stable")[:limit]
        survivors = [survivors[int(i)] for i in keep]
    return survivors


def _select_archive_route(scene: Scene, archive: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    matrix = np.stack([_objectives(scene, path) for path in archive])
    low = matrix.min(axis=0)
    span = np.maximum(matrix.max(axis=0) - low, 1e-8)
    normalized = (matrix - low) / span
    profile = scene_profile(scene)
    weights = np.asarray([1.0, 1.0, 1.0 + profile.descriptor[0], 1.0 + profile.descriptor[3],
                          1.0 + profile.descriptor[6], 0.8, 1.0, 1.0 + profile.descriptor[0]])
    return archive[int(np.argmin(np.average(normalized, axis=1, weights=weights)))]


def _construct_ant(scene: Scene, pheromone: np.ndarray, profile: SceneProfile,
    rng: np.random.Generator, *, exploit: bool) -> list[tuple[int, int]] | None:
    route = [scene.start]
    on_path = {scene.start}
    point = scene.start
    distance_field = _distance_to_goal(scene.date)
    goal_vector = np.asarray([scene.goal[1] - scene.start[1], scene.start[0] - scene.goal[0]], dtype=np.float64)
    goal_norm = max(float(np.linalg.norm(goal_vector)), 1e-12)
    goal_cost = max(float(distance_field[scene.start]), 1e-12)
    weights = ALGORITHM_SETTINGS["ant_guidance_weights"]
    alpha = profile.alpha_exploit if exploit else profile.alpha_explore
    beta = profile.beta_exploit if exploit else profile.beta_explore
    max_steps = int(ALGORITHM_SETTINGS["ant_max_expansions"])
    for _ in range(max_steps):
        if point == scene.goal:
            return route
        current_distance = float(distance_field[point])
        slack_ratio = float(ALGORITHM_SETTINGS["ant_global_distance_slack_ratio"])
        choices: list[tuple[int, int]] = []
        edge_costs: dict[tuple[int, int], float] = {}
        for nxt in scene.neighbors(point):
            if nxt in on_path:
                continue
            edge_cost = scene.step_cost(
                point, nxt, ice_weight=float(weights["ice"]),
                thickness_weight=float(weights["thickness"]),
                chart_weight=float(weights["chart"]),
                current_weight=float(weights["current"]), ais_weight=0.0,
            )
            remaining = float(distance_field[nxt])
            if (remaining < current_distance - 1e-9
                    and remaining + edge_cost <= current_distance + slack_ratio * edge_cost):
                choices.append(nxt)
                edge_costs[nxt] = edge_cost
        if not choices:
            return None
        logits = []
        for nxt in choices:
            dr, dc = nxt[0] - point[0], nxt[1] - point[1]
            move = np.asarray([dc, -dr], dtype=np.float64)
            move /= max(float(np.linalg.norm(move)), 1e-12)
            toward = np.asarray([scene.goal[1] - nxt[1], nxt[0] - scene.goal[0]], dtype=np.float64)
            progress = max(0.02, float(move @ toward) / max(float(np.linalg.norm(toward)), 1e-12))
            edge_cost = edge_costs[nxt]
            d = NEIGHBOR_STEPS.index((dr, dc))
            tau = max(float(pheromone[point[0], point[1], d]), 1e-8)
            remaining = float(distance_field[nxt])
            global_distance = goal_cost / max(goal_cost + remaining, 1e-8)
            directional = 0.5 + 0.5 * progress
            heuristic = directional * global_distance / max(edge_cost, 1e-8)
            logits.append(alpha * math.log(tau) + beta * math.log(max(heuristic, 1e-8)))
        logits = np.asarray(logits, dtype=np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        point = choices[int(rng.choice(len(choices), p=probs))]
        route.append(point)
        on_path.add(point)
    return route if route[-1] == scene.goal else None


def _edge_environment(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ice = np.zeros((*scene.shape, len(NEIGHBOR_STEPS)), dtype=np.float32)
    thickness = np.zeros_like(ice)
    chart = np.zeros_like(ice)
    for index, (dr, dc) in enumerate(NEIGHBOR_STEPS):
        r0, r1 = max(0, -dr), min(scene.rows, scene.rows - dr)
        c0, c1 = max(0, -dc), min(scene.cols, scene.cols - dc)
        first = (slice(r0, r1), slice(c0, c1))
        second = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        ice[first + (index,)] = 0.5 * (scene.ice_concentration[first] + scene.ice_concentration[second])
        thickness[first + (index,)] = np.clip(
            0.25 * (scene.ice_thickness[first] + scene.ice_thickness[second]), 0.0, 1.0
        )
        chart[first + (index,)] = 0.5 * (scene.chart_risk[first] + scene.chart_risk[second])
    return ice, thickness, chart


def _update_pheromones(scene: Scene, tau_e: np.ndarray, tau_x: np.ndarray,
                       routes_e: list[list[tuple[int, int]]], routes_x: list[list[tuple[int, int]]],
                       profile: SceneProfile, iteration: int, iterations: int,
                       edge_fields: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    updated = []
    for branch, tau in enumerate((tau_e, tau_x)):
        start = profile.evaporation_start[branch]
        end = profile.evaporation_end[branch]
        scheduled = start + (iteration - 1) / max(iterations - 1, 1) * (end - start)
        edge_ice, edge_thickness, edge_chart = edge_fields
        local = (
            profile.risk_ice * edge_ice
            + profile.risk_thickness * edge_thickness
            + profile.risk_chart * edge_chart
        )
        rho_min, rho_max = ALGORITHM_SETTINGS["evaporation_rate_bounds"]
        rho = np.clip(scheduled + local, float(rho_min), float(rho_max))
        new = tau * (1.0 - rho)
        routes = routes_e if branch == 0 else routes_x
        for route in routes:
            length = max(float(_metrics(scene, route)["path_length_km"]), 1e-8)
            deposit = float(ALGORITHM_SETTINGS["pheromone_deposit_constant"]) / length
            for a, b in zip(route[:-1], route[1:]):
                dr, dc = b[0] - a[0], b[1] - a[1]
                d = NEIGHBOR_STEPS.index((dr, dc))
                new[a[0], a[1], d] += deposit
        updated.append(np.clip(new, 1e-4, 25.0))
    # Exchange spatial information after independent branch updates.
    e, x = updated
    mix_e, mix_x = ALGORITHM_SETTINGS["pheromone_exchange"]
    x_to_e = x / max(float(x.sum()), 1e-12) * float(e.sum())
    e_to_x = e / max(float(e.sum()), 1e-12) * float(x.sum())
    tau_e[:] = np.clip((1.0 - mix_e) * e + mix_e * x_to_e, 1e-4, 25.0)
    tau_x[:] = np.clip((1.0 - mix_x) * x + mix_x * e_to_x, 1e-4, 25.0)


def siaco(scene: Scene, *, ants: int = CONFIG.ants, iterations: int = CONFIG.siaco_iterations,
          seed: int = 0, archive_limit: int = int(ALGORITHM_SETTINGS["pareto_archive_limit"])) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Run scene-adaptive dual-pheromone search with local repair and Pareto archiving."""
    rng = np.random.default_rng(seed)
    profile = scene_profile(scene)
    tau_e = np.ones((*scene.shape, len(NEIGHBOR_STEPS)), dtype=np.float32)
    tau_x = np.ones_like(tau_e)
    edge_fields = _edge_environment(scene)
    archive: list[list[tuple[int, int]]] = []
    history: list[dict[str, float]] = []
    total_ants = max(1, int(ants))
    for iteration in range(1, max(1, int(iterations)) + 1):
        routes_e: list[list[tuple[int, int]]] = []
        routes_x: list[list[tuple[int, int]]] = []
        feasible_count = 0
        n_explore = int(round(total_ants * profile.exploration_share))
        for ant in range(total_ants):
            exploit = ant >= n_explore
            tau = tau_x if exploit else tau_e
            route = _construct_ant(scene, tau, profile, rng, exploit=exploit)
            if route is None or route[-1] != scene.goal:
                continue
            route = local_repair(scene, route)
            if not _hard_safe(scene, route) or not _metrics(scene, route)["dynamic_feasible"]:
                continue
            target = routes_x if exploit else routes_e
            target.append(route)
            feasible_count += 1
            archive = _archive_add(scene, archive, route, archive_limit)
        _update_pheromones(scene, tau_e, tau_x, routes_e, routes_x, profile, iteration, max(iterations, 1), edge_fields)
        if archive:
            representative = _select_archive_route(scene, archive)
            route_values = _metrics(scene, representative)
        else:
            route_values = {"total_cost": float("inf"), "path_length_km": float("inf")}
        history.append({
            "iteration": float(iteration), "archive_size": float(len(archive)),
            "feasible_ants": float(feasible_count),
            "best_total_cost": float(route_values["total_cost"]),
            "best_path_length_km": float(route_values["path_length_km"]),
        })
    if not archive:
        raise RuntimeError(f"S-IACO found no feasible route for {scene.date}")
    return _select_archive_route(scene, archive), history
