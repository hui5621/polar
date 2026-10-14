from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .metrics import route_metrics
from .scene import NEIGHBOR_STEPS, Scene
from .search import astar, line_is_free, line_cells


@dataclass(frozen=True)
class SearchProfile:
    alpha: float
    beta: float
    evaporation: float
    exploration_share: float
    ice_weight: float
    chart_weight: float


def scene_profile(scene: Scene) -> SearchProfile:
    """Derive one continuous parameter profile from the fixed scene raster."""
    water = ~scene.blocked
    mean_ice = float(scene.ice_concentration[water].mean()) if np.any(water) else 0.0
    mean_risk = float(scene.chart_risk[water].mean()) if np.any(water) else 0.0
    return SearchProfile(
        alpha=1.0 + 0.4 * mean_ice,
        beta=2.5 + 0.5 * (1.0 - mean_ice),
        evaporation=float(np.clip(0.25 + 0.2 * mean_risk, 0.2, 0.5)),
        exploration_share=float(np.clip(0.55 - 0.2 * mean_ice, 0.35, 0.65)),
        ice_weight=2.0 + 3.0 * mean_ice,
        chart_weight=2.0 + 2.0 * mean_risk,
    )


def _turn_complexity(route: list[tuple[int, int]]) -> float:
    total = 0.0
    for a, b, c in zip(route, route[1:], route[2:]):
        first = np.asarray([b[1] - a[1], -(b[0] - a[0])], dtype=np.float64)
        second = np.asarray([c[1] - b[1], -(c[0] - b[0])], dtype=np.float64)
        den = float(np.linalg.norm(first) * np.linalg.norm(second))
        if den:
            total += math.degrees(math.acos(float(np.clip(first @ second / den, -1.0, 1.0))))
    return total


def local_repair(
    scene: Scene,
    route: list[tuple[int, int]],
    *,
    max_gap: int = 18,
) -> list[tuple[int, int]]:
    """Replace a short route segment when a safe, lower-cost connection exists."""
    if len(route) < 3:
        return list(route)
    repaired = [route[0]]
    cursor = 0
    while cursor < len(route) - 1:
        best = cursor + 1
        upper = min(len(route) - 1, cursor + max(2, int(max_gap)))
        old_cost = 0.0
        for end in range(cursor + 1, upper + 1):
            old_cost += scene.step_cost(route[end - 1], route[end])
            if end < cursor + 2 or not line_is_free(scene, route[cursor], route[end]):
                continue
            straight = line_cells(route[cursor], route[end])
            new_cost = sum(scene.step_cost(a, b) for a, b in zip(straight, straight[1:]))
            if new_cost < old_cost:
                best = end
        repaired.extend(line_cells(route[cursor], route[best])[1:])
        cursor = best
    return repaired


def _objectives(scene: Scene, route: list[tuple[int, int]]) -> np.ndarray:
    values = route_metrics(scene, route)
    return np.asarray([
        values["path_length_km"],
        values["total_cost"],
        values["mean_ice_concentration"],
        values["mean_ice_thickness_m"],
        values["mean_chart_risk"],
        -values["current_assist_m_s"],
        values["turn_complexity_deg"],
    ], dtype=np.float64)


def _dominates(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first <= second) and np.any(first < second))


def _archive_add(
    scene: Scene,
    archive: list[list[tuple[int, int]]],
    candidate: list[tuple[int, int]],
    *,
    limit: int = 40,
) -> list[list[tuple[int, int]]]:
    if not candidate or candidate[-1] != scene.goal:
        return archive
    signature = tuple(candidate)
    if any(tuple(path) == signature for path in archive):
        return archive
    candidate_obj = _objectives(scene, candidate)
    survivors = [path for path in archive if not _dominates(candidate_obj, _objectives(scene, path))]
    if any(_dominates(_objectives(scene, path), candidate_obj) for path in survivors):
        return archive
    survivors.append(candidate)
    if len(survivors) > limit:
        matrix = np.stack([_objectives(scene, path) for path in survivors])
        low = matrix.min(axis=0)
        span = np.maximum(matrix.max(axis=0) - low, 1e-8)
        rank = np.mean((matrix - low) / span, axis=1)
        keep = np.argsort(rank)[:limit]
        survivors = [survivors[int(i)] for i in keep]
    return survivors


def _select_archive_route(scene: Scene, archive: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    if not archive:
        raise RuntimeError(f"S-IACO found no feasible route for {scene.date}")
    matrix = np.stack([_objectives(scene, path) for path in archive])
    low = matrix.min(axis=0)
    span = np.maximum(matrix.max(axis=0) - low, 1e-8)
    score = np.mean((matrix - low) / span, axis=1)
    return archive[int(np.argmin(score))]


def _construct_ant(
    scene: Scene,
    pheromone: np.ndarray,
    profile: SearchProfile,
    rng: np.random.Generator,
    *,
    improved_direction: bool,
) -> list[tuple[int, int]] | None:
    route = [scene.start]
    visited = {scene.start}
    point = scene.start
    goal_distance = math.dist(scene.start, scene.goal)
    max_steps = min(scene.rows * scene.cols, max(4 * (scene.rows + scene.cols), int(3.0 * goal_distance)))
    for _ in range(max_steps):
        if point == scene.goal:
            return route
        choices = [nxt for nxt in scene.neighbors(point) if nxt not in visited]
        if not choices:
            return None
        log_weights = []
        for nxt in choices:
            dr, dc = nxt[0] - point[0], nxt[1] - point[1]
            length = math.hypot(dr, dc)
            current_weight = 0.35 if improved_direction else 0.0
            edge_cost = scene.step_cost(
                point, nxt,
                ice_weight=profile.ice_weight,
                chart_weight=profile.chart_weight,
                current_weight=current_weight,
            )
            eta = 1.0 / max(edge_cost, 1e-6)
            remain = math.dist(nxt, scene.goal)
            current_distance = math.dist(point, scene.goal)
            progress = max(0.05, (current_distance - remain) / max(length, 1e-6))
            # Basic ACO gets a mild goal-distance heuristic; Zhou-IACO and
            # S-IACO additionally emphasize directional progress.
            eta *= progress ** 2 if improved_direction else (0.25 + 0.75 * progress)
            cell_tau = max(float(pheromone[nxt]), 1e-8)
            log_weights.append(profile.alpha * math.log(cell_tau) + profile.beta * math.log(max(eta, 1e-8)))
        log_weights_arr = np.asarray(log_weights, dtype=np.float64)
        log_weights_arr -= log_weights_arr.max()
        probs = np.exp(log_weights_arr)
        probs /= probs.sum()
        point = choices[int(rng.choice(len(choices), p=probs))]
        route.append(point)
        visited.add(point)
    return route if route[-1] == scene.goal else None


def ant_colony_route(
    scene: Scene,
    *,
    mode: str = "aco",
    ants: int = 30,
    iterations: int = 80,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Classical ACO and direction-guided Zhou-IACO reference implementations."""
    rng = np.random.default_rng(seed)
    profile = scene_profile(scene)
    if mode == "aco":
        profile = SearchProfile(1.0, 2.0, 0.35, 1.0, 0.0, 0.0)
        improved = False
    elif mode == "zhou_iaco":
        improved = True
    else:
        raise ValueError(f"Unsupported ant-colony mode: {mode}")

    pheromone = np.ones(scene.shape, dtype=np.float32)
    best: list[tuple[int, int]] | None = None
    history: list[dict[str, float]] = []
    for iteration in range(1, max(1, int(iterations)) + 1):
        routes: list[list[tuple[int, int]]] = []
        scores: list[float] = []
        for _ in range(max(1, int(ants))):
            route = _construct_ant(scene, pheromone, profile, rng, improved_direction=improved)
            if route is None or route[-1] != scene.goal:
                continue
            metric = route_metrics(scene, route)
            routes.append(route)
            scores.append(float(metric["total_cost"]))
        pheromone *= 1.0 - profile.evaporation
        if routes:
            for route, score in zip(routes, scores):
                deposit = 1.0 / max(score, 1e-6)
                for point in route:
                    pheromone[point] += deposit
            local_best = routes[int(np.argmin(scores))]
            if best is None or route_metrics(scene, local_best)["total_cost"] < route_metrics(scene, best)["total_cost"]:
                best = local_best
            if mode == "zhou_iaco":
                rho = profile.evaporation * (0.75 + 0.5 * iteration / max(iterations, 1))
                pheromone *= (1.0 - rho) / max(1.0 - profile.evaporation, 1e-6)
        history.append({
            "iteration": float(iteration),
            "best_total_cost": float(route_metrics(scene, best)["total_cost"]) if best else float("inf"),
            "feasible_ants": float(len(routes)),
        })
    if best is None:
        raise RuntimeError(f"No feasible {mode} route for {scene.date}")
    return best, history


def siaco(
    scene: Scene,
    *,
    ants: int = 30,
    iterations: int = 80,
    seed: int = 0,
    archive_limit: int = 40,
) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Scene-adaptive IACO with dual pheromone fields and Pareto route selection."""
    rng = np.random.default_rng(seed)
    profile = scene_profile(scene)
    exploration = np.ones(scene.shape, dtype=np.float32)
    exploitation = np.ones(scene.shape, dtype=np.float32)
    archive: list[list[tuple[int, int]]] = []
    history: list[dict[str, float]] = []
    incumbent: list[tuple[int, int]] | None = None

    for iteration in range(1, max(1, int(iterations)) + 1):
        feasible: list[list[tuple[int, int]]] = []
        for _ in range(max(1, int(ants))):
            use_explore = rng.random() < profile.exploration_share
            field = exploration if use_explore else exploitation
            route = _construct_ant(scene, field, profile, rng, improved_direction=True)
            if route is None or route[-1] != scene.goal:
                continue
            route = local_repair(scene, route)
            feasible.append(route)
            archive = _archive_add(scene, archive, route, limit=archive_limit)

        exploration *= 1.0 - profile.evaporation
        exploitation *= 1.0 - profile.evaporation
        for route in feasible:
            objective = _objectives(scene, route)
            deposit = 1.0 / max(float(objective[1]), 1e-6)
            for point in route:
                exploration[point] += deposit
        if archive:
            chosen = _select_archive_route(scene, archive)
            deposit = 2.0 / max(float(_objectives(scene, chosen)[1]), 1e-6)
            for point in chosen:
                exploitation[point] += deposit
            incumbent = chosen
        history.append({
            "iteration": float(iteration),
            "archive_size": float(len(archive)),
            "feasible_ants": float(len(feasible)),
            "best_total_cost": float(route_metrics(scene, incumbent)["total_cost"]) if incumbent else float("inf"),
        })

    if not archive and incumbent is None:
        raise RuntimeError(f"S-IACO found no connected route for {scene.date}")
    return _select_archive_route(scene, archive) if archive else incumbent, history
