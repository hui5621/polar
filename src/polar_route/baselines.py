from __future__ import annotations

from .scene import Scene
from .search import astar, smooth_line_of_sight
from .siaco import ant_colony_route, siaco


BASELINE_NAMES = ("astar", "current_astar", "los_astar", "aco", "zhou_iaco")


def run_baseline(
    scene: Scene,
    name: str,
    *,
    seed: int = 0,
    ants: int = 30,
    iterations: int = 80,
) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Run one graph-search or ant-colony reference planner."""
    if name == "astar":
        route = astar(scene, current_weight=0.0, ais_weight=0.0)
        return _require(route, name, scene), []
    if name == "current_astar":
        route = astar(scene, current_weight=1.0, ais_weight=0.0)
        return _require(route, name, scene), []
    if name == "los_astar":
        route = astar(scene, current_weight=0.0, ais_weight=0.0)
        return smooth_line_of_sight(scene, _require(route, name, scene)), []
    if name == "aco":
        return ant_colony_route(scene, mode="aco", ants=ants, iterations=iterations, seed=seed)
    if name == "zhou_iaco":
        return ant_colony_route(scene, mode="zhou_iaco", ants=ants, iterations=iterations, seed=seed)
    if name == "siaco":
        return siaco(scene, ants=ants, iterations=iterations, seed=seed)
    raise ValueError(f"Unknown baseline {name!r}; choose from {', '.join((*BASELINE_NAMES, 'siaco'))}")


def _require(route: list[tuple[int, int]] | None, name: str, scene: Scene) -> list[tuple[int, int]]:
    if route is None:
        raise RuntimeError(f"{name} did not find a connected route for {scene.date}")
    return route
