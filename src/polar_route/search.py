from __future__ import annotations

import heapq
import itertools
import math

from .config import UNSPECIFIED
from .scene import NEIGHBOR_STEPS, Scene


_COST_WEIGHTS = UNSPECIFIED["route_cost_weights"]


def astar(
    scene: Scene,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    *,
    ice_weight: float = float(_COST_WEIGHTS["ice"]),
    thickness_weight: float = float(_COST_WEIGHTS["thickness"]),
    chart_weight: float = float(_COST_WEIGHTS["risk"]),
    current_weight: float = float(_COST_WEIGHTS["current"]),
    ais_weight: float = 0.0,
    turn_weight: float = 0.0,
    bounds: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]] | None:
    """Grid A* with environmental edge costs and an optional turn penalty."""
    start = scene.start if start is None else start
    goal = scene.goal if goal is None else goal
    if not scene.in_bounds(start) or not scene.in_bounds(goal):
        return None
    if scene.blocked[start] or scene.blocked[goal]:
        return None
    if bounds is not None:
        row_min, row_max, col_min, col_max = bounds
        if not (row_min <= start[0] <= row_max and col_min <= start[1] <= col_max):
            return None
        if not (row_min <= goal[0] <= row_max and col_min <= goal[1] <= col_max):
            return None

    counter = itertools.count()
    start_key = (start[0], start[1], -1)
    g_score = {start_key: 0.0}
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    queue: list[tuple[float, int, tuple[int, int, int]]] = []
    min_multiplier = max(0.25, 1.0 - abs(float(current_weight)) * float(scene.current_speed.max()))
    heapq.heappush(queue, (min_multiplier * math.dist(start, goal), next(counter), start_key))
    goal_key: tuple[int, int, int] | None = None

    while queue:
        _, _, key = heapq.heappop(queue)
        row, col, previous_dir = key
        if (row, col) == goal:
            goal_key = key
            break
        point = (row, col)
        old_cost = g_score[key]
        for nxt in scene.neighbors(point):
            if bounds is not None and not (row_min <= nxt[0] <= row_max and col_min <= nxt[1] <= col_max):
                continue
            dr, dc = nxt[0] - row, nxt[1] - col
            direction = NEIGHBOR_STEPS.index((dr, dc))
            edge = scene.step_cost(
                point, nxt,
                ice_weight=ice_weight,
                thickness_weight=thickness_weight,
                chart_weight=chart_weight,
                current_weight=current_weight,
                ais_weight=ais_weight,
            )
            if previous_dir >= 0 and turn_weight:
                pdr, pdc = NEIGHBOR_STEPS[previous_dir]
                dot = (pdr * dr + pdc * dc) / (math.hypot(pdr, pdc) * math.hypot(dr, dc))
                edge += float(turn_weight) * (1.0 - dot)
            next_key = (nxt[0], nxt[1], direction if turn_weight else -1)
            candidate = old_cost + edge
            if candidate >= g_score.get(next_key, math.inf):
                continue
            g_score[next_key] = candidate
            parent[next_key] = key
            heuristic = min_multiplier * math.dist(nxt, goal)
            heapq.heappush(queue, (candidate + heuristic, next(counter), next_key))

    if goal_key is None:
        return None
    route: list[tuple[int, int]] = []
    cursor: tuple[int, int, int] | None = goal_key
    while cursor is not None:
        route.append((cursor[0], cursor[1]))
        cursor = parent[cursor]
    route.reverse()
    return route


def line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Integer Bresenham segment, including both endpoints."""
    r0, c0 = a
    r1, c1 = b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    error = dc - dr
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((r0, c0))
        if (r0, c0) == (r1, c1):
            break
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c0 += sc
        if twice < dc:
            error += dc
            r0 += sr
    return cells


def line_is_free(scene: Scene, a: tuple[int, int], b: tuple[int, int]) -> bool:
    cells = line_cells(a, b)
    if any(scene.blocked[cell] for cell in cells):
        return False
    for first, second in zip(cells, cells[1:]):
        dr, dc = second[0] - first[0], second[1] - first[1]
        if dr and dc and (scene.blocked[first[0] + dr, first[1]] or scene.blocked[first[0], first[1] + dc]):
            return False
    return True


def smooth_line_of_sight(scene: Scene, route: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedily remove visible intermediate cells while preserving hard safety."""
    if len(route) < 3:
        return list(route)
    output = [route[0]]
    index = 0
    while index < len(route) - 1:
        target = len(route) - 1
        while target > index + 1 and not line_is_free(scene, route[index], route[target]):
            target -= 1
        output.extend(line_cells(route[index], route[target])[1:])
        index = target
    return output
