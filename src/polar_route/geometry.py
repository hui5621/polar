from __future__ import annotations

import math
from collections.abc import Sequence

from .config import UNSPECIFIED


def segment_samples(
    a: Sequence[float], b: Sequence[float], *, rows: int, cols: int,
    step_cells: float | None = None,
) -> list[tuple[int, int, float]]:
    """Sample a row/column segment at the reporting resolution.

    Samples start just after the first vertex and include the final vertex,
    matching the frozen-grid route evaluator used for the paper metrics.
    """
    p0r, p0c = float(a[0]), float(a[1])
    p1r, p1c = float(b[0]), float(b[1])
    distance = math.hypot(p1r - p0r, p1c - p0c)
    step = float(step_cells if step_cells is not None else UNSPECIFIED["route_sample_step_cells"])
    steps = max(1, int(math.ceil(distance / max(step, 1e-9))))
    out: list[tuple[int, int, float]] = []
    for index in range(1, steps + 1):
        t = index / steps
        row = int(math.floor(p0r + t * (p1r - p0r) + 0.5))
        col = int(math.floor(p0c + t * (p1c - p0c) + 0.5))
        out.append((min(max(row, 0), rows - 1), min(max(col, 0), cols - 1), t))
    return out


def turn_angles_deg(path: Sequence[Sequence[float]]) -> list[float]:
    angles: list[float] = []
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        # Projected east/north directions; row increases southward.
        v1 = (float(b[1]) - float(a[1]), float(a[0]) - float(b[0]))
        v2 = (float(c[1]) - float(b[1]), float(b[0]) - float(c[0]))
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 1e-12 or n2 <= 1e-12:
            continue
        cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append(math.degrees(math.acos(cosine)))
    return angles
