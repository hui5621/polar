from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PAPER, UNSPECIFIED

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "environments" / "projected_nsr_2025"
MANIFEST_PATH = DATA_DIR / "manifest.json"

NEIGHBOR_STEPS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)


@dataclass
class Scene:
    date: str
    start: tuple[int, int]
    goal: tuple[int, int]
    cell_km: float
    ice_class: np.ndarray
    ice_concentration: np.ndarray
    ice_thickness: np.ndarray
    current_u: np.ndarray
    current_v: np.ndarray
    current_speed: np.ndarray
    chart_risk: np.ndarray
    chart_block_mask: np.ndarray
    land_mask: np.ndarray
    ice_block_mask: np.ndarray
    bathymetry: np.ndarray
    ais_corridor: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.land_mask.shape

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @cached_property
    def blocked(self) -> np.ndarray:
        # Eq. (2) hard-infeasibility mask includes the ship's 0.80 ice limit.
        return (
            (self.land_mask > 0)
            | (self.chart_block_mask > 0)
            | (self.ice_block_mask > 0)
            | (self.ice_concentration >= PAPER.vessel_max_ice_concentration)
        )

    def in_bounds(self, point: tuple[int, int]) -> bool:
        row, col = point
        return 0 <= row < self.rows and 0 <= col < self.cols

    def neighbors(self, point: tuple[int, int]) -> Iterable[tuple[int, int]]:
        row, col = point
        blocked = self.blocked
        for dr, dc in NEIGHBOR_STEPS:
            nxt = (row + dr, col + dc)
            if not self.in_bounds(nxt) or blocked[nxt]:
                continue
            # A diagonal move may not cut across a blocked corner.
            if dr and dc and (blocked[row + dr, col] or blocked[row, col + dc]):
                continue
            yield nxt

    def current_assist(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        dr, dc = b[0] - a[0], b[1] - a[1]
        norm = float(np.hypot(dr, dc))
        if norm == 0.0:
            return 0.0
        # Raster rows grow downwards; projected north is therefore -row.
        east, north = dc / norm, -dr / norm
        u = 0.5 * (float(self.current_u[a]) + float(self.current_u[b]))
        v = 0.5 * (float(self.current_v[a]) + float(self.current_v[b]))
        return east * u + north * v

    def step_cost(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
        *,
        ice_weight: float = float(UNSPECIFIED["route_cost_weights"]["ice"]),
        thickness_weight: float = float(UNSPECIFIED["route_cost_weights"]["thickness"]),
        chart_weight: float = float(UNSPECIFIED["route_cost_weights"]["risk"]),
        current_weight: float = float(UNSPECIFIED["route_cost_weights"]["current"]),
        ais_weight: float = 0.0,
    ) -> float:
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        ice = 0.5 * (float(self.ice_concentration[a]) + float(self.ice_concentration[b]))
        thick = 0.5 * (float(self.ice_thickness[a]) + float(self.ice_thickness[b]))
        chart = 0.5 * (float(self.chart_risk[a]) + float(self.chart_risk[b]))
        assist = self.current_assist(a, b)
        normalized_thick = float(np.clip(thick / 2.0, 0.0, 1.0))
        soft = float(np.clip(ice_weight * ice + thickness_weight * normalized_thick + chart_weight * chart, 0.0, 1.0))
        multiplier = (
            1.0 + ice_weight * ice + thickness_weight * normalized_thick
            + chart_weight * chart + float(UNSPECIFIED["route_cost_weights"]["soft"]) * soft
            - current_weight * assist + ais_weight * (1.0 - 0.5 * (float(self.ais_corridor[a]) + float(self.ais_corridor[b])))
        )
        return length * max(0.25, multiplier)


def list_scenes() -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [row["date"] for row in manifest["dates"]]


def load_scene(scene: str | int) -> Scene:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = manifest["dates"]
    if isinstance(scene, int):
        if scene < 1 or scene > len(rows):
            raise ValueError(f"Scene number must be between 1 and {len(rows)}")
        try:
            item = rows[scene - 1]
        except IndexError as exc:
            raise ValueError(f"Scene number must be between 1 and {len(rows)}") from exc
    else:
        token = str(scene).replace("_", "-")
        item = next((row for row in rows if row["date"] == token), None)
        if item is None:
            item = next((row for row in rows if row["date"].replace("-", "") == token), None)
        if item is None:
            available = ", ".join(row["date"] for row in rows)
            raise ValueError(f"Unknown scene {scene!r}. Available dates: {available}")

    with np.load(DATA_DIR / item["file"], allow_pickle=False) as pack:
        values = {key: pack[key].copy() for key in pack.files}
    scene_data = Scene(
        date=item["date"],
        start=tuple(int(x) for x in values["start_rc"]),
        goal=tuple(int(x) for x in values["goal_rc"]),
        cell_km=float(values["cell_km"]),
        ice_class=values["ice_class"],
        ice_concentration=values["ice_concentration"],
        ice_thickness=values["ice_thickness"],
        current_u=values["current_u"],
        current_v=values["current_v"],
        current_speed=values["current_speed"],
        chart_risk=values["chart_risk"],
        chart_block_mask=values["chart_block_mask"],
        land_mask=values["land_mask"],
        ice_block_mask=values["ice_block_mask"],
        bathymetry=values["bathymetry"],
        ais_corridor=values["ais_corridor"],
    )
    if scene_data.blocked[scene_data.start] or scene_data.blocked[scene_data.goal]:
        raise ValueError(f"Start or goal is blocked in scene {scene_data.date}")
    return scene_data
