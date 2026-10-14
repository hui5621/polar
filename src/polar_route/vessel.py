"""SDA ice-resistance, power-balance, current, travel-time and fuel model.

STW controls resistance, shaft power and fuel rate. Along-track ocean current
is added only to obtain SOG, which controls travel time and total fuel.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from .config import PAPER
from .geometry import segment_samples
from .scene import Scene


G = 9.81


@lru_cache(maxsize=8192)
def safe_speed_kmh(ice: float, thickness_m: float) -> tuple[float, float, float, float]:
    """Return maximum feasible STW, ice resistance, required shaft power, margin."""
    c = float(np.clip(ice, 0.0, 1.0))
    h = max(0.0, float(thickness_m))
    cfg = PAPER
    if c > cfg.vessel_max_ice_concentration:
        return 0.0, 0.0, 0.0, 0.0

    rho_i, rho_w = cfg.vessel_ice_density_kg_m3, cfg.vessel_water_density_kg_m3
    sigma, young, nu = cfg.vessel_ice_bending_strength_pa, cfg.vessel_ice_youngs_modulus_pa, cfg.vessel_ice_poisson_ratio
    mu = cfg.vessel_ice_friction
    phi = math.radians(cfg.vessel_stem_angle_deg)
    alpha = math.radians(cfg.vessel_waterline_angle_deg)
    psi = math.atan(math.tan(phi) / max(math.sin(alpha), 1e-12))
    cos_psi = max(math.cos(psi), 1e-12)
    if c <= 1e-12 or h <= 1e-12:
        rc = rb = rs = 0.0
    else:
        rc = 0.5 * sigma * h**2 * (
            math.tan(phi) + mu * math.cos(phi) / cos_psi
        ) / max(1.0 - mu * math.sin(phi) / cos_psi, 1e-12)
        bending_denominator = math.sqrt(
            max(young / (12.0 * (1.0 - nu**2) * G * rho_w), 1e-12)
        )
        rb = (
            (27.0 / 64.0) * sigma * cfg.vessel_beam_m * h**1.5 / bending_denominator
            * (math.tan(psi) + mu * math.cos(phi) / max(cos_psi * math.sin(alpha), 1e-12))
            * (1.0 + 1.0 / cos_psi)
        )
        submergence = (
            cfg.vessel_draft_m * (cfg.vessel_beam_m + cfg.vessel_draft_m)
            / max(cfg.vessel_beam_m + 2.0 * cfg.vessel_draft_m, 1e-12)
            + mu * (
                0.7 * cfg.vessel_length_m
                - cfg.vessel_draft_m / max(math.tan(phi), 1e-12)
                - cfg.vessel_beam_m / max(4.0 * math.tan(alpha), 1e-12)
                + cfg.vessel_draft_m * math.cos(phi) * cos_psi
                * math.sqrt(1.0 / max(math.sin(phi) ** 2, 1e-12) + 1.0 / max(math.tan(alpha) ** 2, 1e-12))
            )
        )
        rs = (rho_w - rho_i) * G * h * cfg.vessel_beam_m * submergence

    vref = cfg.vessel_reference_speed_kmh / 3.6
    vub = cfg.vessel_speed_upper_kmh / 3.6
    available = cfg.vessel_installed_power_kw * cfg.vessel_usable_power_fraction
    reference_resistance = (
        available * cfg.vessel_propulsion_efficiency * cfg.vessel_open_water_power_fraction
        * 1000.0 / max(vref, 1e-12)
    )

    def dynamics(speed_ms: float) -> tuple[float, float]:
        ice_resistance = 0.0
        if c > 1e-12 and h > 1e-12:
            level = (
                (rc + rb) * (1.0 + 1.4 * speed_ms / math.sqrt(G * h))
                + rs * (1.0 + 9.4 * speed_ms / math.sqrt(G * cfg.vessel_length_m))
            )
            ice_resistance = max(level, 0.0) * c**cfg.vessel_ice_concentration_exponent
        open_water = reference_resistance * (speed_ms / max(vref, 1e-12)) ** cfg.vessel_open_water_exponent
        required = (open_water + ice_resistance) * speed_ms / cfg.vessel_propulsion_efficiency / 1000.0
        return ice_resistance, required

    lo, hi = 0.0, vub
    for _ in range(cfg.vessel_solver_iterations):
        mid = 0.5 * (lo + hi)
        _, required = dynamics(mid)
        if required <= available:
            lo = mid
        else:
            hi = mid
    ice_resistance, required = dynamics(lo)
    speed_kmh = lo * 3.6
    return speed_kmh, ice_resistance, required, max(0.0, 1.0 - required / max(available, 1e-12))


def evaluate_route_vessel(scene: Scene, route: list[tuple[int, int]]) -> dict[str, Any]:
    """Evaluate vessel feasibility using the route's projected current field."""
    if len(route) < 2:
        return _empty_vessel()

    segment_data: list[tuple[float, list[tuple[int, int]], float, float]] = []
    all_samples: list[tuple[int, int]] = []
    all_stw: list[float] = []
    all_sog: list[float] = []
    all_ice_resistance: list[float] = []
    all_power: list[float] = []
    all_margin: list[float] = []
    all_current: list[float] = []
    for a, b in zip(route[:-1], route[1:]):
        dr, dc = float(b[0] - a[0]), float(b[1] - a[1])
        grid_distance = math.hypot(dr, dc)
        segment_km = grid_distance * scene.cell_km
        if segment_km <= 1e-12:
            continue
        samples = segment_samples(a, b, rows=scene.rows, cols=scene.cols)
        if not samples:
            continue
        east, north = dc / grid_distance, -dr / grid_distance
        segment_start = len(all_samples)
        for row, col, _ in samples:
            stw, ice_resistance, power, margin = safe_speed_kmh(
                float(scene.ice_concentration[row, col]), float(scene.ice_thickness[row, col])
            )
            current_parallel_ms = east * float(scene.current_u[row, col]) + north * float(scene.current_v[row, col])
            sog = stw + 3.6 * current_parallel_ms
            all_samples.append((row, col))
            all_stw.append(stw)
            all_sog.append(sog)
            all_ice_resistance.append(ice_resistance)
            all_power.append(power)
            all_margin.append(margin)
            all_current.append(current_parallel_ms)
        segment_data.append((segment_km, list(range(segment_start, len(all_samples))), east, north))

    stw_arr = np.asarray(all_stw, dtype=np.float64)
    sog_arr = np.asarray(all_sog, dtype=np.float64)
    resistance_arr = np.asarray(all_ice_resistance, dtype=np.float64)
    power_arr = np.asarray(all_power, dtype=np.float64)
    margin_arr = np.asarray(all_margin, dtype=np.float64)
    current_arr = np.asarray(all_current, dtype=np.float64)
    prop_bad = int(np.count_nonzero(stw_arr <= 1e-9))
    ground_bad = int(np.count_nonzero((stw_arr > 1e-9) & (sog_arr <= 1e-9)))
    infeasible_segment_count = 0
    travel_time_h = 0.0
    fuel_tons = 0.0
    for distance_km, indices, _, _ in segment_data:
        idx = np.asarray(indices, dtype=np.int64)
        seg_stw, seg_sog = stw_arr[idx], sog_arr[idx]
        if not len(idx) or np.any(seg_stw <= 1e-9) or np.any(seg_sog <= 1e-9):
            infeasible_segment_count += 1
            continue
        # Harmonic mean SOG and sample-wise fuel rate × SOG transit time.
        travel_time_h += distance_km * float(np.mean(1.0 / seg_sog))
        sample_hours = distance_km / seg_sog / len(idx)
        fuel_rate_tph = power_arr[idx] * PAPER.vessel_sfoc_g_kwh / 1_000_000.0
        fuel_tons += float(np.sum(fuel_rate_tph * sample_hours))

    feasible = bool(segment_data and not prop_bad and not ground_bad and not infeasible_segment_count)
    return {
        "travel_time_h": float(travel_time_h) if feasible else None,
        "fuel_tons": float(fuel_tons) if feasible else None,
        "dynamic_feasible": feasible,
        "mean_safe_speed_kmh": float(np.mean(stw_arr)) if len(stw_arr) else 0.0,
        "min_safe_speed_kmh": float(np.min(stw_arr)) if len(stw_arr) else 0.0,
        "mean_sog_speed_kmh": float(np.mean(sog_arr)) if len(sog_arr) else 0.0,
        "min_sog_speed_kmh": float(np.min(sog_arr)) if len(sog_arr) else 0.0,
        "mean_ice_resistance_n": float(np.mean(resistance_arr)) if len(resistance_arr) else 0.0,
        "mean_required_shaft_power_kw": float(np.mean(power_arr)) if len(power_arr) else 0.0,
        "propulsion_margin_min": float(np.min(margin_arr)) if len(margin_arr) else 0.0,
        "propulsion_infeasible_samples": prop_bad,
        "ground_speed_infeasible_samples": ground_bad,
        "infeasible_segments": infeasible_segment_count,
        "mean_current_assist_m_s": float(np.mean(current_arr)) if len(current_arr) else 0.0,
    }


def _empty_vessel() -> dict[str, Any]:
    return {
        "travel_time_h": None, "fuel_tons": None, "dynamic_feasible": False,
        "mean_safe_speed_kmh": 0.0, "min_safe_speed_kmh": 0.0,
        "mean_sog_speed_kmh": 0.0, "min_sog_speed_kmh": 0.0,
        "mean_ice_resistance_n": 0.0, "mean_required_shaft_power_kw": 0.0,
        "propulsion_margin_min": 0.0, "propulsion_infeasible_samples": 0,
        "ground_speed_infeasible_samples": 0, "infeasible_segments": 0,
        "mean_current_assist_m_s": 0.0,
    }
