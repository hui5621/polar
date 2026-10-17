"""Numerical settings for route planning and training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteConfig:
    # State and candidate scoring.
    base_state_dim: int = 17
    state_dim: int = 25
    action_dim: int = 6
    preview_scale: float = 5.0
    preview_w_cost: float = 1.0
    preview_w_thickness: float = 18.0
    preview_w_turn: float = 0.03
    preview_w_current: float = 4.0
    preview_w_ice_exposure: float = 0.0
    vessel_w_travel_time: float = 0.40
    vessel_w_power_margin: float = 2.0
    vessel_w_ice_infeasibility: float = 0.35
    vessel_w_turn_infeasibility: float = 0.25

    # Candidate mask and Shadow gate.
    turn_mask_restrict: float = 0.45
    turn_mask_shorten: float = 0.70
    rejoin_pressure_threshold: float = 0.55
    progress_threshold: float = 0.70
    gate_w_cost: float = 0.28
    gate_w_ice: float = 360.0
    gate_w_thickness: float = 300.0
    gate_w_turn: float = 0.04
    gate_w_current: float = 0.85
    gate_base_margin: float = 0.02
    gate_recovery_margin: float = 0.60
    gate_keep_margin: float = 0.90
    gate_recovery_enable: float = 0.12
    gate_keep_enable: float = 0.46

    # SDA vessel model.
    vessel_length_m: float = 128.9
    vessel_beam_m: float = 24.0
    vessel_draft_m: float = 7.0
    vessel_reference_speed_kmh: float = 24.076
    vessel_speed_upper_kmh: float = 26.5
    vessel_solver_iterations: int = 32
    vessel_installed_power_kw: float = 11_000.0
    vessel_usable_power_fraction: float = 0.85
    vessel_propulsion_efficiency: float = 0.70
    vessel_open_water_power_fraction: float = 0.55
    vessel_open_water_exponent: float = 2.0
    vessel_ice_density_kg_m3: float = 900.0
    vessel_water_density_kg_m3: float = 1_025.0
    vessel_ice_bending_strength_pa: float = 600_000.0
    vessel_ice_youngs_modulus_pa: float = 5.0e9
    vessel_ice_poisson_ratio: float = 0.30
    vessel_ice_friction: float = 0.10
    vessel_stem_angle_deg: float = 25.0
    vessel_waterline_angle_deg: float = 20.0
    vessel_max_ice_concentration: float = 0.80
    vessel_ice_concentration_exponent: float = 2.0
    vessel_sfoc_g_kwh: float = 205.0

    # Training.
    ants: int = 30
    siaco_iterations: int = 225
    max_episodes: int = 500
    max_decision_steps: int = 420
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.20
    minibatch_size: int = 128
    ppo_epochs: int = 3
    entropy_coef: float = 0.20
    value_coef: float = 0.50
    actor_learning_rate: float = 5.0e-5
    critic_learning_rate: float = 5.0e-5
    max_grad_norm: float = 0.50

    # Reward normalization scales and coefficients.
    reward_progress_scale: float = 2.07
    scale_path_length: float = 6.9853
    scale_total_cost: float = 113.3375
    scale_travel_time: float = 0.28135
    scale_mean_ice: float = 0.075341
    scale_mean_thickness: float = 0.092083
    scale_ice_exposure: float = 12.75399
    scale_turn_complexity: float = 33.90919
    scale_current_assist: float = 0.557353
    reward_progress_weight: float = 0.05
    reward_quality_weight: float = 1.00
    reward_rejoin_penalty: float = 0.25
    reward_hard_penalty: float = 1.00
    reward_soft_penalty: float = 0.0
    reward_goal_bonus: float = 0.0
    reward_timeout_penalty: float = 0.0

    # Auxiliary learning coefficients.
    aux_behavior_cloning: float = 0.02
    aux_counterfactual: float = 0.08
    aux_quality_boundary: float = 0.0
    aux_route_boundary: float = 0.08
    aux_action_margin: float = 0.20
    aux_kl: float = 0.01


CONFIG = RouteConfig()

# Deterministic algorithm parameters.
ALGORITHM_SETTINGS = {
    "pareto_tolerance_normalized": 1e-3,
    "pareto_archive_limit": 80,
    "repair_radius_cells": (4, 12),
    "repair_max_windows_per_route": 8,
    "soft_risk_threshold_quantile": 0.75,
    "soft_risk_ratio_threshold": 0.15,
    "soft_budget_excess_per_km": 0.08,
    "route_cost_weights": {"ice": 1.0, "thickness": 0.50, "risk": 0.10, "soft": 0.10, "current": 0.20},
    "environment_hard_ice_concentration": 0.80,
    "environment_hard_ice_thickness_m": 1.50,
    "route_sample_step_cells": 0.35,
    "ice_exposure_threshold": 0.01,
    "turn_complexity_threshold_deg": 6.0,
    "decision_stride_cells": 2,
    "candidate_horizon_cells": 12,
    "ant_max_expansions": 12000,
    "ant_global_distance_slack_ratio": 0.05,
    "pheromone_deposit_constant": 1.0,
    "pheromone_exchange": (0.18, 0.10),
    "evaporation_rate_bounds": (0.03, 0.65),
    "gate_operator_adjustments": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "safe_rejoin_bias": 0.75,
    "repeat_smooth_penalty": 0.10,
    "rejected_rejoin_bias": 0.10,
    "candidate_quality_bias_clip": 2.0,
    "aux_action_margin_threshold": 0.20,
    "candidate_search_weights": {
        "risk_detour": {"ice": 4.0, "thickness": 1.2, "chart": 1.0, "current": 0.15, "turn": 0.0},
        "current_shift": {"ice": 2.5, "thickness": 0.7, "chart": 0.4, "current": 1.25, "turn": 0.0},
        "smooth_turn": {"ice": 2.5, "thickness": 0.7, "chart": 0.3, "current": 0.5, "turn": 1.5},
        "safe_rejoin": {"ice": 2.0, "thickness": 0.5, "chart": 0.25, "current": 0.6, "turn": 0.3},
    },
    "ant_guidance_weights": {"ice": 4.0, "thickness": 1.2, "chart": 1.0, "current": 0.15},
}
