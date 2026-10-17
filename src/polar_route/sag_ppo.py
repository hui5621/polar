from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import CONFIG, ALGORITHM_SETTINGS
from .geometry import turn_angles_deg
from .metrics import route_metrics
from .ppo import DiscretePPO
from .scene import Scene
from .search import astar, line_cells, line_is_free, smooth_line_of_sight
from .siaco import _route_soft_excess, _soft_budget, _soft_risk
from .vessel import safe_speed_kmh


OPERATORS = (
    "KEEP_PRIOR", "SHORTEN_RECONNECT", "RISK_DETOUR",
    "CURRENT_ASSIST_SHIFT", "SMOOTH_TURN", "SAFE_REJOIN_PRIOR",
)


@dataclass
class Candidate:
    path: list[tuple[int, int]] | None
    metrics: dict[str, Any]
    deltas: np.ndarray
    q_raw: float
    gate_score: float
    accepted: bool
    turn_pressure: float
    rejoin_pressure: float


@dataclass
class Decision:
    base_observation: np.ndarray
    mask: np.ndarray
    bias_state: np.ndarray
    bias_quality: np.ndarray
    candidates: list[Candidate]
    prior_path: list[tuple[int, int]]
    prior_score: float
    progress: float


def _local_bounds(scene: Scene, start: tuple[int, int], goal: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    return (max(0, min(start[0], goal[0]) - margin), min(scene.rows - 1, max(start[0], goal[0]) + margin),
            max(0, min(start[1], goal[1]) - margin), min(scene.cols - 1, max(start[1], goal[1]) + margin))


def _segment_metrics(scene: Scene, route: list[tuple[int, int]]) -> dict[str, Any]:
    return route_metrics(scene, route)


def _candidate_paths(scene: Scene, prior: list[tuple[int, int]], start_index: int, end_index: int,
                     *, window: int) -> list[list[tuple[int, int]] | None]:
    start, goal = prior[start_index], prior[end_index]
    prior_segment = list(prior[start_index:end_index + 1])
    bounds = _local_bounds(scene, start, goal, max(2, int(window)))
    shorten: list[tuple[int, int]] | None
    if line_is_free(scene, start, goal):
        shorten = line_cells(start, goal)
    else:
        shorten = smooth_line_of_sight(scene, prior_segment)
    weights = ALGORITHM_SETTINGS["candidate_search_weights"]
    risk_w = weights["risk_detour"]
    risk = astar(scene, start, goal, ice_weight=risk_w["ice"], thickness_weight=risk_w["thickness"],
                 chart_weight=risk_w["chart"], current_weight=risk_w["current"],
                 turn_weight=risk_w["turn"], ais_weight=0.0, bounds=bounds)
    current_w = weights["current_shift"]
    current = astar(scene, start, goal, ice_weight=current_w["ice"], thickness_weight=current_w["thickness"],
                    chart_weight=current_w["chart"], current_weight=current_w["current"],
                    turn_weight=current_w["turn"], ais_weight=0.0, bounds=bounds)
    smooth_w = weights["smooth_turn"]
    smooth = astar(scene, start, goal, ice_weight=smooth_w["ice"], thickness_weight=smooth_w["thickness"],
                   chart_weight=smooth_w["chart"], current_weight=smooth_w["current"],
                   turn_weight=smooth_w["turn"], ais_weight=0.0, bounds=bounds)
    rejoin_w = weights["safe_rejoin"]
    rejoin = astar(scene, start, goal, ice_weight=rejoin_w["ice"], thickness_weight=rejoin_w["thickness"],
                   chart_weight=rejoin_w["chart"], current_weight=rejoin_w["current"],
                   turn_weight=rejoin_w["turn"], ais_weight=0.0, bounds=bounds)
    if smooth:
        smooth = smooth_line_of_sight(scene, smooth)
    # Every operator reconnects to the same downstream prior anchor, so the
    # selected segment can be verified and spliced without changing P0.
    return [prior_segment, shorten, risk, current, smooth, rejoin]


def _hard_safe(scene: Scene, route: list[tuple[int, int]] | None) -> bool:
    return bool(route and all(scene.in_bounds(p) and not scene.blocked[p] for p in route)
                and all(line_is_free(scene, a, b) for a, b in zip(route[:-1], route[1:])))


def _segment_deltas(prior: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    prior_tt = float(prior["travel_time_h"] or 1e9)
    candidate_tt = float(candidate["travel_time_h"] or 1e9)
    return np.asarray([
        float(prior["path_length_km"]) - float(candidate["path_length_km"]),
        float(prior["total_cost"]) - float(candidate["total_cost"]),
        prior_tt - candidate_tt,
        float(prior["mean_ice_concentration"]) - float(candidate["mean_ice_concentration"]),
        float(prior["mean_ice_thickness_m"]) - float(candidate["mean_ice_thickness_m"]),
        float(prior["ice_exposure_km"]) - float(candidate["ice_exposure_km"]),
        float(prior["turn_complexity_deg"]) - float(candidate["turn_complexity_deg"]),
        float(candidate["current_assist_m_s"]) - float(prior["current_assist_m_s"]),
    ], dtype=np.float64)


def _preview(scene: Scene, route: list[tuple[int, int]] | None, prior_metrics: dict[str, Any],
             prior_segment: list[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int],
             operator_index: int) -> Candidate:
    if route is None:
        return Candidate(None, {}, np.zeros(8), -1e6, -1e6, False, 1.0, 1.0)
    metrics = _segment_metrics(scene, route)
    deltas = _segment_deltas(prior_metrics, metrics)
    travel_delta = float(deltas[2])
    margin_delta = float(metrics.get("propulsion_margin_min", 0.0)) - float(prior_metrics.get("propulsion_margin_min", 0.0))
    infeasible_delta = float(prior_metrics.get("icebreaker_infeasible_samples", 0.0)) - float(metrics.get("icebreaker_infeasible_samples", 0.0))
    turn_bad_delta = sum(a >= 45.0 for a in turn_angles_deg(prior_segment)) - sum(a >= 45.0 for a in turn_angles_deg(route))
    vessel_preview = (
        CONFIG.vessel_w_travel_time * travel_delta
        + CONFIG.vessel_w_power_margin * margin_delta
        + CONFIG.vessel_w_ice_infeasibility * infeasible_delta
        + CONFIG.vessel_w_turn_infeasibility * turn_bad_delta
    )
    q_raw = (
        CONFIG.preview_w_cost * deltas[1]
        + CONFIG.preview_w_thickness * deltas[4]
        + CONFIG.preview_w_turn * deltas[6]
        + CONFIG.preview_w_current * deltas[7]
        + CONFIG.preview_w_ice_exposure * deltas[5]
        + vessel_preview
    )
    hard = _hard_safe(scene, route)
    dynamic = bool(metrics.get("dynamic_feasible", False))
    soft = _soft_risk(scene)
    threshold = float(np.quantile(soft[~scene.blocked], float(ALGORITHM_SETTINGS["soft_risk_threshold_quantile"])))
    budget = _soft_budget(scene, float(metrics["path_length_km"]))
    soft_ok = _route_soft_excess(scene, route, threshold) <= budget
    rejoin_ok = route[0] == start and route[-1] == goal
    turn_angles = turn_angles_deg(prior_segment)
    turn_pressure = float(np.clip(sum(turn_angles) / max(len(turn_angles) * 90.0, 1.0), 0.0, 1.0))
    straight = max(math.dist(start, goal), 1e-9)
    candidate_length = sum(math.dist(a, b) for a, b in zip(route[:-1], route[1:]))
    rejoin_pressure = float(np.clip((candidate_length / straight - 1.0) / 2.0, 0.0, 1.0))
    feasible = hard and dynamic and soft_ok and rejoin_ok
    metrics["_soft_ok"] = soft_ok
    metrics["_hard_ok"] = hard
    metrics["_rejoin_ok"] = rejoin_ok
    metrics["_dynamic_ok"] = dynamic
    gate = (
        CONFIG.gate_w_cost * deltas[1]
        + CONFIG.gate_w_ice * deltas[3]
        + CONFIG.gate_w_thickness * deltas[4]
        + CONFIG.gate_w_turn * deltas[6]
        + CONFIG.gate_w_current * deltas[7]
        + float(ALGORITHM_SETTINGS["gate_operator_adjustments"][operator_index])
    )
    return Candidate(route, metrics, deltas, float(q_raw), float(gate), feasible, turn_pressure, rejoin_pressure)


def _decision_observation(scene: Scene, prior: list[tuple[int, int]], index: int,
                          candidates: list[Candidate]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point = prior[index]
    goal = scene.goal
    route_n = max(len(prior) - 1, 1)
    goal_dc, goal_north = goal[1] - point[1], point[0] - goal[0]
    goal_norm = max(math.hypot(goal_dc, goal_north), 1e-9)
    prev = prior[max(0, index - 1)]
    nxt = prior[min(len(prior) - 1, index + 1)]
    tdc, tnorth = nxt[1] - prev[1], prev[0] - nxt[0]
    tangent_norm = max(math.hypot(tdc, tnorth), 1e-9)
    dx, dy = (point[1] - scene.start[1]), (scene.start[0] - point[0])
    norm_x = max(scene.cols - 1, 1)
    norm_y = max(scene.rows - 1, 1)
    ice = float(scene.ice_concentration[point])
    thickness = float(scene.ice_thickness[point])
    risk = float(scene.chart_risk[point])
    speed, _, _, margin = safe_speed_kmh(ice, thickness)
    mobility = speed / max(CONFIG.vessel_speed_upper_kmh, 1e-9)
    current_u = float(scene.current_u[point])
    current_v = float(scene.current_v[point])
    current_scale = max(float(np.quantile(scene.current_speed[~scene.blocked], 0.98)), 1e-6)
    clearance = 0
    for radius in range(1, 9):
        r0, r1 = max(0, point[0] - radius), min(scene.rows, point[0] + radius + 1)
        c0, c1 = max(0, point[1] - radius), min(scene.cols, point[1] + radius + 1)
        if np.any(scene.blocked[r0:r1, c0:c1]):
            break
        clearance = radius
    turn_angles = turn_angles_deg(prior[max(0, index - 1):min(len(prior), index + 2)])
    turn_info = float(turn_angles[0] / 180.0) if turn_angles else 0.0
    base = np.asarray([
        2.0 * dx / norm_x - 1.0, 2.0 * dy / norm_y - 1.0,
        goal_dc / goal_norm, goal_north / goal_norm,
        min(goal_norm / max(math.hypot(scene.rows, scene.cols), 1.0), 1.0),
        tdc / tangent_norm, tnorth / tangent_norm,
        0.0, index / route_n,
        ice, np.clip(thickness / 2.0, 0.0, 1.0), risk, mobility,
        np.clip(current_u / current_scale, -1.0, 1.0),
        np.clip(current_v / current_scale, -1.0, 1.0),
        clearance / 8.0, turn_info,
    ], dtype=np.float32)
    q_obs = np.asarray([float(np.clip(c.q_raw / CONFIG.preview_scale, -3.0, 3.0)) for c in candidates], dtype=np.float32)
    quality_limit = float(ALGORITHM_SETTINGS["candidate_quality_bias_clip"])
    quality_bias = np.asarray([float(np.clip(q, -quality_limit, quality_limit)) for q in q_obs], dtype=np.float32)
    turn_pressure = candidates[0].turn_pressure
    rejoin_pressure = candidates[0].rejoin_pressure
    state_bias = np.zeros(CONFIG.action_dim, dtype=np.float32)
    if turn_pressure > 0.60:
        state_bias[4] += 1.05
    if rejoin_pressure > CONFIG.rejoin_pressure_threshold or index / route_n > CONFIG.progress_threshold:
        state_bias[5] += float(ALGORITHM_SETTINGS["safe_rejoin_bias"])
    if turn_pressure > CONFIG.turn_mask_restrict:
        state_bias[4] += 0.2
    return base, q_obs, state_bias, quality_bias


def build_decisions(scene: Scene, prior: list[tuple[int, int]], *, window: int = 12,
                    max_decision_steps: int = CONFIG.max_decision_steps,
                    apply_guidance_mask: bool = True) -> list[Decision]:
    if len(prior) < 2 or prior[0] != scene.start or prior[-1] != scene.goal:
        raise ValueError("S-IACO prior must connect the scene start and goal")
    stride = int(ALGORITHM_SETTINGS["decision_stride_cells"])
    decisions: list[Decision] = []
    index = 0
    while index < len(prior) - 1 and len(decisions) < max_decision_steps:
        end_index = min(index + stride, len(prior) - 1)
        prior_segment = list(prior[index:end_index + 1])
        raw_paths = _candidate_paths(scene, prior, index, end_index, window=window)
        prior_metrics = _segment_metrics(scene, prior_segment)
        candidates = [
            _preview(scene, path, prior_metrics, prior_segment, prior[index], prior[end_index], operator_index)
            for operator_index, path in enumerate(raw_paths)
        ]
        mask = np.asarray([c.accepted for c in candidates], dtype=np.bool_)
        mask[0] = True
        if apply_guidance_mask:
            turn_pressure = candidates[0].turn_pressure
            if turn_pressure > CONFIG.turn_mask_restrict:
                mask[2] = False
                mask[3] = False
            if turn_pressure > CONFIG.turn_mask_shorten:
                mask[1] = False
            if candidates[1].deltas[1] < -1e-9:
                mask[1] = False
            if candidates[2].deltas[3] <= 0.0 and candidates[2].deltas[4] <= 0.0:
                mask[2] = False
            if candidates[3].deltas[7] <= 0.0:
                mask[3] = False
        if not mask[0]:
            raise RuntimeError("KEEP_PRIOR must remain an available fallback")
        base, q_obs, state_bias, quality_bias = _decision_observation(scene, prior, index, candidates)
        progress = index / max(len(prior) - 1, 1)
        decisions.append(Decision(base, mask, state_bias, quality_bias, candidates,
                                  prior_segment, float(prior_metrics["total_cost"]), progress))
        index = end_index
    if not decisions:
        raise RuntimeError("No SAG-PPO decision segments were generated")
    return decisions


def _guided_bias(decision: Decision, alpha: float, previous_action: int, previous_accepted: bool) -> np.ndarray:
    bias = alpha * decision.bias_state + decision.bias_quality
    if previous_action == 4:
        bias[4] -= float(ALGORITHM_SETTINGS["repeat_smooth_penalty"])
    if not previous_accepted:
        bias[5] += float(ALGORITHM_SETTINGS["rejected_rejoin_bias"])
    return bias.astype(np.float32)


def _observation(decision: Decision, previous_action: int, previous_accepted: bool) -> np.ndarray:
    history = np.asarray([
        2.0 * previous_action / max(CONFIG.action_dim - 1, 1) - 1.0,
        1.0 if previous_accepted else -1.0,
    ], dtype=np.float32)
    preview = np.asarray(
        [np.clip(candidate.q_raw / CONFIG.preview_scale, -3.0, 3.0)
         for candidate in decision.candidates],
        dtype=np.float32,
    )
    return np.concatenate([decision.base_observation, preview, history])


def _gate(decision: Decision, proposed: int, alpha: float) -> int:
    gate_strength = float(np.clip(2.0 * alpha**2, 0.0, 1.0))
    proposal = decision.candidates[proposed]
    keep = decision.candidates[0]
    if not proposal.accepted:
        return 0
    executed = proposed
    keep_margin = CONFIG.gate_base_margin + CONFIG.gate_keep_margin * (1.0 - gate_strength)
    recovery_margin = CONFIG.gate_base_margin + CONFIG.gate_recovery_margin * (1.0 - gate_strength)
    if gate_strength >= CONFIG.gate_keep_enable and proposal.q_raw < keep.q_raw - keep_margin:
        executed = 0
    if gate_strength >= CONFIG.gate_recovery_enable:
        eligible = []
        for action in (4, 5):
            candidate = decision.candidates[action]
            if not candidate.accepted:
                continue
            if candidate.metrics["total_cost"] > keep.metrics["total_cost"] + 1e-9:
                continue
            if candidate.metrics["mean_ice_concentration"] > keep.metrics["mean_ice_concentration"] + 1e-9:
                continue
            if candidate.metrics["mean_ice_thickness_m"] > keep.metrics["mean_ice_thickness_m"] + 1e-9:
                continue
            if candidate.gate_score >= proposal.gate_score + recovery_margin:
                eligible.append(action)
        if eligible:
            executed = max(eligible, key=lambda a: decision.candidates[a].gate_score)
    return int(executed)


def _evaluate_policy(scene: Scene, decisions: list[Decision], agent: DiscretePPO,
                     alpha: float) -> list[tuple[int, int]]:
    previous_action = 0
    previous_accepted = True
    actions: list[int] = []
    for decision in decisions:
        observation = _observation(decision, previous_action, previous_accepted)
        bias = _guided_bias(decision, alpha, previous_action, previous_accepted)
        proposed, _, _, _ = agent.act(observation, decision.mask, bias, deterministic=True)
        executed = _gate(decision, proposed, alpha)
        actions.append(executed)
        previous_action = executed
        previous_accepted = proposed == executed
    return _assemble_route(scene, decisions, actions)


def _reward(decision: Decision, executed: int, *, is_last: bool, is_timeout: bool = False) -> float:
    candidate = decision.candidates[executed]
    prior = decision.candidates[0]
    scales = np.asarray([
        CONFIG.scale_path_length, CONFIG.scale_total_cost, CONFIG.scale_travel_time,
        CONFIG.scale_mean_ice, CONFIG.scale_mean_thickness, CONFIG.scale_ice_exposure,
        CONFIG.scale_turn_complexity, CONFIG.scale_current_assist,
    ])
    gains = np.clip(candidate.deltas / scales, -1.0, 1.0)
    g_eff = float(np.mean(gains[[0, 1, 2]]))
    g_risk = float(np.mean(gains[[3, 4, 5]]))
    g_geo = float(gains[6])
    g_env = float(gains[7])
    prior_distance = sum(math.dist(a, b) for a, b in zip(decision.prior_path[:-1], decision.prior_path[1:]))
    progress = float(np.clip(max(prior_distance, 0.0) / CONFIG.reward_progress_scale, 0.0, 1.0))
    valid_rejoin = bool(candidate.metrics.get("_rejoin_ok", False))
    hard_violation = not bool(candidate.metrics.get("_hard_ok", False))
    soft_violation = not bool(candidate.metrics.get("_soft_ok", False))
    goal = bool(is_last)
    return float(
        CONFIG.reward_progress_weight * progress
        + CONFIG.reward_quality_weight * (g_eff + g_risk + g_geo + g_env) / 4.0
        - CONFIG.reward_rejoin_penalty * (not valid_rejoin)
        - CONFIG.reward_hard_penalty * hard_violation
        - CONFIG.reward_soft_penalty * soft_violation
        + CONFIG.reward_goal_bonus * goal
        - CONFIG.reward_timeout_penalty * bool(is_timeout)
    )


def _assemble_route(scene: Scene, decisions: list[Decision], actions: list[int]) -> list[tuple[int, int]]:
    route = [scene.start]
    for decision, action in zip(decisions, actions):
        segment = decision.candidates[action].path
        if segment is None or segment[0] != route[-1]:
            raise RuntimeError("Executed candidate does not connect to the route prefix")
        route.extend(segment[1:])
    return route


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sag_ppo(scene: Scene, prior: list[tuple[int, int]], *, episodes: int = CONFIG.max_episodes,
                  seed: int = 0, device: str = "cpu", window: int = int(ALGORITHM_SETTINGS["candidate_horizon_cells"]),
                  max_decision_steps: int = CONFIG.max_decision_steps) -> tuple[list[tuple[int, int]], list[dict[str, float]], DiscretePPO]:
    """Train SAG-PPO on a fixed S-IACO prior and six verified local operators."""
    _seed_everything(seed)
    decisions = build_decisions(scene, prior, window=window, max_decision_steps=max_decision_steps)
    agent = DiscretePPO(device=device)
    history: list[dict[str, float]] = []
    episode_count = max(1, int(episodes))
    for episode in range(1, episode_count + 1):
        alpha = (episode - 1) / max(episode_count - 1, 1)
        previous_action = 0
        previous_accepted = True
        rewards: list[float] = []
        rollout: list[dict[str, Any]] = []
        for step, decision in enumerate(decisions):
            observation = _observation(decision, previous_action, previous_accepted)
            bias = _guided_bias(decision, alpha, previous_action, previous_accepted)
            proposed, logp, value, old_probs = agent.act(observation, decision.mask, bias)
            executed = _gate(decision, proposed, alpha)
            last_step = step == len(decisions) - 1
            reward = _reward(
                decision, executed, is_last=(last_step and decision.prior_path[-1] == scene.goal),
                is_timeout=(last_step and decision.prior_path[-1] != scene.goal),
            )
            rewards.append(reward)
            rollout.append({
                "observation": observation,
                "mask": decision.mask.copy(),
                "bias": bias.copy(),
                "proposed_action": proposed,
                "executed_action": executed,
                "log_prob": logp,
                "old_probs": old_probs,
                "value": value,
                "reward": reward,
                "done": step == len(decisions) - 1,
                "actor_include": float(proposed == executed),
            })
            previous_action = executed
            previous_accepted = proposed == executed
        losses = agent.update(rollout)
        evaluation_route = _evaluate_policy(scene, decisions, agent, alpha)
        metrics = route_metrics(scene, evaluation_route)
        history.append({
            "episode": float(episode), "reward": float(np.mean(rewards) if rewards else 0.0),
            "route_total_cost": float(metrics["total_cost"]),
            "route_length_km": float(metrics["path_length_km"]),
            "route_travel_time_h": float(metrics["travel_time_h"] or 0.0),
            "route_fuel_tons": float(metrics["fuel_tons"] or 0.0),
            "route_ice_exposure_km": float(metrics["ice_exposure_km"]),
            "route_turn_complexity_deg": float(metrics["turn_complexity_deg"]),
            "actor_inclusion_rate": float(np.mean([r["actor_include"] for r in rollout])),
            "gate_rewrite_rate": float(np.mean([r["actor_include"] == 0.0 for r in rollout])),
            **losses,
        })

    final_route = _evaluate_policy(scene, decisions, agent, alpha=1.0)
    return final_route, history, agent
