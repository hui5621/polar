from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .metrics import route_metrics
from .ppo import DiscretePPO
from .scene import Scene
from .search import astar


OPERATORS = (
    "KEEP_PRIOR",
    "SHORTEN_RECONNECT",
    "RISK_DETOUR",
    "CURRENT_ASSIST_SHIFT",
    "SMOOTH_TURN",
    "SAFE_REJOIN_PRIOR",
)


@dataclass
class Decision:
    observation: np.ndarray
    mask: np.ndarray
    bias: np.ndarray
    paths: list[list[tuple[int, int]] | None]
    scores: np.ndarray
    prior_score: float


def _segment_score(scene: Scene, route: list[tuple[int, int]]) -> float:
    values = route_metrics(scene, route)
    return float(values["total_cost"] + 0.005 * values["turn_complexity_deg"])


def _local_bounds(scene: Scene, start: tuple[int, int], goal: tuple[int, int], margin: int = 18) -> tuple[int, int, int, int]:
    return (
        max(0, min(start[0], goal[0]) - margin),
        min(scene.rows - 1, max(start[0], goal[0]) + margin),
        max(0, min(start[1], goal[1]) - margin),
        min(scene.cols - 1, max(start[1], goal[1]) + margin),
    )


def _candidate_paths(
    scene: Scene,
    prior_segment: list[tuple[int, int]],
) -> list[list[tuple[int, int]] | None]:
    start, goal = prior_segment[0], prior_segment[-1]
    bounds = _local_bounds(scene, start, goal)
    return [
        list(prior_segment),
        astar(scene, start, goal, ice_weight=0.0, thickness_weight=0.0, chart_weight=0.0,
              current_weight=0.0, ais_weight=0.0, bounds=bounds),
        astar(scene, start, goal, ice_weight=7.0, thickness_weight=1.5, chart_weight=8.0,
              current_weight=0.0, ais_weight=0.0, bounds=bounds),
        astar(scene, start, goal, ice_weight=2.0, thickness_weight=0.5, chart_weight=3.0,
              current_weight=1.5, ais_weight=0.0, bounds=bounds),
        astar(scene, start, goal, ice_weight=2.0, thickness_weight=0.5, chart_weight=3.0,
              current_weight=0.5, ais_weight=0.0, turn_weight=0.8, bounds=bounds),
        astar(scene, start, goal, ice_weight=4.0, thickness_weight=1.0, chart_weight=6.0,
              current_weight=0.25, ais_weight=0.0, turn_weight=0.2, bounds=bounds),
    ]


def _is_safe_segment(scene: Scene, candidate: list[tuple[int, int]] | None, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if not candidate or candidate[0] != start or candidate[-1] != goal:
        return False
    if any(not scene.in_bounds(point) or scene.blocked[point] for point in candidate):
        return False
    for first, second in zip(candidate, candidate[1:]):
        if second not in scene.neighbors(first):
            return False
    return True


def _decision_observation(
    scene: Scene,
    prior: list[tuple[int, int]],
    first_index: int,
    last_index: int,
    paths: list[list[tuple[int, int]] | None],
    scores: np.ndarray,
) -> np.ndarray:
    start, end = prior[first_index], prior[last_index]
    segment_cells = np.asarray(prior[first_index:last_index + 1], dtype=np.int32)
    rows, cols = segment_cells[:, 0], segment_cells[:, 1]
    ice = scene.ice_concentration[rows, cols]
    thick = scene.ice_thickness[rows, cols]
    risk = scene.chart_risk[rows, cols]
    speed = scene.current_speed[rows, cols]
    corridor = scene.ais_corridor[rows, cols]
    assist = [scene.current_assist(a, b) for a, b in zip(prior[first_index:last_index], prior[first_index + 1:last_index + 1])]
    prior_metrics = route_metrics(scene, prior[first_index:last_index + 1])
    distance = max(math.dist(start, scene.goal), 1.0)
    progress = (distance - math.dist(end, scene.goal)) / distance
    candidate_delta = np.zeros(6, dtype=np.float32)
    scale = max(abs(float(scores[0])), 1.0)
    for index, score in enumerate(scores):
        candidate_delta[index] = float(np.clip((scores[0] - score) / scale, -1.0, 1.0))
    features = [
        first_index / max(len(prior) - 1, 1),
        last_index / max(len(prior) - 1, 1),
        start[0] / max(scene.rows - 1, 1),
        start[1] / max(scene.cols - 1, 1),
        end[0] / max(scene.rows - 1, 1),
        end[1] / max(scene.cols - 1, 1),
        float(np.mean(ice)),
        float(np.mean(thick)) / 2.0,
        float(np.mean(risk)),
        float(np.mean(speed)),
        float(np.mean(assist)) if assist else 0.0,
        float(np.mean(corridor)),
        float(prior_metrics["path_length_km"]) / max(scene.cell_km * 12.0, 1.0),
        float(prior_metrics["total_cost"]) / 50.0,
        float(prior_metrics["turn_complexity_deg"]) / 180.0,
        math.dist(end, scene.goal) / max(math.hypot(scene.rows, scene.cols), 1.0),
        float(np.quantile(ice, 0.9)),
        float(np.quantile(risk, 0.9)),
        float(progress),
    ]
    return np.asarray([*features, *candidate_delta.tolist()], dtype=np.float32)


def build_decisions(
    scene: Scene,
    prior: list[tuple[int, int]],
    *,
    window: int = 12,
    bias_scale: float = 0.5,
) -> list[Decision]:
    if len(prior) < 2 or prior[0] != scene.start or prior[-1] != scene.goal:
        raise ValueError("The fixed S-IACO prior must connect the scene start and goal")
    decisions: list[Decision] = []
    first = 0
    while first < len(prior) - 1:
        last = min(first + max(1, int(window)), len(prior) - 1)
        prior_segment = list(prior[first:last + 1])
        paths = _candidate_paths(scene, prior_segment)
        valid = np.asarray([
            _is_safe_segment(scene, path, prior_segment[0], prior_segment[-1]) for path in paths
        ], dtype=np.bool_)
        valid[0] = True
        scores = np.asarray([
            _segment_score(scene, path) if path is not None else float("inf") for path in paths
        ], dtype=np.float32)
        finite = scores[np.isfinite(scores) & valid]
        spread = max(float(finite.std()), 1.0)
        center = float(finite.mean()) if len(finite) else float(scores[0])
        bias = np.zeros(6, dtype=np.float32)
        for index, score in enumerate(scores):
            if valid[index] and np.isfinite(score):
                bias[index] = float(np.clip((center - float(score)) / spread * bias_scale, -2.0, 2.0))
        observation = _decision_observation(scene, prior, first, last, paths, scores)
        decisions.append(Decision(observation, valid, bias, paths, scores, float(scores[0])))
        first = last
    return decisions


def _shadow_gate(scene: Scene, decision: Decision, proposed: int, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Keep the proposal when it remains hard-safe and rejoins its anchor."""
    path = decision.paths[proposed]
    if decision.mask[proposed] and _is_safe_segment(scene, path, start, goal):
        return proposed
    return 0


def _assemble_route(scene: Scene, decisions: list[Decision], actions: list[int]) -> list[tuple[int, int]]:
    route = [scene.start]
    for decision, action in zip(decisions, actions):
        segment = decision.paths[action]
        if segment is None or segment[0] != route[-1]:
            raise RuntimeError("Selected local segment does not connect to the route")
        route.extend(segment[1:])
    return route


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sag_ppo(
    scene: Scene,
    prior: list[tuple[int, int]],
    *,
    episodes: int = 300,
    seed: int = 0,
    device: str = "cpu",
    window: int = 12,
) -> tuple[list[tuple[int, int]], list[dict[str, float]], DiscretePPO]:
    """Train SAG-PPO on local edits of one fixed S-IACO global prior."""
    _seed_everything(seed)
    decisions = build_decisions(scene, prior, window=window)
    agent = DiscretePPO(device=device)
    history: list[dict[str, float]] = []
    for episode in range(1, max(1, int(episodes)) + 1):
        rollout: list[dict[str, Any]] = []
        actions: list[int] = []
        rewards: list[float] = []
        for index, decision in enumerate(decisions):
            proposed, logp, value = agent.act(decision.observation, decision.mask, decision.bias)
            prior_segment = decision.paths[0]
            assert prior_segment is not None
            executed = _shadow_gate(scene, decision, proposed, prior_segment[0], prior_segment[-1])
            executed_path = decision.paths[executed]
            assert executed_path is not None
            score = _segment_score(scene, executed_path)
            reward = float(np.clip((decision.prior_score - score) / max(abs(decision.prior_score), 1.0) * 10.0, -2.0, 2.0))
            actions.append(executed)
            rewards.append(reward)
            rollout.append({
                "observation": decision.observation,
                "mask": decision.mask,
                "bias": decision.bias,
                "action": proposed,
                "logp": logp,
                "value": value,
                "reward": reward,
                "done": index == len(decisions) - 1,
                # The actor receives credit only when the gate executed its proposal.
                "actor_include": float(proposed == executed),
            })
        losses = agent.update(rollout)
        route = _assemble_route(scene, decisions, actions)
        metrics = route_metrics(scene, route)
        history.append({
            "episode": float(episode),
            "reward": float(np.mean(rewards)) if rewards else 0.0,
            "route_total_cost": float(metrics["total_cost"]),
            "route_length_km": float(metrics["path_length_km"]),
            "route_ice_exposure_km": float(metrics["ice_exposure_km"]),
            "route_turn_complexity_deg": float(metrics["turn_complexity_deg"]),
            **losses,
        })

    final_actions: list[int] = []
    for decision in decisions:
        proposed, _, _ = agent.act(decision.observation, decision.mask, decision.bias, deterministic=True)
        prior_segment = decision.paths[0]
        assert prior_segment is not None
        final_actions.append(_shadow_gate(scene, decision, proposed, prior_segment[0], prior_segment[-1]))
    return _assemble_route(scene, decisions, final_actions), history, agent
