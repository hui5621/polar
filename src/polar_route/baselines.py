from __future__ import annotations

import numpy as np

from .config import PAPER
from .ppo import DiscretePPO
from .scene import Scene
from .search import astar, smooth_line_of_sight
from .siaco import ant_colony_route
from .sag_ppo import (
    _assemble_route,
    _gate,
    _guided_bias,
    _reward,
    build_decisions,
)


BASELINE_NAMES = ("astar", "ca_star", "los_astar", "aco", "zhou_iaco")
REFINEMENT_NAMES = ("random_matched", "greedy_matched", "bias_only", "ppo")
PAPER_METHOD_NAMES = {
    "astar": "A*", "ca_star": "CA*", "los_astar": "LosA*", "aco": "ACO",
    "zhou_iaco": "Zhou-IACO", "siaco": "S-IACO", "random_matched": "Random-matched",
    "greedy_matched": "Greedy-matched", "bias_only": "Bias-only", "ppo": "PPO", "sagppo": "SAG-PPO",
}


def run_baseline(
    scene: Scene,
    name: str,
    *,
    seed: int = 0,
    ants: int = 30,
    iterations: int = PAPER.siaco_iterations,
) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Run one graph-search or ant-colony reference planner."""
    if name == "astar":
        route = astar(scene, current_weight=0.0, ais_weight=0.0)
        return _require(route, name, scene), []
    if name == "ca_star":
        route = astar(scene, current_weight=1.0, ais_weight=0.0)
        return _require(route, name, scene), []
    if name == "los_astar":
        route = astar(scene, current_weight=0.0, ais_weight=0.0)
        return smooth_line_of_sight(scene, _require(route, name, scene)), []
    if name == "aco":
        return ant_colony_route(scene, mode="aco", ants=ants, iterations=iterations, seed=seed)
    if name == "zhou_iaco":
        return ant_colony_route(scene, mode="zhou_iaco", ants=ants, iterations=iterations, seed=seed)
    raise ValueError(f"Unknown baseline {name!r}; choose from {', '.join((*BASELINE_NAMES, 'siaco'))}")


def run_refinement_control(
    scene: Scene,
    prior: list[tuple[int, int]],
    name: str,
    *,
    seed: int = 0,
    episodes: int = PAPER.max_episodes,
    device: str = "cpu",
    max_decision_steps: int = PAPER.max_decision_steps,
) -> tuple[list[tuple[int, int]], list[dict[str, float]]]:
    """Run paper-matched candidate selectors and standard PPO."""
    if name not in REFINEMENT_NAMES:
        raise ValueError(f"Unknown refinement control {name!r}")
    decisions = build_decisions(
        scene, prior, max_decision_steps=max_decision_steps,
        apply_guidance_mask=name not in {"random_matched", "greedy_matched", "bias_only", "ppo"},
    )
    rng = np.random.default_rng(seed)
    agent = DiscretePPO(device=device) if name == "ppo" else None
    logs: list[dict[str, float]] = []
    episodes_run = max(1, int(episodes)) if name == "ppo" else 1
    final_actions: list[int] = []

    for episode in range(1, episodes_run + 1):
        previous_action, previous_accepted = 0, True
        selected_actions: list[int] = []
        rewards: list[float] = []
        rollout: list[dict] = []
        for step, decision in enumerate(decisions):
            observation = np.concatenate([
                decision.base_observation,
                np.asarray([np.clip(c.q_raw / PAPER.preview_scale, -3.0, 3.0) for c in decision.candidates], dtype=np.float32),
                np.asarray([2.0 * previous_action / max(PAPER.action_dim - 1, 1) - 1.0,
                            1.0 if previous_accepted else -1.0], dtype=np.float32),
            ])
            alpha = (episode - 1) / max(episodes_run - 1, 1)
            if name == "random_matched":
                proposal = int(rng.choice(np.flatnonzero(decision.mask)))
                logp = value = 0.0
                old_probs = np.zeros(PAPER.action_dim, dtype=np.float32)
            elif name == "greedy_matched":
                proposal = int(max(np.flatnonzero(decision.mask), key=lambda a: decision.candidates[int(a)].q_raw))
                logp = value = 0.0
                old_probs = np.zeros(PAPER.action_dim, dtype=np.float32)
            elif name == "bias_only":
                bias = _guided_bias(decision, 1.0, previous_action, previous_accepted)
                proposal = int(np.argmax(np.where(decision.mask, bias, -1e9)))
                logp = value = 0.0
                old_probs = np.zeros(PAPER.action_dim, dtype=np.float32)
            else:
                bias = np.zeros(PAPER.action_dim, dtype=np.float32)
                proposal, logp, value, old_probs = agent.act(observation, decision.mask, bias)

            # Random/Greedy/Bias-only use the same verifier. PPO is the
            # learning baseline without SAG-PPO's post-sampling gate.
            executed = _gate(decision, proposal, alpha) if name != "ppo" else proposal
            last_step = step == len(decisions) - 1
            reward = _reward(
                decision, executed, is_last=(last_step and decision.prior_path[-1] == scene.goal),
                is_timeout=(last_step and decision.prior_path[-1] != scene.goal),
            )
            selected_actions.append(executed)
            rewards.append(reward)
            if name == "ppo":
                rollout.append({
                    "observation": observation, "mask": decision.mask.copy(), "bias": bias,
                    "proposed_action": proposal, "executed_action": executed, "log_prob": logp,
                    "old_probs": old_probs, "value": value, "reward": reward,
                    "done": step == len(decisions) - 1, "actor_include": 1.0,
                })
            previous_action = executed
            previous_accepted = proposal == executed
        if name == "ppo":
            losses = agent.update(rollout, auxiliary=False)
            route = _assemble_route(scene, decisions, selected_actions)
            logs.append({"episode": float(episode), "reward": float(np.mean(rewards)), **losses})
            if episode == episodes_run:
                final_actions = selected_actions

    if name == "ppo":
        previous_action, previous_accepted = 0, True
        final_actions = []
        for decision in decisions:
            observation = np.concatenate([
                decision.base_observation,
                np.asarray([np.clip(c.q_raw / PAPER.preview_scale, -3.0, 3.0) for c in decision.candidates], dtype=np.float32),
                np.asarray([2.0 * previous_action / max(PAPER.action_dim - 1, 1) - 1.0,
                            1.0 if previous_accepted else -1.0], dtype=np.float32),
            ])
            proposal, _, _, _ = agent.act(
                observation, decision.mask, np.zeros(PAPER.action_dim, dtype=np.float32), deterministic=True
            )
            final_actions.append(proposal)
            previous_action, previous_accepted = proposal, True
    if name != "ppo":
        final_actions = selected_actions
    return _assemble_route(scene, decisions, final_actions), logs


def _require(route: list[tuple[int, int]] | None, name: str, scene: Scene) -> list[tuple[int, int]]:
    if route is None:
        raise RuntimeError(f"{name} did not find a connected route for {scene.date}")
    return route
