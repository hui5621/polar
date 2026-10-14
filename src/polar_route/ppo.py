from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import PAPER, UNSPECIFIED


class DiscretePPO:
    """PPO Actor–Critic used by PPO and SAG-PPO.

    Actor and Critic are independent 25–48–48–6 and 25–48–48–1 networks.
    Guided distributions retain the exact action mask and logit bias used at
    sampling so the PPO likelihood ratio is evaluated on the proposed action.
    """

    def __init__(self, obs_dim: int = PAPER.state_dim, action_dim: int = PAPER.action_dim,
                 *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, action_dim)
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1)
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=PAPER.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=PAPER.critic_learning_rate)

    def _distribution(self, observation: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor) -> Categorical:
        logits = self.actor(observation) + bias
        logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, observation: np.ndarray, action_mask: np.ndarray, logit_bias: np.ndarray,
            *, deterministic: bool = False) -> tuple[int, float, float, np.ndarray]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).reshape(1, -1)
        bias = torch.as_tensor(logit_bias, dtype=torch.float32, device=self.device).reshape(1, -1)
        distribution = self._distribution(obs, mask, bias)
        action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(self.critic(obs).squeeze(-1).item()),
            distribution.probs.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def update(self, rollout: list[dict[str, Any]], *, auxiliary: bool = True) -> dict[str, float]:
        if not rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                    "bc_loss": 0.0, "counterfactual_loss": 0.0, "route_boundary_loss": 0.0,
                    "action_margin_loss": 0.0, "kl_loss": 0.0}

        rewards = np.asarray([row["reward"] for row in rollout], dtype=np.float32)
        values = np.asarray([row["value"] for row in rollout] + [0.0], dtype=np.float32)
        dones = np.asarray([row["done"] for row in rollout], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rollout))):
            keep_going = 1.0 - dones[index]
            delta = rewards[index] + PAPER.gamma * values[index + 1] * keep_going - values[index]
            gae = delta + PAPER.gamma * PAPER.gae_lambda * keep_going * gae
            advantages[index] = gae
        returns = advantages + values[:-1]
        actor_rows = np.asarray([r["actor_include"] > 0.0 for r in rollout], dtype=bool)
        if actor_rows.any():
            actor_mean = float(advantages[actor_rows].mean())
            actor_std = float(advantages[actor_rows].std())
            advantages = (advantages - actor_mean) / (actor_std + 1e-8)

        obs = torch.as_tensor(np.stack([r["observation"] for r in rollout]), dtype=torch.float32, device=self.device)
        proposed = torch.as_tensor([r["proposed_action"] for r in rollout], dtype=torch.int64, device=self.device)
        masks = torch.as_tensor(np.stack([r["mask"] for r in rollout]), dtype=torch.bool, device=self.device)
        biases = torch.as_tensor(np.stack([r["bias"] for r in rollout]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([r["log_prob"] for r in rollout], dtype=torch.float32, device=self.device)
        old_probs = torch.as_tensor(np.stack([r["old_probs"] for r in rollout]), dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        targets = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        include = torch.as_tensor([r["actor_include"] for r in rollout], dtype=torch.float32, device=self.device)
        included_count = include.sum().clamp_min(1.0)

        losses: dict[str, list[float]] = {k: [] for k in (
            "actor_loss", "critic_loss", "entropy", "bc_loss", "counterfactual_loss",
            "route_boundary_loss", "action_margin_loss", "kl_loss")}
        count = len(rollout)
        batch_size = PAPER.minibatch_size
        for _ in range(PAPER.ppo_epochs):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, batch_size):
                idx = order[start:start + batch_size]
                distribution = self._distribution(obs[idx], masks[idx], biases[idx])
                logp = distribution.log_prob(proposed[idx])
                ratio = torch.exp(logp - old_logp[idx])
                clipped = torch.clamp(ratio, 1.0 - PAPER.ppo_clip, 1.0 + PAPER.ppo_clip)
                actor_mask = include[idx]
                actor_denom = actor_mask.sum().clamp_min(1.0)
                policy_loss = -(torch.minimum(ratio * adv[idx], clipped * adv[idx]) * actor_mask).sum() / actor_denom
                entropy = (distribution.entropy() * actor_mask).sum() / actor_denom

                # Preview-derived auxiliary terms. The manuscript reports their
                # coefficients but does not give a separate closed form for
                # each term; these standard local preference losses use the
                # stored preview qualities and eligibility mask.
                q = obs[idx, PAPER.base_state_dim:PAPER.base_state_dim + PAPER.action_dim]
                valid = masks[idx]
                q_valid = q.masked_fill(~valid, -1e9)
                best = torch.argmax(q_valid, dim=-1)
                raw_logits = self.actor(obs[idx]) + biases[idx]
                raw_logits = raw_logits.masked_fill(~valid, -1e9)
                bc_rows = nn.functional.cross_entropy(raw_logits, best, reduction="none")
                bc_loss = (bc_rows * actor_mask).sum() / actor_denom

                pair_losses = []
                for a in range(PAPER.action_dim):
                    for b in range(a + 1, PAPER.action_dim):
                        pair_valid = valid[:, a] & valid[:, b] & (actor_mask > 0.0)
                        if not bool(pair_valid.any()):
                            continue
                        target = (q[pair_valid, a] > q[pair_valid, b]).float()
                        preferred_margin = raw_logits[pair_valid, a] - raw_logits[pair_valid, b]
                        signed_margin = torch.where(target > 0.5, preferred_margin, -preferred_margin)
                        pair_losses.append(nn.functional.softplus(-signed_margin).mean())
                cf_loss = torch.stack(pair_losses).mean() if pair_losses else policy_loss * 0.0

                # Rejoin-boundary regularization applies only once the prior
                # progress context reaches the paper's 0.70 threshold.
                late = (obs[idx, 8] >= PAPER.progress_threshold) & valid[:, 5] & (actor_mask > 0.0)
                if bool(late.any()):
                    route_loss = nn.functional.cross_entropy(raw_logits[late], torch.full_like(best[late], 5))
                else:
                    route_loss = policy_loss * 0.0
                if bool((actor_mask > 0.0).any()):
                    chosen_logit = raw_logits.gather(1, proposed[idx].unsqueeze(1)).squeeze(1)
                    best_logit = raw_logits.gather(1, best.unsqueeze(1)).squeeze(1)
                    margin_rows = torch.relu(
                        float(UNSPECIFIED["aux_action_margin_threshold"]) - (best_logit - chosen_logit)
                    ) * actor_mask
                    action_margin_loss = margin_rows.sum() / actor_denom
                else:
                    action_margin_loss = policy_loss * 0.0
                log_old_probs = torch.log(old_probs[idx].clamp_min(1e-30))
                kl_terms = torch.where(
                    masks[idx],
                    old_probs[idx] * (log_old_probs - distribution.logits.log_softmax(-1)),
                    torch.zeros_like(old_probs[idx]),
                )
                kl_rows = torch.sum(kl_terms, dim=-1)
                kl_loss = (kl_rows * actor_mask).sum() / actor_denom

                actor_loss = (
                    policy_loss - PAPER.entropy_coef * entropy
                    + (PAPER.aux_behavior_cloning * bc_loss if auxiliary else 0.0)
                    + (PAPER.aux_counterfactual * cf_loss if auxiliary else 0.0)
                    + (PAPER.aux_quality_boundary * policy_loss.detach() * 0.0 if auxiliary else 0.0)
                    + (PAPER.aux_route_boundary * route_loss if auxiliary else 0.0)
                    + (PAPER.aux_action_margin * action_margin_loss if auxiliary else 0.0)
                    + (PAPER.aux_kl * kl_loss if auxiliary else 0.0)
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), PAPER.max_grad_norm)
                self.actor_optimizer.step()

                value = self.critic(obs[idx]).squeeze(-1)
                critic_loss = PAPER.value_coef * torch.square(value - targets[idx]).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), PAPER.max_grad_norm)
                self.critic_optimizer.step()

                for name, tensor in (("actor_loss", actor_loss), ("critic_loss", critic_loss), ("entropy", entropy),
                                     ("bc_loss", bc_loss), ("counterfactual_loss", cf_loss),
                                     ("route_boundary_loss", route_loss), ("action_margin_loss", action_margin_loss),
                                     ("kl_loss", kl_loss)):
                    losses[name].append(float(tensor.detach().cpu()))
        return {name: float(np.mean(rows)) if rows else 0.0 for name, rows in losses.items()}

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
