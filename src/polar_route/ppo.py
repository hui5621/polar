from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class DiscretePPO:
    """Small clipped PPO actor-critic for the six route-editing actions."""

    def __init__(
        self,
        obs_dim: int = 25,
        action_dim: int = 6,
        *,
        device: str = "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.20,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        update_epochs: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, action_dim)
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1)
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.update_epochs = int(update_epochs)

    def _distribution(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        logit_bias: np.ndarray,
    ) -> tuple[Categorical, torch.Tensor]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).reshape(1, -1)
        bias = torch.as_tensor(logit_bias, dtype=torch.float32, device=self.device).reshape(1, -1)
        logits = self.actor(obs) + bias
        logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        logit_bias: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        distribution, value = self._distribution(observation, action_mask, logit_bias)
        action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def update(self, rollout: list[dict[str, Any]]) -> dict[str, float]:
        if not rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        rewards = np.asarray([row["reward"] for row in rollout], dtype=np.float32)
        values = np.asarray([row["value"] for row in rollout] + [0.0], dtype=np.float32)
        dones = np.asarray([row["done"] for row in rollout], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            continuation = 1.0 - dones[index]
            delta = rewards[index] + self.gamma * values[index + 1] * continuation - values[index]
            gae = delta + self.gamma * self.gae_lambda * continuation * gae
            advantages[index] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.as_tensor(np.stack([row["observation"] for row in rollout]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([row["action"] for row in rollout], dtype=torch.int64, device=self.device)
        masks = torch.as_tensor(np.stack([row["mask"] for row in rollout]), dtype=torch.bool, device=self.device)
        biases = torch.as_tensor(np.stack([row["bias"] for row in rollout]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([row["logp"] for row in rollout], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        actor_include = torch.as_tensor([row["actor_include"] for row in rollout], dtype=torch.float32, device=self.device)
        actor_denom = actor_include.sum().clamp_min(1.0)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        for _ in range(self.update_epochs):
            logits = self.actor(obs) + biases
            logits = logits.masked_fill(~masks, -1e9)
            distribution = Categorical(logits=logits)
            logp = distribution.log_prob(actions)
            ratio = torch.exp(logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -(torch.minimum(unclipped, clipped) * actor_include).sum() / actor_denom
            entropy = (distribution.entropy() * actor_include).sum() / actor_denom
            actor_loss = policy_loss - self.entropy_coef * entropy
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            value = self.critic(obs).squeeze(-1)
            critic_loss = self.value_coef * torch.square(value - target).mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)
