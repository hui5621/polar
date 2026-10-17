from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .config import CONFIG


class SAGPPOPolicy:
    """Actor-critic policy used for SAG-PPO route refinement."""

    def __init__(self, obs_dim: int = CONFIG.state_dim, action_dim: int = CONFIG.action_dim,
                 *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, action_dim)
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1)
        ).to(self.device)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            if "critic" in checkpoint:
                self.critic.load_state_dict(checkpoint["critic"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.actor.load_state_dict(checkpoint["state_dict"])
        else:
            self.actor.load_state_dict(checkpoint)
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray, action_mask: np.ndarray, logit_bias: np.ndarray,
            *, deterministic: bool = True) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).reshape(1, -1)
        bias = torch.as_tensor(logit_bias, dtype=torch.float32, device=self.device).reshape(1, -1)
        logits = (self.actor(obs) + bias).masked_fill(~mask, float("-inf"))
        distribution = Categorical(logits=logits)
        action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return int(action.item())

