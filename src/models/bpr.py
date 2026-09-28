"""Bayesian Personalized Ranking matrix factorization."""

from __future__ import annotations

import torch
from torch import nn


class BPRMatrixFactorization(nn.Module):
    def __init__(self, user_count: int, item_count: int, embedding_dim: int) -> None:
        super().__init__()
        if min(user_count, item_count, embedding_dim) < 1:
            raise ValueError("user_count, item_count, and embedding_dim must be positive")
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.item_embedding = nn.Embedding(item_count, embedding_dim)
        self.item_bias = nn.Embedding(item_count, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = (self.user_embedding(users) * self.item_embedding(items)).sum(dim=1)
        return interaction + self.item_bias(items).squeeze(1)

    def pairwise_loss(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> torch.Tensor:
        difference = self(users, positive_items) - self(users, negative_items)
        return -torch.nn.functional.logsigmoid(difference).mean()

