"""Pointwise neural collaborative-filtering architectures."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, user_count: int, item_count: int, embedding_dim: int) -> None:
        super().__init__()
        _validate_dimensions(user_count, item_count, embedding_dim)
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.item_embedding = nn.Embedding(item_count, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        interaction = self.user_embedding(users) * self.item_embedding(items)
        return self.output(interaction).squeeze(1)


class MLP(nn.Module):
    def __init__(
        self,
        user_count: int,
        item_count: int,
        embedding_dim: int,
        hidden_layers: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dimensions(user_count, item_count, embedding_dim)
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.item_embedding = nn.Embedding(item_count, embedding_dim)
        self.layers, final_width = _mlp_layers(
            embedding_dim * 2, hidden_layers, dropout
        )
        self.output = nn.Linear(final_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.user_embedding(users), self.item_embedding(items)], dim=1
        )
        return self.output(self.layers(features)).squeeze(1)


class NeuMF(nn.Module):
    def __init__(
        self,
        user_count: int,
        item_count: int,
        embedding_dim: int,
        hidden_layers: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_dimensions(user_count, item_count, embedding_dim)
        self.gmf_user_embedding = nn.Embedding(user_count, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(item_count, embedding_dim)
        self.mlp_user_embedding = nn.Embedding(user_count, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(item_count, embedding_dim)
        self.mlp_layers, final_width = _mlp_layers(
            embedding_dim * 2, hidden_layers, dropout
        )
        self.output = nn.Linear(embedding_dim + final_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        ):
            nn.init.normal_(embedding.weight, std=0.01)
        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user_embedding(users) * self.gmf_item_embedding(items)
        mlp_input = torch.cat(
            [self.mlp_user_embedding(users), self.mlp_item_embedding(items)], dim=1
        )
        mlp = self.mlp_layers(mlp_input)
        return self.output(torch.cat([gmf, mlp], dim=1)).squeeze(1)


def _mlp_layers(
    input_width: int, hidden_layers: Sequence[int], dropout: float
) -> tuple[nn.Sequential, int]:
    if not hidden_layers or min(hidden_layers) < 1:
        raise ValueError("hidden_layers must contain positive widths")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    modules: list[nn.Module] = []
    width = input_width
    for next_width in hidden_layers:
        modules.extend([nn.Linear(width, next_width), nn.ReLU()])
        if dropout:
            modules.append(nn.Dropout(dropout))
        width = next_width
    return nn.Sequential(*modules), width


def _validate_dimensions(user_count: int, item_count: int, embedding_dim: int) -> None:
    if min(user_count, item_count, embedding_dim) < 1:
        raise ValueError("user_count, item_count, and embedding_dim must be positive")

