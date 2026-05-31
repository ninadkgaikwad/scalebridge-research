from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 10


class SupervisedTrainer:
    """Reusable PyTorch supervised trainer for P1 black-box baselines."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainerConfig | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for _epoch in range(self.config.epochs):
            train_loss = self._train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)
            if val_loader is not None:
                val_loss = self.evaluate_loss(val_loader)
                history["val_loss"].append(val_loss)
        return history

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        return total_loss / max(n_batches, 1)
