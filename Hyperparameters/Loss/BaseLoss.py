from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLoss(ABC, nn.Module):
    @staticmethod
    def suggest_hyperparameters(trial):
        raise NotImplementedError

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass logic."""
        pass