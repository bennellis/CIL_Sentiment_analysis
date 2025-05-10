
from typing import Tuple, Optional, Dict, Any
from Hyperparameters.Models.BaseMLP import BaseModel

import torch
import torch.nn as nn
class LSTMClassifier(BaseModel):
    is_variable_length = True

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, lr: float = 0.001):
        super().__init__(lr=lr)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 3)
        self.optimizer = self._configure_optimizer()
        self.to(self.device)
        self._init_weights()

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x, lengths, y = batch
        return x, y, {'lengths': lengths}

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])