from typing import Tuple, Optional, Dict, Any
from Hyperparameters.Models.BaseMLP import BaseModel

import torch
import torch.nn as nn

class BiRNNClassifier(BaseModel):
    """RNN classification head to be used on top of a bert-like encoder"""
    is_variable_length = True

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3,
                 dropout: float = 0.2, lr: float = 0.001):
        super().__init__(lr=lr)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 3)  # Multiply hidden size by 2 for bidirectional
        self.optimizer = self._configure_optimizer()
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x, lengths, y = batch
        return x, y, {'lengths': lengths}

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, hidden = self.rnn(x)  # hidden shape: (num_layers * num_directions, batch, hidden_dim)

        if isinstance(output, nn.utils.rnn.PackedSequence):
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Concatenate last hidden state from both directions
        if isinstance(hidden, tuple):  # If LSTM
            hidden = hidden[0]
        last_forward = hidden[-2]
        last_backward = hidden[-1]
        final_hidden = torch.cat((last_forward, last_backward), dim=1)

        return self.fc(final_hidden)