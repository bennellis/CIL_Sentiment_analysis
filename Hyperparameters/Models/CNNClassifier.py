from typing import Tuple, Optional, Dict, Any
from Hyperparameters.Models.BaseMLP import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(BaseModel):
    """CNN Classification head to be used on top of a bert-like encoder"""
    is_variable_length = True

    def __init__(self, input_dim: int, num_classes: int = 3,
                 kernel_sizes: list = [2,4,6,8], num_filters: int = 128,
                 dropout: float = 0.5, lr: float = 0.001):
        super().__init__(lr=lr)
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, input_dim))
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.optimizer = self._configure_optimizer()
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x, lengths, y = batch
        return x, y, {}

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.unsqueeze(1)  # shape: (batch_size, 1, seq_len, input_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(B, num_filters, seq_len-k+1), ...]
        x = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in x]  # [(B, num_filters), ...]
        x = torch.cat(x, 1)  # shape: (B, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        return self.fc(x)