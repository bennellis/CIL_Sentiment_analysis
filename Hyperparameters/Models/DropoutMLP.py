from Hyperparameters.Models.BaseMLP import BaseMLP

import torch.nn as nn
class DropoutMLP(BaseMLP):
    def __init__(self, *args, dropout_prob=0.2, **kwargs):
        self.dropout_prob = dropout_prob
        super().__init__(*args, **kwargs)

    def _build_layers(self, input_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, output_dim)
        )
