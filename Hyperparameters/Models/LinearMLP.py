from Hyperparameters.Models.BaseMLP import BaseMLP

import torch.nn as nn

class LinearMLP(BaseMLP):
    def _build_layers(self, input_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.Linear(self.hidden_dim2, output_dim)
        )