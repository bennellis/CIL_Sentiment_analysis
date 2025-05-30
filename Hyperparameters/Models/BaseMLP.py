import numpy as np

from Hyperparameters.Models.BaseModel import BaseModel


class BaseMLP(BaseModel):
    def __init__(self, input_dim, output_dim=3, lr=0.001, hidden_dim1 = 128, hidden_dim2 = 64, temperature = 0.5, ce_weight = 0.25):
        super().__init__(lr=lr,temperature = temperature, ce_weight = ce_weight)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self._build_layers(input_dim, output_dim)
        self.optimizer = self._configure_optimizer()
        self.to(self.device)

    def _build_layers(self, input_dim, output_dim):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.layers(x)
