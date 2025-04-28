import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters.registry import register_model


@register_model("SimpleModel")
class ModelDummy(nn.Module):
    def __init__(self, dropout=0.0):
        super(ModelDummy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    @staticmethod
    def suggest_hyperparams(trial):
        from Hyperparameters.Utils.Misc import suggest_namespaced_params

        param_defs = {
            "dropout": lambda t, n: t.suggest_float(n, 0.1, 0.9)
        }

        return suggest_namespaced_params(trial, "ModelDummy", param_defs)

