import torch
import torch.nn as nn
from Hyperparameters.Loss.BaseLoss import BaseLoss
from Hyperparameters.Utils.Misc import suggest_namespaced_params
from Hyperparameters.registry import register_criterion


@register_criterion('CustomLoss')
class CustomLoss(BaseLoss):
    def __init__(self,
                 temperature: float = 0.5,
                 ce_weight=0.25,
                 **kwargs
                 ):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        class_values = torch.tensor([-1.0, 0.0, 1.0], device=y_pred.device)
        y_pred_prob = torch.softmax(y_pred / self.temperature,
                                    dim=1)
        errors = torch.abs(class_values.unsqueeze(0) - y_true.float().unsqueeze(
            1))
        per_sample_error = (y_pred_prob * errors).sum(dim=1)
        mae = per_sample_error.mean()
        loss = mae / 2.0

        if self.ce_weight < 1:
            loss = (1.0 - self.ce_weight) * loss
        else:
            loss = 0

        if self.ce_weight == 0:
            ce_loss = 0
        else:
            ce_loss = nn.CrossEntropyLoss()(y_pred, (y_true + 1).long()) * self.ce_weight  #CE loss for hybrid loss
        # ce_loss = 0
        return loss + ce_loss

    @staticmethod
    def suggest_hyperparameters(trial):
        param_defs = {
            "temperature": lambda t, n: t.suggest_float(n, 0.5, 1.0),
            "ce_weight": lambda t, n: t.suggest_float(n, 0.0, 0.7),
        }
        return suggest_namespaced_params(trial, "CustomLoss", param_defs)