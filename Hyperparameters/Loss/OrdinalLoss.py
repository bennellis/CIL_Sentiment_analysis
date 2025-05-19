import torch
import torch.nn as nn
from Hyperparameters.Loss.BaseLoss import BaseLoss
from Hyperparameters.Utils.Misc import get_device, suggest_namespaced_params
from Hyperparameters.registry import register_criterion


@register_criterion(name='OrdinalLoss')
class OrdinalLoss(BaseLoss):
    def __init__(
        self,
        temperature: float = 0.5,
        ce_weight: float = 0.25,
        use_mae: bool = False,
        use_cdw_ce: bool = True,
        margin: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.use_mae = use_mae
        self.use_cdw_ce = use_cdw_ce
        self.margin = margin

        self.device = get_device()

        # Class values correspond to [-1, 0, 1] → useful for ordinal distance
        self.class_values = torch.tensor([-1.0, 0.0, 1.0], device=self.device)

        # Penalty matrix: higher penalties for semantically distant misclassifications
        self.penalty_matrix = torch.tensor([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0]
        ], device=self.device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs) -> torch.Tensor:
        y_pred_prob = torch.softmax(y_pred / self.temperature, dim=1)
        target_indices = (y_true + 1).long()  # Map {-1,0,1} → {0,1,2}

        loss = 0.0

        # -------------------------------------
        # (1) Ordinal-Aware MAE loss (optional)
        # -------------------------------------
        if self.use_mae:
            errors = torch.abs(self.class_values.unsqueeze(0) - y_true.unsqueeze(1).float())  # shape: [B, 3]
            mae_per_sample = (y_pred_prob * errors).sum(dim=1)  # expected error
            mae_loss = mae_per_sample.mean() / 2.0  # normalize to match 0-1 range
            loss += (1.0 - self.ce_weight) * mae_loss if self.ce_weight < 1 else 0.0

        # -------------------------------------
        # (2) Custom Distance-Weighted CE
        # -------------------------------------
        if self.use_cdw_ce:
            penalties = self.penalty_matrix[target_indices]  # [B, 3]
            prob_m = y_pred_prob + self.margin
            clipped = torch.clamp(prob_m, max=1.0)
            log_probs = torch.log(1.0 - clipped + 1e-9)
            weighted_log_probs = penalties * log_probs
            cdw_ce_loss = -weighted_log_probs.sum(dim=1).mean()
            loss += cdw_ce_loss

        # -------------------------------------
        # (3) Optional CE Loss
        # -------------------------------------
        if self.ce_weight > 0:
            ce_loss = nn.CrossEntropyLoss()(y_pred, target_indices)
            loss += self.ce_weight * ce_loss

        return loss

    @staticmethod
    def suggest_hyperparameters(trial):
        param_defs = {
            "margin": lambda t, n: t.suggest_float(n, 0.1, 0.5),
            "temperature": lambda t, n: t.suggest_float(n, 0.5, 1.0),
            "ce_weight": lambda t, n: t.suggest_float(n, 0.0, 0.7),
        }
        return suggest_namespaced_params(trial, "OrdinalLoss", param_defs)