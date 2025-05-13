import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, ce_weight = 0.25):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        class_values = torch.tensor([-1.0, 0.0, 1.0], device=y_pred.device)
        y_pred_prob = torch.softmax(y_pred / self.temperature, dim=1) # predicted probabilites after softmax for each class
        errors = torch.abs(class_values.unsqueeze(0) - y_true.float().unsqueeze(1)) # 2 if pos / neg mistake, 1 if nut mistake 0 if correct
        # errors = torch.clamp(torch.abs(class_values.unsqueeze(0) - y_true.float().unsqueeze(1))*2.0 - 1.0, min=0.0) # 3 if pos / neg mistake, 1 if nut mistake 0 if correct
        per_sample_error = (y_pred_prob * errors).sum(dim=1)
        mae = per_sample_error.mean()
        loss = mae/2.0

        if self.ce_weight < 1:
            loss = (1.0 - self.ce_weight) * loss
        else:
            loss = 0

        if self.ce_weight == 0:
            ce_loss = 0
        else:
            ce_loss = nn.CrossEntropyLoss()(y_pred, (y_true+1).long()) * self.ce_weight #CE loss for hybrid loss
        # ce_loss = 0
        return loss + ce_loss

