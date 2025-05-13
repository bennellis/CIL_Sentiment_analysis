import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, ce_weight = 0.25, margin = 0.1, use_cdw = False):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.margin = margin
        self.use_cdw = use_cdw
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_values = torch.tensor([-1.0, 0.0, 1.0], device=self.device)
        self.penalty_matrix = torch.tensor([
            [0.0, 1.0, 2.0],  # true class = -1 → index 0
            [1.0, 0.0, 1.0],  # true class =  0 → index 1
            [2.0, 1.0, 0.0]  # true class =  1 → index 2
        ], device=self.device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_prob = torch.softmax(y_pred / self.temperature,
                                    dim=1)  # predicted probabilites after softmax for each class
        cdw_ce_loss = 0
        if self.use_cdw:
            distances = self.penalty_matrix[y_true+1]
            prob_m = y_pred_prob + self.margin

            log_probs = torch.log(1 - torch.min(prob_m, torch.ones_like(prob_m)) + 1e-9)
            weighted_log_probs = distances * log_probs
            cdw_ce_loss = -weighted_log_probs.sum(dim=1).mean()
            return cdw_ce_loss

        errors = torch.abs(self.class_values.unsqueeze(0) - y_true.float().unsqueeze(1)) # 2 if pos / neg mistake, 1 if nut mistake 0 if correct
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

