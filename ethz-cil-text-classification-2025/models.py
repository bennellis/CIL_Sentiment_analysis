from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from transformers import BertTokenizer, BertModel, pipeline, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoConfig, TrainingArguments, Trainer, AutoModel, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm.auto import tqdm

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Any
import custom_dataloader
import importlib
import logging

# Set up basic config for logging
logging.basicConfig(
    filename="logs/training.log",        # Log file name
    filemode="a",                   # Append mode
    level=logging.INFO,             # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format
    datefmt="%Y-%m-%d %H:%M:%S"     # Timestamp format
)




class BaseModel(ABC, nn.Module):
    """Abstract base class for all PyTorch models."""
    use_dataloader = True

    def __init__(self, lr: float, temperature:float = 0.5, ce_weight:float = 0.25):
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.criterion = self._configure_criterion()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass logic."""
        pass

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure the optimizer (override in subclasses if needed)."""
        return optim.Adam(self.parameters(), lr=self.lr)

    def _configure_criterion(self) -> nn.Module:
        """Configure the loss function (override if needed)."""
        return CustomLoss(temperature = self.temperature, ce_weight = self.ce_weight)

    def _configure_scheduler(self, optimizer: optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Unpack a batch into inputs, targets, and extra arguments."""
        x, y = batch[0], batch[1]
        return x, y, {}

    def _adjust_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to target labels (e.g., 0,1,2 → -1,0,1)."""
        return predictions - 1

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 5):
        """Train the model with optional validation."""
        tqdm.write(f'Training {self.__class__.__name__} on {self.device}')
        logging.info(f'Training {self.__class__.__name__} on {self.device}')
        train_losses, train_accs, train_neg_accs, train_pos_accs, train_nut_accs = [], [], [], [], []
        val_losses, val_accs, val_neg_accs, val_pos_accs, val_nut_accs = [], [], [], [], []
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        self.scheduler = self._configure_scheduler(self.optimizer,warmup_steps,total_steps)

        for epoch in range(epochs):
            # Training
            self.train()
            train_loss, train_acc, train_neg_acc, train_nut_acc, train_pos_acc = self._run_epoch(train_loader, training=True)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_neg_accs.append(train_neg_acc)
            train_nut_accs.append(train_nut_acc)
            train_pos_accs.append(train_pos_acc)

            # Validation
            val_loss, val_acc, val_neg_acc, val_nut_acc, val_pos_acc = None, None, None, None, None
            if val_loader:
                val_loss, val_acc, val_neg_acc, val_nut_acc, val_pos_acc = self.evaluate(val_loader)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_neg_accs.append(val_neg_acc)
                val_nut_accs.append(val_nut_acc)
                val_pos_accs.append(val_pos_acc)

            # Logging
            s = (f"Epoch {epoch + 1}/{epochs}: \n" +
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} " +
                f"Train Neg Acc: {train_neg_acc:.4f}, Nut Acc: {train_nut_acc:.4f}, Pos Acc: {train_pos_acc:.4f}" +
                (f"\n Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} " if val_loss else "") +
                (f"Val Neg Acc: {val_neg_acc:.4f}, Nut Acc: {val_nut_acc:.4f}, Pos Acc: {val_pos_acc:.4f}" if val_loss else ""))
            tqdm.write(s)
            logging.info(s)

        self.plot_metrics(train_losses, train_accs, val_losses, val_accs)

    def _run_epoch(self, data_loader: DataLoader, training: bool = True) -> Tuple[float, float, float, float, float]:
        """Run one epoch (training or evaluation)."""
        total_loss, total_correct, total_samples = 0.0, 0, 0
        total_neg, total_neg_correct = 0,0
        total_nut, total_nut_correct = 0, 0
        total_pos, total_pos_correct = 0, 0
        pbar = tqdm(data_loader, desc=f"{'Training' if training else 'Evaluating'}",
                    unit='batch', leave=False)

        for batch in pbar:
            x, y, kwargs = self._unpack_batch(batch)
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            outputs = self(x, **kwargs)
            loss = self.criterion(outputs, y)
            # print(outputs)

            # Backward pass (if training)
            if training:
                self.optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Metrics
            preds = outputs.argmax(dim=1)
            adjusted_preds = self._adjust_predictions(preds)
            # print(adjusted_preds)
            # print(y)
            total_neg_correct += ((adjusted_preds==y) & (y == -1)).sum().item()
            total_neg += (adjusted_preds==-1).sum().item()
            total_nut_correct += ((adjusted_preds == y) & (y == 0)).sum().item()
            total_nut += (adjusted_preds == 0).sum().item()
            total_pos_correct += ((adjusted_preds == y) & (y == 1)).sum().item()
            total_pos += (adjusted_preds == 1).sum().item()
            total_correct += (adjusted_preds == y).sum().item()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            pbar.set_postfix({
                'samples': total_samples,
                'loss': f"{total_loss / total_samples:.4f}",
                'acc': f"{total_correct / total_samples:.4f}",
                'neg': f"{total_neg_correct / total_neg if total_neg else -1:.4f}",
                'nut': f"{total_nut_correct / total_nut if total_nut else -1:.4f}",
                'pos': f"{total_pos_correct / total_pos if total_pos else -1:.4f}",
            })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        avg_neg_acc = total_neg_correct / total_neg if total_neg else -1
        avg_nut_acc = total_nut_correct / total_nut if total_nut else -1
        avg_pos_acc = total_pos_correct / total_pos if total_pos else -1
        return avg_loss, avg_acc, avg_neg_acc, avg_nut_acc, avg_pos_acc

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float,float,float,float]:
        """Evaluate the model on a data loader."""
        self.eval()
        with torch.no_grad():
            return self._run_epoch(data_loader, training=False)

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """Generate predictions."""
        self.eval()
        all_preds = []
        pbar = tqdm(data_loader, desc="Predicting",unit='batch', leave=False)
        with torch.no_grad():
            for batch in pbar:
                x, _, kwargs = self._unpack_batch(batch)
                x = x.to(self.device)
                outputs = self(x, **kwargs)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
        return self._adjust_predictions(torch.cat(all_preds))

    def plot_metrics(self, train_losses, train_accuracies, val_losses=None, val_accuracies=None):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Val Loss')
        plt.title('Loss over epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Acc')
        if val_accuracies:
            plt.plot(val_accuracies, label='Val Acc')
        plt.title('Accuracy over epochs')
        plt.legend()

        plt.show()

# class Logistic_Regression_Baseline(BaseModel):
#     is_variable_length=False
#
#     def __init__(self, C=1.0, max_iter=100):
#         self.model = LogisticRegression(
#             C=C, max_iter=max_iter
#         )
#         self._is_fitted = False
#
#     def fit(self, train_loader):
#         self._is_fitted = True
#         x_train = []
#         y_train = []
#         for x_batch, y_batch in train_loader:
#             x_train.append(x_batch)
#             y_train.append(y_batch)
#         return self.model.fit(x_train, y_train)
#
#     def predict(self,X):
#         if not self._is_fitted:
#             raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
#         return self.model.predict(X)
#
#     def coef(self):
#         if not self._is_fitted:
#             raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
#         return self.model.coef_
    


# *********************** PYTORCH MODELS *****************************
class CustomLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, ce_weight = 0.25):
        super().__init__()
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_values = torch.tensor([-1.0, 0.0, 1.0], device=self.device)
        self.penalty_matrix = torch.tensor([
            [0.0, 1.0, 2.0],  # true class = -1 → index 0
            [1.0, 0.0, 1.0],  # true class =  0 → index 1
            [2.0, 1.0, 0.0]  # true class =  1 → index 2
        ], device = self.device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # class_values = torch.tensor([-1.0, 0.0, 1.0], device=y_pred.device)
        y_pred_prob = torch.softmax(y_pred / self.temperature, dim=1) # predicted probabilites after softmax for each class
        errors = torch.abs(self.class_values.unsqueeze(0) - y_true.float().unsqueeze(1)) # 2 if pos / neg mistake, 1 if nut mistake 0 if correct
        # errors = torch.clamp(torch.abs(class_values.unsqueeze(0) - y_true.float().unsqueeze(1))*2.0 - 1.0, min=0.0) # 3 if pos / neg mistake, 1 if nut mistake 0 if correct
        # print(y_pred_prob)
        # print(f"errors: {errors}")
        per_sample_error = (y_pred_prob * errors).sum(dim=1)
        # print(per_sample_error)
        mae = per_sample_error.mean()
        loss = mae/2.0

        target_indices = y_true + 1
        # pred_indices = torch.argmax(y_pred, dim=1)
        # # print(target_indices)
        # # print(pred_indices)
        distances = self.penalty_matrix[target_indices]
        margin = 0.1
        prob_m = y_pred_prob+margin

        log_probs = torch.log(1 - torch.min(prob_m, torch.ones_like(prob_m)) + 1e-9)
        weighted_log_probs = distances * log_probs
        cdw_ce_loss = -weighted_log_probs.sum(dim=1).mean()

        # # print(penalties)
        # print(y_pred_prob.shape)
        # ce_loss_w = F.cross_entropy(y_pred, target_indices, reduction='none')
        # # print(ce_loss_w)
        #
        # # Apply penalty as weight
        # weighted_loss = ce_loss_w * penalties
        # print(weighted_loss)
        # print(weighted_loss.mean())
        # print(ce_loss_w.mean())

        if self.ce_weight < 1:
            loss = (1.0 - self.ce_weight) * loss
        else:
            loss = 0

        if self.ce_weight == 0:
            ce_loss = 0
        else:
            ce_loss = nn.CrossEntropyLoss()(y_pred, (y_true+1).long()) * self.ce_weight #CE loss for hybrid loss
        # ce_loss = 0
        # return loss + ce_loss + cdw_ce_loss
        return cdw_ce_loss
        # return weighted_loss.mean()
        # return loss

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


class LinearMLP(BaseMLP):
    def _build_layers(self, input_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.Linear(self.hidden_dim2, output_dim)
        )

class NonLinearMLP(BaseMLP):
    def _build_layers(self, input_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, output_dim)
        )

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
    

# ******************************** Variable length models ***********************************
class LSTMClassifier(BaseModel):
    is_variable_length = True

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, lr: float = 0.001):
        super().__init__(lr=lr)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 3)
        self.optimizer = self._configure_optimizer()
        self.to(self.device)
        self._init_weights()

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x, lengths, y = batch
        return x, y, {'lengths': lengths}

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

class BertPreTrainedClassifier(BaseModel):
    is_variable_length = True
    def __init__(self, model_name, input_dim: int = None, lr: float = 0.00001, pt_lr_top: float = 1e-5,
                 pt_lr_mid: float = 1e-6, pt_lr_bot: float = 1e-7,
                 frozen = False, class_order = [2,0,1], dropout=0.1,
                 temperature = 0.5, ce_weight = 0.25, custom_ll = False):
        super().__init__(lr=lr, temperature=temperature, ce_weight=ce_weight)
        self.lr = lr
        config = AutoConfig.from_pretrained(model_name)
        # config.hidden_dropout_prob = dropout  # default is 0.1
        # config.attention_probs_dropout_prob = dropout  # default is 0.1
        config.num_labels = 3
        self.custom_ll = custom_ll
        self.pt_lr_top = pt_lr_top
        self.pt_lr_mid = pt_lr_mid
        self.pt_lr_bot = pt_lr_bot
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.custom_ll:
            self.model = AutoModel.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True,
                config=config
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(config.hidden_size, 512),  # BERT's hidden size -> 512
                nn.GELU(),  # or nn.ReLU()
                nn.Dropout(dropout),
                nn.Linear(512, config.num_labels)
            )
            self.classifier.to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True,
                config=config
            )
        self.model.to(self.device)
        self.optimizer = self._configure_optimizer()
        self.to(self.device)
        self.frozen = frozen
        self.class_order = class_order

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with attention mask handling"""
        attention_mask = kwargs.get('attention_mask', None)
        # print(type(attention_mask))
        # print(type(x))
        # print(x.shape)
        if self.frozen:
            if self.custom_ll:
                logits = self.classifier(x.float())
            else:
                logits = self.model.classifier(x.float())
        else:
            if self.custom_ll:
                outputs = self.model(
                    input_ids=x,
                    attention_mask=attention_mask
                )
                pooled_output = outputs.last_hidden_state[:, 0, :]
                # pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
                logits = self.classifier(pooled_output)
            else:
                logits = self.model(
                    input_ids=x,
                    attention_mask=attention_mask
                ).logits
        # print(res.shape)
        # From tests, the new "neutral" head was the first location, then positive, then negative.
        # so we need to reshape the output to be negative, neutral, positive, which is what happens below.
        return logits[:, self.class_order]

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """AdamW optimizer with weight decay"""
        # layers = self.model.transformer.layer #transformer or bert
        for attr in ("encoder", "transformer", "layers"):  # This is to set the tokenizer correctly for different model architectures.
            backbone = getattr(self.model, attr, None)
            if backbone is not None:
                if attr in ("encoder","transformer"):
                    backbone = backbone.layer
                break
        # print(self.model)
        # print(self.model.layers)
        layers = backbone
        num_layers = len(layers)
        third = num_layers // 3

        # Split into bottom, mid, top
        bottom_layers = layers[:third]
        mid_layers = layers[third:2 * third]
        top_layers = layers[2 * third:]
        return optim.AdamW(
            [
                {'params': self.classifier.parameters(), 'lr': self.lr},
                # {'params': self.model.parameters(), 'lr': self.pt_lr},
                {'params': bottom_layers.parameters(), 'lr': self.pt_lr_bot},  # Bottom layers
                {'params': mid_layers.parameters(), 'lr': self.pt_lr_mid},  # Mid layers
                {'params': top_layers.parameters(), 'lr': self.pt_lr_top},  #Top Layers
            ],
            weight_decay=0.01
        )

    # def _configure_criterion(self) -> nn.Module:
    #     """Cross entropy loss"""
    #     return nn.CrossEntropyLoss()

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Unpack HF-formatted batch"""

        if self.frozen: #If we've pre-computed the forward pass through the model, we just need to train the MLP
            return (
                batch[0],
                batch[2],
                {'attention_mask': None}
            )
        else: #If we're doing the full pass, and inputs are tokens with an attention mask
            return (
                batch[0][:, 0].long(),
                batch[2],
                {'attention_mask': batch[0][:, 1].long().to(self.device)}
            )
            # print(batch)

    # def _adjust_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
    #     """No adjustment needed for 0/1/2 labels"""
    #     predictions = torch.where(predictions == 2, torch.tensor(-1, device=predictions.device), predictions)
    #     return predictions
