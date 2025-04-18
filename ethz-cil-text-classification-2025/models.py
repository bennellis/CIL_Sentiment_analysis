from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Any


class BaseModel(ABC, nn.Module):
    """Abstract base class for all PyTorch models."""
    use_dataloader = True

    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
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
        return CustomLoss()

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Unpack a batch into inputs, targets, and extra arguments."""
        x, y = batch[0], batch[1]
        return x, y, {}

    def _adjust_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to target labels (e.g., 0,1,2 → -1,0,1)."""
        return predictions - 1

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 5):
        """Train the model with optional validation."""
        print(f'Training {self.__class__.__name__} on {self.device}')
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        for epoch in range(epochs):
            # Training
            self.train()
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            val_loss, val_acc = None, None
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

            # Logging
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}" +
                  (f" | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}" if val_loss else ""))

        self.plot_metrics(train_losses, train_accs, val_losses, val_accs)

    def _run_epoch(self, data_loader: DataLoader, training: bool = True) -> Tuple[float, float]:
        """Run one epoch (training or evaluation)."""
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in data_loader:
            x, y, kwargs = self._unpack_batch(batch)
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            outputs = self(x, **kwargs)
            loss = self.criterion(outputs, y)

            # Backward pass (if training)
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Metrics
            preds = outputs.argmax(dim=1)
            adjusted_preds = self._adjust_predictions(preds)
            total_correct += (adjusted_preds == y).sum().item()
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on a data loader."""
        self.eval()
        with torch.no_grad():
            return self._run_epoch(data_loader, training=False)

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """Generate predictions."""
        self.eval()
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
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
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        class_values = torch.tensor([-1.0, 0.0, 1.0], device=y_pred.device)
        y_pred_prob = torch.softmax(y_pred / self.temperature, dim=1)
        y_pred_expected = (y_pred_prob * class_values).sum(dim=1)
        mae = torch.abs(y_pred_expected - y_true.float()).mean()
        loss = mae/2

        # ce_loss = nn.CrossEntropyLoss()(y_pred, (y_true+1).long()) #CE loss for hybrid loss
        ce_loss = 0
        return loss + ce_loss




class BaseMLP(BaseModel):
    def __init__(self, input_dim, output_dim=3, lr=0.001):
        super().__init__(lr=lr)
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
            nn.Linear(input_dim, 128),
            nn.Linear(128, 64),
            nn.Linear(64, output_dim)
        )

class NonLinearMLP(BaseMLP):
    def _build_layers(self, input_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

class DropoutMLP(BaseMLP):
    def __init__(self, *args, dropout_prob=0.2, **kwargs):
        self.dropout_prob = dropout_prob
        super().__init__(*args, **kwargs)

    def _build_layers(self, input_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
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

