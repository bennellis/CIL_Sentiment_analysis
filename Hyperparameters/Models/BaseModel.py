
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

import mlflow
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from Hyperparameters.Models.CustomLoss import CustomLoss


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

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 5, log_mlflow: bool = False, plot_metrics: bool = True):
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

            if log_mlflow:
                mlflow.log_metric('train_loss', train_loss)
                mlflow.log_metric('train_acc', train_acc)
                mlflow.log_metric('train_neg_acc', train_neg_acc)
                mlflow.log_metric('train_pos_acc', train_pos_acc)
                mlflow.log_metric('train_nut_acc', train_nut_acc)
                if val_loader:
                    mlflow.log_metric('val_loss', val_loss)
                    mlflow.log_metric('val_acc', val_acc)
                    mlflow.log_metric('val_neg_acc', val_neg_acc)
                    mlflow.log_metric('val_pos_acc', val_pos_acc)
                    mlflow.log_metric('val_nut_acc', val_nut_acc)
            # Logging
            s = (f"Epoch {epoch + 1}/{epochs}: \n" +
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} " +
                f"Train Neg Acc: {train_neg_acc:.4f}, Nut Acc: {train_nut_acc:.4f}, Pos Acc: {train_pos_acc:.4f}" +
                (f"\n Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} " if val_loss else "") +
                (f"Val Neg Acc: {val_neg_acc:.4f}, Nut Acc: {val_nut_acc:.4f}, Pos Acc: {val_pos_acc:.4f}" if val_loss else ""))
            tqdm.write(s)
            logging.info(s)

        if plot_metrics:
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