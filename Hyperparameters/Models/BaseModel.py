
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import os

import mlflow
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_absolute_error

from Hyperparameters.Models.CustomLoss import CustomLoss


class BaseModel(ABC, nn.Module):
    """Abstract base class for all PyTorch models."""
    use_dataloader = True

    def __init__(self, lr: float, temperature:float = 0.5, ce_weight:float = 0.25, margin:float = 0.1, use_cdw:bool = False):
        super().__init__()
        self.lr = lr
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.margin = margin
        self.use_cdw = use_cdw
        self.criterion = self._configure_criterion()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.best_score = 0
        self.best_loss = 100

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass logic."""
        pass

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure the optimizer (override in subclasses if needed)."""
        return optim.Adam(self.parameters(), lr=self.lr)

    def _configure_criterion(self) -> nn.Module:
        """Configure the loss function (override if needed)."""
        return CustomLoss(temperature = self.temperature,
                          ce_weight = self.ce_weight,
                          margin = self.margin,
                          use_cdw = self.use_cdw)

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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        log_mlflow: bool = False,
        plot_metrics: bool = True,
        validations_per_epoch: int = 1,
        initial_steps: int = 0,
        early_save = False,
    ):
        """Train the model with optional validation."""
        tqdm.write(f'Training {self.__class__.__name__} on {self.device}')
        logging.info(f'Training {self.__class__.__name__} on {self.device}')
        steps_per_epoch = len(train_loader)
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        self.scheduler = self._configure_scheduler(self.optimizer,warmup_steps,total_steps)

        for epoch in range(epochs):
            s = f"Epoch {epoch + 1}/{epochs}: \n"
            tqdm.write(s)
            logging.info(s)

            # Training
            self.train()
            train_loss, train_acc, mae, train_neg_acc, train_nut_acc, train_pos_acc = self._run_epoch(
                train_loader,
                val_loader,
                validations_per_epoch = validations_per_epoch,
                log_mlflow=log_mlflow,
                training=True,
                steps= initial_steps + steps_per_epoch*epoch,
                early_save = early_save
            )

            if log_mlflow:
                self.log_to_mlflow(True, train_loss, train_acc, mae, train_neg_acc, train_nut_acc,
                                   train_pos_acc, initial_steps + steps_per_epoch*(epoch+1))

            # Logging
            s = (f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mae: {mae:.4f}, Lscore: {1-mae*0.5:.4f}, " +
                f"Train Neg Acc: {train_neg_acc:.4f}, Nut Acc: {train_nut_acc:.4f}, Pos Acc: {train_pos_acc:.4f}")
            tqdm.write(s)
            logging.info(s)

    def _run_epoch(
            self,
            data_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            validations_per_epoch:int = 1,
            log_mlflow:bool = False,
            training: bool = True,
            steps: int = 0,
            early_save:bool = False,
    ) -> Tuple[float, float, float, float, float, float]:
        """Run one epoch (training or evaluation)."""
        total_loss, total_correct, total_samples = 0.0, 0, 0
        total_neg, total_neg_correct = 0,0
        total_nut, total_nut_correct = 0, 0
        total_pos, total_pos_correct = 0, 0
        total_absolute_error = 0
        validate_every = max(1, len(data_loader) // 10)
        count = 0
        validation_strs = ''
        # if training:
        with tqdm(data_loader, desc=f"{'Training' if training else 'Evaluating'}",
                    unit='batch', leave=False, position = 0) as pbar:
        # else:
        #     pbar = data_loader

            for batch in pbar:
                count+=1
                x, y, kwargs = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self(x, **kwargs)
                # print(x.shape)
                # print(outputs.shape)
                loss = self.criterion(outputs, y)
                # print(outputs)

                # Backward pass (if training)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    if val_loader and count%validate_every == 0:
                        pbar.clear()
                        validation_strs = (self.validate(val_loader,log_mlflow=log_mlflow, steps = steps+count, early_save = early_save))


                # Metrics
                preds = outputs.argmax(dim=1)
                adjusted_preds = self._adjust_predictions(preds) #TODO: add mae loss and LScore
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
                total_absolute_error += abs(adjusted_preds - y).sum().item()
                pbar.set_postfix({
                    'samples': total_samples,
                    'loss': f"{total_loss / total_samples:.4f}",
                    'acc': f"{total_correct / total_samples:.4f}",
                    'mae': f"{total_absolute_error / total_samples:.4f}",
                    'neg': f"{total_neg_correct / total_neg if total_neg else -1:.4f}",
                    'nut': f"{total_nut_correct / total_nut if total_nut else -1:.4f}",
                    'pos': f"{total_pos_correct / total_pos if total_pos else -1:.4f}",
                })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        mae = total_absolute_error / total_samples
        avg_neg_acc = total_neg_correct / total_neg if total_neg else -1
        avg_nut_acc = total_nut_correct / total_nut if total_nut else -1
        avg_pos_acc = total_pos_correct / total_pos if total_pos else -1
        if training:
            tqdm.write(validation_strs)
            logging.info(validation_strs)

        return avg_loss, avg_acc, mae, avg_neg_acc, avg_nut_acc, avg_pos_acc

    def validate(self, data_loader: DataLoader, log_mlflow: bool = False, steps: int = 0, early_save:bool = False):
        """Validate the model on the validation set"""
        val_loss, val_acc, mae, val_neg_acc, val_nut_acc, val_pos_acc = self.evaluate(data_loader)
        if log_mlflow:
            self.log_to_mlflow(False,val_loss, val_acc, mae, val_neg_acc, val_nut_acc, val_pos_acc, steps)

        if early_save:
            model_path = "baseline_plus_augmented"
            dir = "saved_weights/" + self.model_name + "/"
            os.makedirs(dir, exist_ok=True)
            if 1-mae*0.5 > self.best_score:
                full_p = dir + model_path + ".pt"
                torch.save(self.state_dict(), full_p)
                print(f"{1-mae*0.5} > {self.best_score}, Saved model weights with best score to {full_p}")
                self.best_score = 1 - mae * 0.5
            if val_loss < self.best_loss:
                full_p = dir + model_path + "best_loss.pt"
                torch.save(self.state_dict(), full_p)
                print(f"{val_loss} < {self.best_loss}, Saved model weights with best loss to {full_p}")
                self.best_score = val_loss

        s = ((f"\nVal Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mae: {mae:.4f}, Lscore: {1-mae*0.5:.4f}, " if val_loss else "") +
            (f"Val Neg Acc: {val_neg_acc:.4f}, Nut Acc: {val_nut_acc:.4f}, Pos Acc: {val_pos_acc:.4f}" if val_loss else ""))
        # tqdm.write(s)
        # logging.info(s)

        self.train()
        return s

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float,float,float,float,float]:
        """Evaluate the model on a data loader."""
        self.eval()
        with torch.no_grad():
            return self._run_epoch(data_loader, training=False)

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """Generate argmax predictions."""
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

    def predict_sm(self, data_loader: DataLoader) -> torch.Tensor:
        """Generate softmax not argmax predictions."""
        self.eval()
        all_preds = []
        pbar = tqdm(data_loader, desc="Predicting",unit='batch', leave=False)
        with torch.no_grad():
            for batch in pbar:
                x, _, kwargs = self._unpack_batch(batch)
                x = x.to(self.device)
                outputs = self(x, **kwargs)
                preds = outputs.softmax(dim=1) #Softmax not argmax
                all_preds.append(preds.cpu())
        return self._adjust_predictions(torch.cat(all_preds))

    def get_logits(self, data_loader: DataLoader) -> torch.Tensor:
        self.eval()
        all_logits = []
        pbar = tqdm(data_loader, desc="Predicting",unit='batch', leave=False)
        with torch.no_grad():
            for batch in pbar:
                x, _, kwargs = self._unpack_batch(batch)
                x = x.to(self.device)
                logits = self(x, **kwargs)
                all_logits.append(logits.cpu())
        return(torch.cat(all_logits))

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

    @staticmethod
    def log_to_mlflow(training:bool, val_loss:float, val_acc:float, mae:float, val_neg_acc:float, val_nut_acc:float, val_pos_acc:float, steps:int):
        if training:
            prefix = 'train_'
        else:
            prefix = 'val_'
        mlflow.log_metric(prefix + 'loss', val_loss, step=steps)
        mlflow.log_metric(prefix + 'acc', val_acc, step=steps)
        mlflow.log_metric(prefix + 'mae', mae, step=steps)
        mlflow.log_metric(prefix + 'L_score', 1 - mae * 0.5, step=steps)
        mlflow.log_metric(prefix + 'neg_acc', val_neg_acc, step=steps)
        mlflow.log_metric(prefix + 'pos_acc', val_pos_acc, step=steps)
        mlflow.log_metric(prefix + 'nut_acc', val_nut_acc, step=steps)