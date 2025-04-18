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
from typing import List, Optional, Callable


class BaseModel(ABC, nn.Module):
    """Base class for PyTorch models using DataLoaders"""
    use_dataloader = True

    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.criterion = CustomLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @abstractmethod
    def fit(self, train_loader: torch.Tensor, epochs=5):
        pass

    @abstractmethod
    def predict(self, data_loader):
        pass

    def evaluate(self, data_loader):
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self(x_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions-1 == y_batch).sum().item()
                total_samples += y_batch.size(0)

        return total_loss / total_samples , total_correct / total_samples

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
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Custom loss function for sentiment analysis.

        Args:
            y_pred: Predicted logits (before softmax), shape: (batch_size, num_classes)
            y_true: True labels (LongTensor), shape: (batch_size,)

        Returns:
            loss (Tensor): Loss value to minimize
        """
        # ce_loss = nn.CrossEntropyLoss()(y_pred, (y_true+1).long()) #CE loss for hybrid loss
        ce_loss = 0


        # Convert logits to probabilities
        temperature = 0.5
        y_pred_prob = torch.softmax(y_pred / temperature, dim=1)  # Now differentiable
        class_values = torch.tensor([-1.0, 0.0, 1.0], device=y_pred.device)

        # Compute weighted sum of class indices for expected class prediction
        y_pred_expected = torch.sum(y_pred_prob * class_values, dim=1)

        # Ensure y_true is float for computation
        y_true = y_true.float()

        # Compute Mean Absolute Error (MAE) on probabilities (differentiable)
        # print(y_pred_prob, y_pred_expected)
        mae = torch.mean(torch.abs(y_pred_expected - y_true))
        

        # Compute the loss: 1 - custom score
        loss = 1 - 0.5 * (2 - mae)
        # loss = 0

        return loss + ce_loss


class BaseMLP(BaseModel):
    def __init__(self, input_dim, output_dim=3, lr=0.001):
        super().__init__(lr=lr)
        self._build_layers(input_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def _build_layers(self, input_dim, output_dim):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def forward(self, x):
        return self.layers(x)

    def fit(self, train_loader, val_loader=None, epochs=5):
        print(f'Training {self.__class__.__name__} on {self.device}')
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(epochs):
            self.train()  # since it validates after every train needs to set back into train mode
            total_train_loss = 0.0
            total_train_correct = 0
            total_train_samples = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * y_batch.size(0)

                predictions = outputs.argmax(dim=1)
                total_train_correct += (predictions-1 == y_batch).sum().item()
                total_train_samples += y_batch.size(0)

            avg_train_loss = total_train_loss / total_train_samples
            train_accuracy = total_train_correct / total_train_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Training Loss: {avg_train_loss:.4f}, '
                  f'Training Accuracy: {train_accuracy:.4f}')

            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Validation Loss: {val_loss:.4f}, '
                      f'Validation Accuracy: {val_accuracy:.4f}')

        self.plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)

    def predict(self, data_loader):
        self.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(self.device)
                outputs = self(x_batch)
                all_preds.append(outputs.argmax(dim=1).cpu())
        return torch.cat(all_preds) - 1  # Remap to [-1, 0, 1]


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

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.2, lr=0.001):
        super().__init__(lr=lr)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 3)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self._init_weights()
        self.to(self.device)
        # print(self.device)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

    def fit(self, train_loader, val_loader = None, epochs=5):

        print(f'Training {self.__class__.__name__} on {self.device}')
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.train() # since it validates after every train needs to set back into train mode
            total_train_loss = 0.0
            total_train_correct = 0
            total_train_samples = 0
            for x_batch, lengths, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(x_batch, lengths)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                total_train_loss += loss.item() * y_batch.size(0)
                predictions = outputs.argmax(dim=1)
                total_train_correct += (predictions-1 == y_batch).sum().item()
                total_train_samples += y_batch.size(0)

            avg_train_loss = total_train_loss / total_train_samples
            train_accuracy = total_train_correct / total_train_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Training Loss: {avg_train_loss:.4f}, '
                  f'Training Accuracy: {train_accuracy:.4f}')

            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                print(f'Epoch [{epoch + 1}/{epochs}], '
                      f'Validation Loss: {val_loss:.4f}, '
                      f'Validation Accuracy: {val_accuracy:.4f}')

        self.plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)

    def predict(self, data_loader):
        self.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, lengths, _ in data_loader:
                x_batch = x_batch.to(self.device)
                outputs = self(x_batch, lengths)
                all_preds.append(outputs.argmax(dim=1).cpu() - 1)
        return torch.cat(all_preds)

    def evaluate(self, data_loader):
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, lengths, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self(x_batch, lengths)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions-1 == y_batch).sum().item()
                total_samples += y_batch.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy
