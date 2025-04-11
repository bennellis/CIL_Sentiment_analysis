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

class BaseModel(ABC):

    use_dataloader: bool

    @abstractmethod
    def fit(self, X_train=None, Y_train=None,train_loader=None):
        pass

    @abstractmethod
    def predict(self, X=None, data_loader=None):
        pass


class Logistic_Regression_Baseline(BaseModel):
    use_dataloader = False

    def __init__(self, C=1.0, max_iter=100):
        self.model = LogisticRegression(
            C=C, max_iter=max_iter
        )
        self._is_fitted = False

    def fit(self, X_train, Y_train, train_loader=None):
        self._is_fitted = True
        return self.model.fit(X_train, Y_train)
    
    def predict(self,X):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def coef(self):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
        return self.model.coef_
    


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
        # Convert logits to probabilities
        y_pred_prob = torch.softmax(y_pred, dim=1)  # Now differentiable

        # Compute weighted sum of class indices for expected class prediction
        y_pred_expected = torch.sum(y_pred_prob * torch.arange(y_pred.shape[1], device=y_pred.device), dim=1)

        # Ensure y_true is float for computation
        y_true = y_true.float()

        # Compute Mean Absolute Error (MAE) on probabilities (differentiable)
        mae = torch.mean(torch.abs(y_pred_expected - y_true))

        # Compute the loss: 1 - custom score
        loss = 1 - 0.5 * (2 - mae)

        return loss
    

class Linear_MLP(BaseModel, nn.Module): 
    use_dataloader = True
    
    def __init__(self, input_dim,lr=0.001):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 3)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = CustomLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)

    def fit(self, train_loader, epochs=5, X_train=None, Y_train=None):
        self.train()
        print(f'Training model {self.__class__.__name__} on {self.device}')
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss+=loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    def predict(self, data_loader,X=None):
        self.eval()
        all_preds = []
        
        with torch.no_grad():
            for X_batch, _ in data_loader:  # Ignore labels if present
                X_batch = X_batch.to(self.device)
                outputs = self(X_batch)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
        
        return torch.cat(all_preds)

    def evaluate(self, data_loader):
        self.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self(X_batch)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total_samples += y_batch.size(0)
                
        return total_correct / total_samples
        


class Non_Linear_MLP(BaseModel, nn.Module): 
    use_dataloader = True
    
    def __init__(self, input_dim,lr=0.001):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = CustomLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)

    def fit(self, train_loader, epochs=5, X_train=None, Y_train=None):
        self.train()
        print(f'Training model {self.__class__.__name__} on {self.device}')
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss+=loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    def predict(self, data_loader,X=None):
        self.eval()
        all_preds = []
        
        with torch.no_grad():
            for X_batch, _ in data_loader:  # Ignore labels if present
                X_batch = X_batch.to(self.device)
                outputs = self(X_batch)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
        
        return torch.cat(all_preds)

    def evaluate(self, data_loader):
        self.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self(X_batch)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total_samples += y_batch.size(0)
                
        return total_correct / total_samples

class Dropout_MLP(BaseModel, nn.Module): 
    use_dataloader = True
    
    def __init__(self, input_dim,lr=0.001,dropout_prob=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = CustomLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)

    def fit(self, train_loader, epochs=5, X_train=None, Y_train=None):
        self.train()
        print(f'Training model {self.__class__.__name__} on {self.device}')
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss+=loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    def predict(self, data_loader,X=None):
        self.eval()
        all_preds = []
        
        with torch.no_grad():
            for X_batch, _ in data_loader:  # Ignore labels if present
                X_batch = X_batch.to(self.device)
                outputs = self(X_batch)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
        
        return torch.cat(all_preds)

    def evaluate(self, data_loader):
        self.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self(X_batch)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total_samples += y_batch.size(0)
                
        return total_correct / total_samples
    