import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import os

import mlflow
import optuna
from sklearn.metrics import mean_absolute_error, confusion_matrix

from Hyperparameters.Dataloader.DynamicUnderSampler import DynamicUnderSampler
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn
from Hyperparameters.Embeddings.BertTokenEmbedder import BertTokenEmbedder
from Hyperparameters.Models.BertPreTrainedClassifier import BertPreTrainedClassifier
from Hyperparameters.Utils.Misc import get_device


class Objective:
    def __init__(self, model_name="answerdotai/ModernBERT-base", csv_path="data/Sentiment/training.csv", seed=42):
        self.seed = seed
        self.model_name = model_name

        # Load and preprocess once
        self.embedder, self.train_data, self.val_data, self.val_texts = self._prepare_data(csv_path)
        self._prepare_dataloaders()

    def _prepare_data(self, csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['label_encoded'] = df['label'].map(label_map)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['sentence'], df['label_encoded'],
            stratify=df['label_encoded'], test_size=0.1, random_state=self.seed
        )

        # Balance training data
        train_data = list(zip(train_texts, train_labels))
        grouped = [[x for x in train_data if x[1] == i] for i in [-1, 0, 1]]
        min_class_size = min(len(g) for g in grouped)
        balanced = sum([
            resample(g, replace=False, n_samples=min_class_size, random_state=self.seed) for g in grouped
        ], [])
        random.shuffle(balanced)
        train_texts, train_labels = zip(*balanced)

        # Create embedding model & features
        embedder = BertTokenEmbedder(self.model_name)
        X_train = embedder.fit_transform(list(train_texts))
        X_val = embedder.transform(list(val_texts))

        Y_train = np.array(train_labels)
        Y_val = np.array(val_labels)

        return embedder, (X_train, Y_train), (X_val, Y_val), list(val_texts)

    def _prepare_dataloaders(self):
        """Precompute frozen and unfrozen dataloaders"""
        X_train, Y_train = self.train_data
        X_val, Y_val = self.val_data

        if self.embedder.is_variable_length:
            train_dataset = EmbeddingDataset(X_train, Y_train)
            val_dataset = EmbeddingDataset(X_val, Y_val)

            # Class-balanced sampler
            train_sampler = DynamicUnderSampler(Y_train, random_state=self.seed)


            self.train_loader_unfrozen = DataLoader(train_dataset, sampler=train_sampler, batch_size=8, collate_fn=collate_fn)
            self.val_loader_unfrozen = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

            cache_name= self.model_name.replace("/", "_")
            cache_path = "cache/" + cache_name
            cache_train_path = cache_path + "_train.pt"
            cache_val_path = cache_path + "_val.pt"


            if os.path.exists(cache_train_path):
                self.train_loader_frozen = torch.load(cache_train_path, weights_only=False)
            else:
                train_loader_pred = DataLoader(train_dataset, sampler=train_sampler, batch_size=8,collate_fn=collate_fn)
                self.train_loader_frozen = self.embedder.precompute_embeddings(train_loader_pred)
                os.makedirs("cache", exist_ok=True)
                torch.save(self.train_loader_frozen, cache_train_path)

            if os.path.exists(cache_val_path):
                self.val_loader_frozen = torch.load(cache_val_path, weights_only=False)
            else:
                self.val_loader_frozen = self.embedder.precompute_embeddings(self.val_loader_unfrozen, val=True)
                os.makedirs("cache", exist_ok=True)
                torch.save(self.val_loader_frozen, cache_val_path)

        else:
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

            self.train_loader_frozen = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32)
            self.train_loader_unfrozen = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=16, shuffle=True)
            self.val_loader = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor), batch_size=64)

    def __call__(self, trial):
        with mlflow.start_run():
            params = BertPreTrainedClassifier.suggest_hyperparameters(trial)
            mlflow.log_params(params)

            model = BertPreTrainedClassifier(model_name=self.model_name,
                                             frozen=True,
                                             **params)
            model.to(get_device())

            # Train classifier head only
            model.fit(self.train_loader_frozen, self.val_loader_frozen, epochs=6, log_mlflow=True)

            # Fine-tune transformer layers
            model.unfreeze(keep_frozen=6)
            model.fit(self.train_loader_unfrozen, self.val_loader_unfrozen, epochs=2, log_mlflow=True)

            # Evaluate
            Y_val = self.val_data[1]
            Y_pred = model.predict(self.val_loader)
            mae = mean_absolute_error(Y_val, Y_pred)
            score = 0.5 * (2 - mae)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("L_score", score)

            conf_matrix = confusion_matrix(Y_val, Y_pred, labels=[-1, 0, 1])
            print(f"Validation Confusion Matrix:\n{conf_matrix}")

            return mae
