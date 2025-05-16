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
from Hyperparameters.Training.ActiveLearningLoop import query_entropy, active_learning_loop
from Hyperparameters.Utils.Misc import get_device


class Objective:
    def __init__(self, model_name="FacebookAI/roberta-large", csv_path="data/Sentiment/training.csv", seed=42):
        self.seed = seed
        self.model_name = model_name
        self.csv_path = csv_path
        # Load and preprocess once
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):

        df = pd.read_csv(self.csv_path, index_col=0)
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['label_encoded'] = df['label'].map(label_map)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['sentence'], df['label_encoded'],
            stratify=df['label_encoded'], test_size=0.1, random_state=self.seed
        )
        embedder = BertTokenEmbedder(self.model_name)
        features = embedder.fit_transform(df['sentence'].to_list())
        labels = df['label_encoded'].to_numpy()

        if embedder.is_variable_length:
            feature_dataset = EmbeddingDataset(features, labels)

            cache_name = self.model_name.replace("/", "_")
            cache_path = "cache/" + cache_name
            emb_dataset_path = cache_path + "emb_dataset.pt"

            if os.path.exists(emb_dataset_path):
                self.embedded_feature_dataset = torch.load(emb_dataset_path, weights_only=False)
            else:
                feature_dataloader = DataLoader(feature_dataset, batch_size=8, collate_fn=collate_fn)
                self.embedded_feature_dataset = embedder.embed_dataset(feature_dataloader)
                os.makedirs("cache", exist_ok=True)
                torch.save(self.embedded_feature_dataset, emb_dataset_path)

        else:
            raise Exception("blaalalal")

    def __call__(self, trial):
        with mlflow.start_run():
            params = BertPreTrainedClassifier.suggest_hyperparameters(trial)
            mlflow.log_params(params)

            model = BertPreTrainedClassifier(model_name=self.model_name,
                                             frozen=True,
                                             **params)

            return active_learning_loop(
                model,
                get_device(),
                self.embedded_feature_dataset,
                query_entropy,
                max_rounds=1000,
                query_batch_size=1000,
                train_epochs_per_round=3,
                initial_label_count=1000,
                val_split=0.2,
                batch_size=32,
                log_mlflow = True
            )


