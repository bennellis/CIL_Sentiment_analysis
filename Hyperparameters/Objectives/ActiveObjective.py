import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset, Subset
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
from Hyperparameters.registry import get_criterion


class Objective:
    def __init__(self,
                 model_name="FacebookAI/roberta-large",
                 csv_path="data/Sentiment/training.csv",
                 seed=42):

        self.seed = seed
        self.model_name = model_name
        self.csv_path = csv_path


        self.criterion_name = "CustomLoss"

        ## data parameters

        self.batch_size_feature_embed = 128
        self.val_split = 0.2

        ## Training

        ## active learning
        self.train_active_learning = True
        self.active_max_rounds = 1000
        self.active_query_batch_size = 1000
        self.active_train_epochs_per_round = 3
        self.active_initial_label_count = 1000
        self.batch_size_frozen = 256

        ## unfrozen
        self.train_unfrozen = True
        self.keep_frozen = 2
        self.epochs_unfrozen = 4
        self.batch_size_unfrozen = 2
        self.labels = None
        self.features = None
        self.train_indices = None
        self.val_indices = None
        self.embedded_feature_dataset = None
        self.train_loader_unfrozen = None
        self.val_loader_unfrozen = None

        self._prepare_dataloaders()


    def _prepare_dataloaders(self):

        df = pd.read_csv(self.csv_path, index_col=0)
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['label_encoded'] = df['label'].map(label_map)

        all_indices = list(range(len(df)))
        self.train_indices, self.val_indices = train_test_split(
            all_indices,
            test_size=self.val_split,
            stratify=df['label_encoded']
        )

        embedder = BertTokenEmbedder(self.model_name)

        self.features = embedder.fit_transform(df['sentence'].to_list(), batch_size=self.batch_size_feature_embed)
        self.labels = df['label_encoded'].to_numpy()

        if embedder.is_variable_length:

            feature_dataset = EmbeddingDataset(self.features, self.labels)

            cache_name = self.model_name.replace("/", "_")
            cache_path = "cache/" + cache_name
            emb_dataset_path = cache_path + "emb_dataset.pt"

            if os.path.exists(emb_dataset_path):
                self.embedded_feature_dataset = torch.load(emb_dataset_path, weights_only=False)
            else:
                feature_dataloader = DataLoader(feature_dataset, batch_size=self.batch_size_feature_embed,
                                                collate_fn=collate_fn)
                self.embedded_feature_dataset = embedder.embed_dataset(feature_dataloader)
                os.makedirs("cache", exist_ok=True)
                torch.save(self.embedded_feature_dataset, emb_dataset_path)

            if self.train_unfrozen:
                train_sampler = DynamicUnderSampler(self.labels[self.train_indices], random_state=self.seed)
                train_dataset_unfrozen = Subset(feature_dataset, self.train_indices)
                self.train_loader_unfrozen = DataLoader(
                    train_dataset_unfrozen,
                    sampler=train_sampler,
                    batch_size=self.batch_size_unfrozen,
                    collate_fn=collate_fn
                )

                val_dataset_unfrozen = Subset(feature_dataset, self.val_indices)
                self.val_loader_unfrozen = DataLoader(
                    val_dataset_unfrozen,
                    batch_size=self.batch_size_unfrozen,
                    collate_fn=collate_fn
                )

        else:
            x_tensor = torch.tensor(self.features, dtype=torch.float32)
            y_tensor = torch.tensor(self.labels, dtype=torch.long)
            tensor_dataset_unfrozen = TensorDataset(x_tensor, y_tensor)
            self.embedded_feature_dataset = tensor_dataset_unfrozen

            train_dataset_unfrozen = Subset(tensor_dataset_unfrozen, self.train_indices)
            self.train_loader_unfrozen = DataLoader(
                train_dataset_unfrozen,
                batch_size=self.batch_size_unfrozen
            )

            val_dataset_unfrozen = Subset(tensor_dataset_unfrozen, self.val_indices)
            self.val_loader_unfrozen = DataLoader(
                val_dataset_unfrozen,
                batch_size=self.batch_size_unfrozen
            )

    def __call__(self, trial):
        with mlflow.start_run(nested=True):
            metadata = {
                k: v for k, v in self.__dict__.items()
                if isinstance(v, (str, int, float, bool))  # Filter out big objects like datasets
            }
            mlflow.set_tags({f"meta.{k}": v for k, v in metadata.items()})

            criterion_params = get_criterion(self.criterion_name).suggest_hyperparameters(trial)
            mlflow.log_params(criterion_params)

            model_params = BertPreTrainedClassifier.suggest_hyperparameters(trial)
            mlflow.log_params(model_params)

            kwargs = {**criterion_params, **model_params}
            model = BertPreTrainedClassifier(
                model_name=self.model_name,
                criterion_name=self.criterion_name,
                frozen=True,
                custom_ll=True,
                **kwargs
            )

            result = None

            if self.train_active_learning:
                result = active_learning_loop(
                        model=model,
                        device=get_device(),
                        dataset=self.embedded_feature_dataset,
                        train_indices=self.train_indices,
                        val_indices=self.val_indices,
                        query_fn=query_entropy,
                        max_rounds=self.active_max_rounds,
                        query_batch_size=self.active_query_batch_size,
                        train_epochs_per_round=self.active_train_epochs_per_round,
                        initial_label_count=self.active_initial_label_count,
                        batch_size=self.batch_size_frozen,
                        plot_metrics=False,
                        log_mlflow=True,
                )

            if self.train_unfrozen:
                model.unfreeze(keep_frozen=self.keep_frozen)
                model.set_lr(lr=model_params["lr_unfrozen"])
                model.fit(
                    self.train_loader_unfrozen,
                    val_loader=None,
                    epochs=self.epochs_unfrozen,
                    plot_metrics=False,
                    log_mlflow=True
                )

                Y_val_pred = model.predict(self.val_loader_unfrozen)
                Y_val = self.labels[self.val_indices]

                mae_val = mean_absolute_error(Y_val, Y_val_pred)
                result = 0.5 * (2 - mae_val)
                print(f'Evaluation Score (validation set): {result:.05f}')
                mlflow.log_metric('mae', mae_val)
                mlflow.log_metric('L_score', result)

            return result
