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

from Hyperparameters.data.preprocessing import Preprocessor


class Objective:
    def __init__(self,
                 model_name="distilbert/distilbert-base-uncased",
                 csv_path="data/Sentiment/training.csv",
                 seed=42,
                 test_model=False,
                 balance_train_dataloader = True,
                 balance_val_dataloader = True,
                 head = 'mlp',
                 use_frozen = True,
                 pre_process = False,
                 pre_process_name = None,
                 use_augmented_data = False,
                 mean_pool = False,):
        self.pre_process = pre_process
        self.mean_pool = mean_pool
        self.use_augmented_data = use_augmented_data
        self.head=head
        self.seed = seed
        self.model_name = model_name
        self.balance_train_dataloader = balance_train_dataloader
        self.balance_val_dataloader = balance_val_dataloader
        self.removed_train = 0
        self.removed_val = 0
        self.added_val_score = 0
        self.pre = Preprocessor(
            remove_urls= pre_process_name == 'remove_urls',
            expand_contractions=pre_process_name == 'expand_contractions',  # can't -> can not
            remove_punctuation=pre_process_name == 'remove_punctuation',
            remove_stopwords=pre_process_name == 'remove_stopwords',
            normalize_repeated_chars=pre_process_name == 'normalize_repeated_chars',  # sooooo -> soo
            lemmatize=pre_process_name == 'lemmatize',
            remove_numbers=pre_process_name == 'remove_numbers',
        )
        self.pre_process_name = pre_process_name

        # Load and preprocess once
        self.embedder, self.train_data, self.val_data, self.val_texts = self._prepare_data(csv_path)
        self._prepare_dataloaders(use_frozen)
        self.test_model = test_model


    def _prepare_data(self, csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['label_encoded'] = df['label'].map(label_map)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['sentence'], df['label_encoded'],
            stratify=df['label_encoded'], test_size=0.1, random_state=self.seed
        )
        if self.use_augmented_data:
            augment_csv_path = 'data/Sentiment/training_augmented_with_reference.csv'
            augmented_df = pd.read_csv(augment_csv_path)
            augmented_df['label_encoded'] = augmented_df['label'].map(label_map)
            # print("Columns in augmented_df:", augmented_df.columns)
            augmented_train_df = augmented_df[(augmented_df['original_id'].isin(train_texts.index))& (augmented_df['is_augmented'] == True)]
            train_texts = pd.concat([train_texts, augmented_train_df['sentence']], ignore_index=True)
            train_labels = pd.concat([train_labels, augmented_train_df['label_encoded']], ignore_index=True)


        if self.pre_process: #pre-process data
            len_orig_train = len(train_texts)
            train_texts, train_kept_indices = self.pre.transform(train_texts)
            train_labels = train_labels.iloc[train_kept_indices].tolist()
            self.removed_train = len_orig_train - len(train_texts)
            print(f'removed training samples: {self.removed_train}')

            len_orig_val = len(val_texts)
            all_indices = set(range(len_orig_val))
            val_texts_new, val_kept_indices = self.pre.transform(val_texts)
            kept_set = set(val_kept_indices)
            removed_indices = all_indices - kept_set
            sum_removed = sum(abs(val_labels[i]) for i in removed_indices)
            val_texts = val_texts_new
            val_labels = val_labels.iloc[val_kept_indices].tolist()
            self.removed_val = len_orig_val - len(val_texts)
            print(f'removed validation samples: {self.removed_val}')

            self.added_val_score = float(sum_removed) / len_orig_val

        NUM_SAMPLES = -1
        NUM_VAR_SAMPLES = -1 if NUM_SAMPLES == -1 else int(NUM_SAMPLES/10)
        # Create embedding model & features
        embedder = BertTokenEmbedder(self.model_name, head = self.head, mean_pool = self.mean_pool)
        X_train = embedder.fit_transform(list(train_texts)[:NUM_SAMPLES])
        X_val = embedder.transform(list(val_texts)[:NUM_VAR_SAMPLES])

        Y_train = np.array(train_labels)[:NUM_SAMPLES]
        Y_val = np.array(val_labels)[:NUM_VAR_SAMPLES]

        return embedder, (X_train, Y_train), (X_val, Y_val), list(val_texts)

    def _static_balance_set(self, sentences, labels):
        ds = list(zip(sentences, labels))
        class_minus1 = [x for x in ds if x[1] == -1]
        class_0 = [x for x in ds if x[1] == 0]
        class_1 = [x for x in ds if x[1] == 1]

        # Find the smallest class size
        min_size = min(len(class_minus1), len(class_0), len(class_1))

        # Undersample all classes to match the smallest class
        class_minus1_bal = resample(class_minus1, replace=False, n_samples=min_size, random_state=self.seed)
        class_0_bal = resample(class_0, replace=False, n_samples=min_size, random_state=self.seed)
        class_1_bal = resample(class_1, replace=False, n_samples=min_size, random_state=self.seed)

        # Combine and shuffle
        balanced_ds = class_minus1_bal + class_0_bal + class_1_bal
        np.random.shuffle(balanced_ds)

        # Split back into sentences and labels
        sentences_bal, labels_bal = zip(*balanced_ds)
        return list(sentences_bal), np.array(labels_bal)

    def _prepare_dataloaders(self, use_frozen):
        """Precompute frozen and unfrozen dataloaders"""
        X_train, Y_train = self.train_data
        X_val, Y_val = self.val_data

        if self.embedder.is_variable_length:
            train_dataset = EmbeddingDataset(X_train, Y_train)
            val_dataset = EmbeddingDataset(X_val, Y_val)

            # Class-balanced sampler
            train_sampler = DynamicUnderSampler(Y_train, random_state=self.seed)

            if self.balance_train_dataloader:
                self.train_loader_unfrozen = DataLoader(train_dataset, sampler=train_sampler, batch_size=8,
                                                        collate_fn=collate_fn)
                ex_train = '_bal'
            else:
                self.train_loader_unfrozen = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn,
                                                        shuffle=True)
                ex_train = ''
            if self.balance_val_dataloader:
                X_val_bal, Y_val_bal = self._static_balance_set(X_val, Y_val)
                val_dataset = EmbeddingDataset(X_val_bal, Y_val_bal)
                self.val_loader_unfrozen = DataLoader(val_dataset,  batch_size=64,
                                                      collate_fn=collate_fn)
                ex_val = '_bal'
            else:
                self.val_loader_unfrozen = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
                ex_val = ''
            if self.head != 'mlp':
                ex_train = ex_train + '_' + self.head
                ex_val = ex_val + '_' + self.head
            if self.use_augmented_data:
                ex_train = ex_train + '_augmented'
            if self.mean_pool:
                ex_train = ex_train + '_mean_pool'
                ex_val = ex_val + '_mean_pool'

            cache_name= self.model_name.replace("/", "_")
            cache_path = "cache/" + cache_name

            cache_train_path = cache_path + ex_train + "_train.pt"
            cache_val_path = cache_path + ex_val + "_val.pt"

            if use_frozen:
                if os.path.exists(cache_train_path) and not self.pre_process:
                    self.train_loader_frozen = torch.load(cache_train_path, weights_only=False)
                else:
                    train_loader_pred = DataLoader(train_dataset, batch_size=8,collate_fn=collate_fn)
                    self.train_loader_frozen = self.embedder.precompute_embeddings(train_loader_pred,
                                                                                   balance = self.balance_train_dataloader)
                    if self.head == 'mlp' and not self.pre_process:
                        os.makedirs("cache", exist_ok=True)
                        torch.save(self.train_loader_frozen, cache_train_path)

                if os.path.exists(cache_val_path) and not self.pre_process:
                    self.val_loader_frozen = torch.load(cache_val_path, weights_only=False)
                else:
                    val_loader_pred = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)
                    self.val_loader_frozen = self.embedder.precompute_embeddings(val_loader_pred, val=True,
                                                                                 balance = self.balance_val_dataloader)
                    if self.head == 'mlp' and not self.pre_process:
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
                                             **params, head = self.head,
                                             mean_pool = self.mean_pool)
            model.to(get_device())

            frozen_epochs = 0
            unfrozen_epochs = 4
            validations_per_epoch = 10
            keep_frozen_layers = 0
            early_save = True

            ex_params = {
                "model_name": self.model_name,
                "frozen_epochs": frozen_epochs,
                "unfrozen_epochs": unfrozen_epochs,
                "validations_per_epoch": validations_per_epoch,
                "keep_frozen_layers": keep_frozen_layers,
                'balance_train_dataloader': self.balance_train_dataloader,
                'balance_val_dataloader': self.balance_val_dataloader,
                'head': self.head,
                'removed_train': self.removed_train,
                'removed_val': self.removed_val,
                'added_val_score': self.added_val_score,
                'pre_process_name': self.pre_process_name,
                'use_augmented_data': self.use_augmented_data,
                'mean_pool': self.mean_pool,
            }
            mlflow.log_params(ex_params)
            mlflow.set_tag("mlflow.note.content", self.model_name + " baseline model test + augmented data") # Use this as a description of the test if wanted

            # Train classifier head only
            pre_steps = 0
            if frozen_epochs > 0:
                model.fit(self.train_loader_frozen, self.val_loader_frozen, epochs=frozen_epochs, log_mlflow=True, initial_steps=0)
                pre_steps = len(self.train_loader_frozen)


            # Fine-tune transformer layers
            model.unfreeze(keep_frozen=keep_frozen_layers)
            model.fit(self.train_loader_unfrozen, self.val_loader_unfrozen, epochs=unfrozen_epochs, log_mlflow=True,
                      validations_per_epoch=validations_per_epoch, initial_steps = pre_steps*frozen_epochs, early_save = early_save)

            # Evaluate
            Y_val = self.val_data[1]
            Y_pred = model.predict(self.val_loader_unfrozen)
            mae = mean_absolute_error(Y_val, Y_pred)
            score = 0.5 * (2 - mae)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("L_score", score)

            conf_matrix = confusion_matrix(Y_val, Y_pred, labels=[-1, 0, 1])



            print(f"Validation Confusion Matrix:\n{conf_matrix}")
            if self.test_model:
                test_data = pd.read_csv('data/Sentiment/test.csv', index_col=0)
                orig_size = len(test_data)
                if self.pre_process:
                    sentences, kept_indices = self.pre.transform(test_data["sentence"])
                    X_test = self.embedder.transform(sentences)
                else:
                    X_test = self.embedder.transform(test_data['sentence'])
                Y_test_fake_labels = np.ones(X_test.shape[0])
                dataset_test = EmbeddingDataset(X_test, Y_test_fake_labels)
                # print(len(dataset_train))
                test_loader = DataLoader(
                    dataset_test,
                    batch_size=64,
                    collate_fn=collate_fn,
                )
                y_test = model.predict(test_loader)

                y_labels = pd.Series(y_test).map({-1: 'negative', 0: 'neutral', 1: 'positive'})
                if self.pre_process:
                    full_labels = pd.Series(['neutral'] * len(test_data), index=test_data.index)
                    full_labels.iloc[kept_indices] = y_labels.values
                    y_labels = full_labels
                submission = pd.DataFrame({'id': test_data.index, 'label': y_labels})
                submission.to_csv('test_predictions.csv', index=False)  # Update filename and path as needed
                print("Test predictions saved to 'test_predictions.csv'")

            return score
