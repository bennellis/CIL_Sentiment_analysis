import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset, Subset

from Hyperparameters.Dataloader.DynamicUnderSampler import DynamicUnderSampler
from Hyperparameters.Embeddings.BertTokenEmbedder import BertTokenEmbedder
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn
from Hyperparameters.Models.BertPreTrainedClassifier import BertPreTrainedClassifier
from Hyperparameters.Training.ActiveLearningLoop import active_learning_loop
from Hyperparameters.Training.ActiveLearningLoop import query_entropy

from Hyperparameters.Utils.Misc import get_device

def main():
    seed = 42

    ## Model Parameters

    model_name = "FacebookAI/roberta-base"
    csv_path = "data/Sentiment/training.csv"
    lr_frozen = 0.0003177750766003565
    lr_unfrozen = 1e-5
    class_order = [0, 1, 2]
    lr_top = 1e-5
    lr_mid = 1e-5
    lr_bot = 1e-5
    dropout = 0.023447680612155252

    ## Loss Parameters



    ### New Loss Parameters


    criterion_name = "CustomLoss"
    loss_params = {
        "temperature": 0.5145615852938422,
        "ce_weight": 0.6203440815982639
    }


    ## data parameters

    batch_size_feature_embed = 128
    val_split = 0.2

    ## Training

    ## active learning
    train_active_learning = True
    active_max_rounds = 1000
    active_query_batch_size = 1000
    active_train_epochs_per_round = 3
    active_initial_label_count = 1000
    batch_size_frozen = 32 # Note that when made bigger must also change the learning rate

    ## unfrozen
    train_unfrozen = True
    keep_frozen = 2
    epochs_unfrozen = 2
    batch_size_unfrozen = 32

    # Output

    save_model = False
    model_save_path = "roberta_large_active_loss"

    ##################


    df = pd.read_csv(csv_path, index_col=0)
    label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['label_encoded'] = df['label'].map(label_map)

    all_indices = list(range(len(df)))
    train_indices, val_indices = train_test_split(all_indices, test_size=val_split, stratify=df['label_encoded'])

    embedder = BertTokenEmbedder(model_name)

    features = embedder.fit_transform(df['sentence'].to_list(), batch_size=batch_size_feature_embed)
    labels = df['label_encoded'].to_numpy()

    train_loader_unfrozen = None
    val_loader_unfrozen = None

    if embedder.is_variable_length:

        feature_dataset = EmbeddingDataset(features, labels)

        cache_name = model_name.replace("/", "_")
        cache_path = "cache/" + cache_name
        emb_dataset_path = cache_path + "emb_dataset.pt"

        if os.path.exists(emb_dataset_path):
            embedded_feature_dataset = torch.load(emb_dataset_path, weights_only=False)
        else:
            feature_dataloader = DataLoader(feature_dataset, batch_size=batch_size_feature_embed, collate_fn=collate_fn)
            embedded_feature_dataset = embedder.embed_dataset(feature_dataloader)
            os.makedirs("cache", exist_ok=True)
            torch.save(embedded_feature_dataset, emb_dataset_path)

        if train_unfrozen:
            train_sampler = DynamicUnderSampler(labels[train_indices], random_state=seed)
            train_dataset_unfrozen = Subset(feature_dataset, train_indices)
            train_loader_unfrozen = DataLoader(
                train_dataset_unfrozen,
                sampler=train_sampler,
                batch_size=batch_size_unfrozen,
                collate_fn=collate_fn
            )

            val_dataset_unfrozen = Subset(feature_dataset, val_indices)
            val_loader_unfrozen = DataLoader(
                val_dataset_unfrozen,
                batch_size=batch_size_unfrozen,
                collate_fn=collate_fn
            )


    else:
        X_tensor = torch.tensor(features, dtype=torch.float32)
        Y_tensor = torch.tensor(labels, dtype=torch.long)
        tensor_dataset_unfrozen = TensorDataset(X_tensor, Y_tensor)
        embedded_feature_dataset = tensor_dataset_unfrozen

        train_dataset_unfrozen = Subset(tensor_dataset_unfrozen, train_indices)
        train_loader_unfrozen = DataLoader(
            train_dataset_unfrozen,
            batch_size=batch_size_unfrozen
        )

        val_dataset_unfrozen = Subset(tensor_dataset_unfrozen, val_indices)
        val_loader_unfrozen = DataLoader(
            val_dataset_unfrozen,
            batch_size=batch_size_unfrozen
        )

    model = BertPreTrainedClassifier(
        model_name=model_name,
        criterion_name=criterion_name,
        lr=lr_frozen,
        pt_lr_bot=lr_bot,
        pt_lr_mid=lr_mid,
        pt_lr_top=lr_top,
        class_order=class_order,
        frozen=True,
        custom_ll=True,
        **loss_params
    )

    if train_active_learning:
        active_learning_loop(
            model=model,
            device=get_device(),
            dataset=embedded_feature_dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            query_fn=query_entropy,
            max_rounds=active_max_rounds,
            query_batch_size=active_query_batch_size,
            train_epochs_per_round=active_train_epochs_per_round,
            initial_label_count=active_initial_label_count,
            batch_size=batch_size_frozen
        )

    if train_unfrozen:
        model.unfreeze(keep_frozen=keep_frozen)
        model.set_lr(lr=lr_unfrozen)
        model.fit(
            train_loader_unfrozen,
            val_loader_unfrozen,
            epochs=epochs_unfrozen,
            plot_metrics=False,
            log_mlflow=False
        )

        Y_val_pred = model.predict(val_loader_unfrozen)
        Y_val = labels[val_indices]

        mae_val = mean_absolute_error(Y_val, Y_val_pred)
        L_score_val = 0.5 * (2 - mae_val)
        print(f'Evaluation Score (validation set): {L_score_val:.05f}')

        conf_matrix = confusion_matrix(Y_val, Y_val_pred, labels=[-1, 0, 1])
        print(conf_matrix)

    if save_model:
        model.model.save_pretrained("cache/" + model_save_path + "pretrained")
        model.model.config.save_pretrained("cache/" + model_save_path + "_config")
        model.tokenizer.save_pretrained("cache/" + model_save_path + "tokenizer")


if __name__ == "__main__":
    main()
