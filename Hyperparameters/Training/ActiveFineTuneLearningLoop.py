import torch
import numpy as np

import mlflow
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

import torch
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np



def active_learning_loop(
        model,
        device,
        train_embed_dataset,
        val_embed_dataset,
        train_unfrozen_dataloader,
        val_unfrozen_dataloader,
        query_fn,
        max_rounds=5,
        query_batch_size=1000,
        train_epochs_per_round=3,
        initial_label_count=1000,
        val_split=0.2,
        batch_size=32,
        log_mlflow=False,
        fine_tune=True
):
    # Split into train/val datasets (by indices)
    train_indices = list(range(len(train_embed_dataset)))
    val_indices = list(range(len(val_embed_dataset)))
    val_subset = val_embed_dataset
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Split train_indices into initial labeled + pool
    np.random.shuffle(train_indices)
    initial_labeled_indices = train_indices[:initial_label_count]
    unlabeled_indices = train_indices[initial_label_count:]

    # Start loop
    labeled_indices = list(initial_labeled_indices)
    pool_indices = list(unlabeled_indices)
    print(len(initial_labeled_indices), len(unlabeled_indices), len(labeled_indices))
    for r in range(max_rounds):
        print(f"\n🔁 Round {r + 1}/{max_rounds} — Labeled: {len(labeled_indices)}")

        train_subset = Subset(train_embed_dataset, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        model.fit(train_loader, val_loader, epochs=train_epochs_per_round, plot_metrics = False, log_mlflow = log_mlflow)


        if len(pool_indices) == 0:
            print("🎉 No more unlabeled samples.")
            break

        new_indices = query_fn(model, device, train_embed_dataset, pool_indices, n_select=query_batch_size)
        labeled_indices += new_indices
        pool_indices = list(set(pool_indices) - set(new_indices))

    Y_val_pred = model.predict(val_loader)
    Y_val = np.array([val_embed_dataset[i]['label'].item() for i in val_indices])


    mae_val = mean_absolute_error(Y_val, Y_val_pred)
    L_score_val = 0.5 * (2 - mae_val)
    print(f'Evaluation Score (validation set): {L_score_val:.05f}')

    conf_matrix = confusion_matrix(Y_val, Y_val_pred, labels=[-1, 0, 1])
    print(conf_matrix)


    if fine_tune:
        print(f'Fine tuning')
        model.unfreeze(keep_frozen=0)
        model.fit(train_unfrozen_dataloader, val_unfrozen_dataloader, epochs=3, log_mlflow=True)
        Y_val_pred = model.predict(val_unfrozen_dataloader)
        Y_val = np.array([val_embed_dataset[i]['label'].item() for i in val_indices])

        mae_val = mean_absolute_error(Y_val, Y_val_pred)
        L_score_val = 0.5 * (2 - mae_val)
        print(f'Evaluation Score (validation set): {L_score_val:.05f}')

        conf_matrix = confusion_matrix(Y_val, Y_val_pred, labels=[-1, 0, 1])
        print(conf_matrix)

    if log_mlflow:
        mlflow.log_metric('mae', mae_val)
        mlflow.log_metric('L_score', L_score_val)


    return L_score_val



import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

def entropy(probs):
    """Compute entropy for a batch of probability distributions"""
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

def query_entropy(model, device, dataset, unlabeled_indices, batch_size=128, n_select=100):
    # Create dataloader for the unlabeled pool
    unlabeled_loader = DataLoader(
        Subset(dataset, unlabeled_indices),
        batch_size=batch_size
    )

    # Get logits using model's helper
    model.eval()
    with torch.no_grad():
        logits = model.get_logits(unlabeled_loader)  # shape: [N, num_classes]

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1).cpu().numpy()  # shape: [N, num_classes]

    # Compute entropy for each sample
    entropies = entropy(probs)  # shape: [N]

    # Get the top-N highest entropy samples
    sorted_indices = np.argsort(-entropies)  # descending
    selected_relative_indices = sorted_indices[:n_select]

    # Map relative indices back to dataset indices
    selected_absolute_indices = [unlabeled_indices[i] for i in selected_relative_indices]

    return selected_absolute_indices
