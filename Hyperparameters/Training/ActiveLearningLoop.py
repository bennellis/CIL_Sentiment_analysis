
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def active_learning_loop(
    model,
    device,
    dataset,
    initial_labeled_indices,
    unlabeled_indices,
    val_loader,
    train_fn,
    query_fn,
    max_rounds=5,
    query_batch_size=100,
    train_epochs_per_round=3,
    batch_size=32
):
    labeled_indices = list(initial_labeled_indices)
    pool_indices = list(unlabeled_indices)

    for round in range(max_rounds):
        print(f"\n Round {round + 1}/{max_rounds} — Labeled: {len(labeled_indices)}")

        # Prepare training loader
        train_loader = DataLoader(Subset(dataset, labeled_indices), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train
        for epoch in range(train_epochs_per_round):
            train_loss = train_fn(model, device, train_loader, optimizer, epoch)

        # Evaluate
        val_accuracy = evaluate(model, device, val_loader)
        print(f"✅ Validation Accuracy: {val_accuracy:.4f}")

        # Stop if no more data to sample
        if len(pool_indices) == 0:
            break

        # Query new samples using the strategy
        new_indices = query_fn(model, device, dataset, pool_indices, n_select=query_batch_size)

        # Update sets
        labeled_indices += new_indices
        pool_indices = list(set(pool_indices) - set(new_indices))

    return labeled_indices


def evaluate(model, device, val_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred)




def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

def query_entropy(model, device, dataset, unlabeled_indices, batch_size=128, n_select=100):
    model.eval()
    scores = []
    selected_indices = []

    unlabeled_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, unlabeled_indices), batch_size=batch_size
    )

    with torch.no_grad():
        for i, (x, _) in enumerate(unlabeled_loader):
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            batch_scores = entropy(probs)

            start_idx = i * batch_size
            batch_indices = unlabeled_indices[start_idx:start_idx + len(x)]
            scores.extend(zip(batch_indices, batch_scores))

    # Sort and select top-N
    sorted_scores = sorted(scores, key=lambda x: -x[1])
    selected_indices = [idx for idx, _ in sorted_scores[:n_select]]
    return selected_indices
