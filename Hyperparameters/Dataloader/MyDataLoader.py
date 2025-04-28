
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

def load_training_data():
    training_data = pd.read_csv('training.csv',index_col = 0)
    label_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    training_data['label_encoded'] = training_data['label'].map(label_mapping)

    sentences = training_data['sentence']
    labels = training_data['label_encoded']


    X_train_pool, X_test, y_train_pool, y_test = train_test_split(sentences, labels, test_size=0.2, stratify=labels)
    initial_indices = select_small_balanced_sample(X_train_pool, y_train_pool, n_per_class=10)
    X_labeled = X_train_pool[initial_indices]
    y_labeled = y_train_pool[initial_indices]

    X_unlabeled = np.delete(X_train_pool, initial_indices, axis=0)
    y_unlabeled = np.delete(y_train_pool, initial_indices, axis=0)




def select_small_balanced_sample(X, y, n_per_class=10, random_state=None):
    """
    Select a small balanced sample from X, y.

    Args:
        X (np.ndarray or list-like): Feature data.
        y (np.ndarray or list-like): Labels.
        n_per_class (int): Number of samples to select per class.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        selected_indices (np.ndarray): Indices of selected samples in X, y.
    """
    rng = np.random.default_rng(seed=random_state)
    y = np.array(y)  # Make sure y is an array
    class_indices = defaultdict(list)

    # Group indices by class
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    selected_indices = []

    for label, indices in class_indices.items():
        if len(indices) < n_per_class:
            raise ValueError(f"Not enough samples for class {label}: requested {n_per_class}, but only {len(indices)} available.")

        selected = rng.choice(indices, size=n_per_class, replace=False)
        selected_indices.extend(selected)

    return np.array(selected_indices)
