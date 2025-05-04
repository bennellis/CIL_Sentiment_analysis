import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import Sampler
from imblearn.under_sampling import RandomUnderSampler


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_list, labels, variable_length = True):
        """
        Args:
            embeddings_list: List of numpy arrays where each array is [seq_len, embedding_dim]
            labels: List of corresponding labels
        """
        # print(type(embeddings_list))
        # print(embeddings_list[0])
        self.embeddings = list(embeddings_list)
        # print(self.embeddings)
        # print(labels)
        self.labels = list(labels)
        # print(self.labels)
        self.variable_length = variable_length
        self._validate_data()

        
    def _validate_data(self):
        # print(emb.ndim)
        assert len(self.embeddings) == len(self.labels), "Mismatch between embeddings and labels"
        assert all(isinstance(emb, np.ndarray) for emb in self.embeddings), "All embeddings must be numpy arrays"
        if self.variable_length:
            assert all(emb.ndim == 2 for emb in self.embeddings), "Each embedding should be 2D (seq_len, emb_dim)"
        else:
            assert all(emb.ndim == 1 for emb in self.embeddings), "Each embedding should be 1D (emb_dim)"
        # Verify lengths match
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        try:
            # print(idx)
            # print(torch.tensor(self.labels[idx]))
            return {
                'embeddings': torch.from_numpy(self.embeddings[idx]).float(),
                'label': torch.tensor(self.labels[idx])
            }
        except IndexError as e:
            print(f"Error accessing index {idx} - max index is {len(self)-1}")
            raise

def collate_fn(batch):
    """
    Custom collate function to handle:
    - Variable-length embeddings
    - Corresponding labels
    Returns:
        padded_embeddings: Padded sequence tensor (batch_size, max_seq_len, embedding_dim)
        lengths: Original lengths of each sequence
        labels: Batch of labels
    """
    # Separate embeddings and labels
    embeddings = [item['embeddings'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Sort by sequence length (descending)
    sorted_indices = sorted(
        range(len(embeddings)),
        key=lambda i: embeddings[i].shape[0],
        reverse=True
    )
    embeddings = [embeddings[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Pad sequences
    lengths = torch.tensor([len(x) for x in embeddings])
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(
        embeddings,
        batch_first=True,
        padding_value=0
    )
    
    # Stack labels
    labels = torch.stack(labels)
    
    return padded_embeddings, lengths, labels


class DynamicUnderSampler(Sampler):
    def __init__(self, y, random_state=None):
        """
        Args:
            y: Array-like, labels of the dataset.
            random_state: Seed for reproducibility.
        """
        super().__init__()
        self.y = np.array(y)
        self.random_state = random_state
        self.sampler = RandomUnderSampler(random_state=random_state)

    def __iter__(self):
        # Resample and shuffle indices every epoch
        _, _ = self.sampler.fit_resample(self.y.reshape(-1, 1), self.y)
        resampled_indices = self.sampler.sample_indices_
        np.random.shuffle(resampled_indices)  # Critical: Shuffle after undersampling
        return iter(resampled_indices)

    def __len__(self):
        # Return length of the minority class * number of classes
        unique, counts = np.unique(self.y, return_counts=True)
        return counts.min() * len(unique)