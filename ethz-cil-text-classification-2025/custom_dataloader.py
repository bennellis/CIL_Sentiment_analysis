import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_list, labels):
        """
        Args:
            embeddings_list: List of numpy arrays where each array is [seq_len, embedding_dim]
            labels: List of corresponding labels
        """
        self.embeddings = list(embeddings_list)
        self.labels = list(labels)
        self._validate_data()
        
    def _validate_data(self):
        assert len(self.embeddings) == len(self.labels), "Mismatch between embeddings and labels"
        assert all(isinstance(emb, np.ndarray) for emb in self.embeddings), "All embeddings must be numpy arrays"
        assert all(emb.ndim == 2 for emb in self.embeddings), "Each embedding should be 2D (seq_len, emb_dim)"
        # Verify lengths match
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        try:
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