
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_list, labels, variable_length=True):
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
            print(f"Error accessing index {idx} - max index is {len(self) - 1}")
            raise