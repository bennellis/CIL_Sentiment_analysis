
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Sampler

import numpy as np

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
