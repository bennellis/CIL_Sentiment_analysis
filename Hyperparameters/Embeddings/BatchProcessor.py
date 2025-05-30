
import numpy as np
from tqdm.auto import tqdm

class BatchProcessor:
    @staticmethod
    def process_in_batches(data, batch_size, process_fn, show_progress=True):
        """Generic batch processing with progress bar.

        Args:
            data: List of items to process
            batch_size: Number of items per batch
            process_fn: Function that processes a single batch
            show_progress: Whether to show tqdm progress bar

        Returns:
            Concatenated results from all batches
        """
        results = []
        iter_range = range(0, len(data), batch_size)
        if show_progress:
            iter_range = tqdm(iter_range, desc="Processing batches")

        for i in iter_range:
            batch = data[i:i + batch_size]
            batch_result = process_fn(batch)
            results.append(batch_result)

        return np.concatenate(results, axis=0) if len(results) > 0 else np.array([])
