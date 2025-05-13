
from typing import List
from abc import ABC, abstractmethod

import numpy as np
class BaseEmbedding(ABC):
    is_variable_length: bool
    pre_compute: bool
    @abstractmethod
    def fit_transform(self, train_sentences: List[str]) -> np.ndarray:
        """
            Fit and transform the training sentences into embeddings

            Args:
                train_sentences (list): List of sentences for embedding

            Returns:
                list[numpy.ndarray]: List of embeddings for each sentence
        """
        pass

    @abstractmethod
    def transform(self, sentences: List[str]) -> np.ndarray:
        """
            Convert a list of sentences to embeddings

            Args:
                sentences (list): List of sentences to convert to embeddings

            Returns:
                list[numpy.ndarray]: List of embeddings for each sentence
        """
        pass