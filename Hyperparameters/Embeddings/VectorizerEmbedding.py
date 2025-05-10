
from typing import List
from Hyperparameters.Embeddings.BaseEmbedding import BaseEmbedding

import numpy as np

class VectorizerEmbedding(BaseEmbedding):
    is_variable_length = False
    pre_compute = False

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self._is_fitted = False

    def fit_transform(self, train_sentences: List[str]) -> np.ndarray:
        self._is_fitted = True
        return self.vectorizer.fit_transform(train_sentences)

    def transform(self, sentences: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit_transform() first")
        return self.vectorizer.transform(sentences)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
