from sklearn.feature_extraction.text import TfidfVectorizer

from Hyperparameters.Embeddings.VectorizerEmbedding import VectorizerEmbedding


class TFIDFEmbedding(VectorizerEmbedding):
    def __init__(self, **kwargs):
        super().__init__(TfidfVectorizer(**kwargs))
