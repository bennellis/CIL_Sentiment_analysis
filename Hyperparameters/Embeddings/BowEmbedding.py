from sklearn.feature_extraction.text import CountVectorizer

from Hyperparameters.Embeddings.VectorizerEmbedding import VectorizerEmbedding


class BOWEmbedding(VectorizerEmbedding):
    def __init__(self, **kwargs):
        super().__init__(CountVectorizer(**kwargs))
