#
# from typing import List, Optional, Callable
#
# from Hyperparameters.Embeddings.BaseEmbedding import BaseEmbedding
#
# from tqdm.auto import tqdm
# from gensim.models import KeyedVectors
# import numpy as np
#
# class Word2VecEmbedding2(BaseEmbedding):
#     is_variable_length = True
#     pre_compute = False
#
#     def __init__(self,
#                  model_path: str,
#                  binary: bool = True,
#                  oov_handler: Optional[Callable] = None,
#                  batch_processor: Optional[Callable] = None):
#         """
#         Initialize Word2Vec embedding model with modular components.
#
#         Args:
#             model_path: Path to Word2Vec model file
#             binary: Whether model is in binary format
#             oov_handler: Function to handle out-of-vocabulary words
#             batch_processor: Custom batch processing function
#         """
#         self.model = self._load_model(model_path, binary)
#         self.vector_size = self.model.vector_size
#         self.oov_handler = oov_handler or self._default_oov_handler
#         self.batch_processor = batch_processor or self._default_batch_processor
#
#     # Model Loading
#     def _load_model(self, file_path: str, binary: bool) -> KeyedVectors:
#         """Load Word2Vec model from file"""
#         return KeyedVectors.load_word2vec_format(file_path, binary=binary)
#
#     # OOV Handling
#     def _default_oov_handler(self, word: str) -> np.ndarray:
#         """Default OOV strategy: zero vector"""
#         return np.zeros(self.vector_size)
#
#     def get_word_embedding(self, word: str) -> np.ndarray:
#         """Get embedding for single word with OOV handling"""
#         return self.model[word] if word in self.model else self.oov_handler(word)
#
#     # Sentence Processing
#     def _embed_sentence(self, sentence: str) -> np.ndarray:
#         """Convert single sentence to word embeddings"""
#         return np.array([self.get_word_embedding(word) for word in sentence.split()])
#
#     # Batch Processing
#     def _default_batch_processor(self, sentences: List[str], batch_size: int) -> List[np.ndarray]:
#         """Process sentences in batches with progress bar"""
#         results = []
#         for i in tqdm(range(0, len(sentences), batch_size)):
#             batch = sentences[i:i + batch_size]
#             results.extend([self._embed_sentence(sent) for sent in batch])
#         return results
#
#     # Public Interface
#     def transform(self, sentences: List[str]) -> List[np.ndarray]:
#         return self.get_word2vec_embeddings_batch(sentences)
#
#     def fit_transform(self, train_sentences: List[str]) -> List[np.ndarray]:
#         return self.get_word2vec_embeddings_batch(train_sentences)
#
#     def get_word2vec_embeddings_batch(self,
#                                       sentences: List[str],
#                                       batch_size: int = 32) -> List[np.ndarray]:
#         """Main batch processing method"""
#         return self.batch_processor(sentences, batch_size)