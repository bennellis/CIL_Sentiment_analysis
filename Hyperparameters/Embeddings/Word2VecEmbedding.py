#
# from typing import List
#
# from Hyperparameters.Embeddings.BaseEmbedding import BaseEmbedding
#
# import numpy as np
# from tqdm.auto import tqdm
# from gensim.models import KeyedVectors
#
# class Word2VecEmbedding(BaseEmbedding):
#     is_variable_length = True
#     pre_compute = False
#
#     def __init__(self, model_path, binary=True):
#         """
#         Initialize the Word2VecEmbedding class to load pre-trained Word2Vec model
#
#         Args:
#             model_path (str): Path to the pre-trained Word2Vec model file.
#             binary (bool): Whether the model is in binary format (default: True)
#         """
#         self.model = self.load_word2vec_model(model_path, binary)
#         self.vector_size = 300  # GoogleNews vectors have a size of 300
#
#     def load_word2vec_model(self, file_path, binary=True):
#
#         # word_vectors = {}
#
#         # with open(file_path, 'rb' if binary else 'r') as f:
#         #     if binary:
#         #         # Read header for binary format (vocab_size and vector_size)
#         #         header = f.readline()
#         #         vocab_size, vector_size = map(int, header.split())
#         #         for line in range(vocab_size):
#         #             word = []
#         #             while True:
#         #                 char = f.read(1)
#         #                 if char == b' ':
#         #                     break
#         #                 word.append(char)
#         #             word = b''.join(word).decode('utf-8')
#         #             vector = np.frombuffer(f.read(4 * vector_size), dtype=np.float32)
#         #             word_vectors[word] = vector
#         #     else:
#         #         # Handle text format (space-separated values)
#         #         for line in f:
#         #             parts = line.split()
#         #             word = parts[0]
#         #             vector = np.array([float(x) for x in parts[1:]])
#         #             word_vectors[word] = vector
#
#         # return word_vectors
#
#         # Load the model
#         return KeyedVectors.load_word2vec_format(file_path, binary=binary)
#
#     def transform(self, sentences: List[str]):
#         return self.get_word2vec_embeddings_batch(sentences)
#
#     def fit_transform(self, train_sentences: List[str]):
#
#         return self.get_word2vec_embeddings_batch(train_sentences)
#
#     def get_word2vec_embeddings_batch(self, sentences, batch_size=32):
#         """
#         Get Word2Vec embeddings for a batch of sentences
#
#         Args:
#             sentences (list): List of sentences to embed
#             batch_size: Number of sentences to process at once
#
#         Returns:
#             list[list[numpy.ndarray]]: List of word embeddings for each sentence
#         """
#         all_embeddings = []
#
#         # Process in batches
#         for i in tqdm(range(0, len(sentences), batch_size)):
#             batch = sentences[i:i + batch_size]
#             batch_word_embeddings = []
#
#             for sentence in batch:
#                 word_embeddings = []
#
#                 # Get embeddings for each word in the sentence
#                 for word in sentence.split():
#                     if word in self.model:
#                         word_embeddings.append(self.model[word])
#                     else:
#                         # If the word isn't in the vocabulary, return a zero vector (or handle it differently)
#                         word_embeddings.append(np.zeros(self.vector_size))
#
#                 batch_word_embeddings.append(np.array(word_embeddings))
#
#             all_embeddings.extend(batch_word_embeddings)
#
#         return all_embeddings