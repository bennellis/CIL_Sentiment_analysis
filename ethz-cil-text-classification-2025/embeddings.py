from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel, pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Tuple
import re
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
import custom_dataloader


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


# Concrete implementations
class BOWEmbedding(VectorizerEmbedding):
    def __init__(self, **kwargs):
        super().__init__(CountVectorizer(**kwargs))


class TFIDFEmbedding(VectorizerEmbedding):
    def __init__(self, **kwargs):
        super().__init__(TfidfVectorizer(**kwargs))
    

class BertPreTrained(BaseEmbedding):
    is_variable_length = False
    pre_compute = False
    def __init__(self,model):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model_embed = BertModel.from_pretrained(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_embed = self.model_embed.to(self.device)
        self.batch_processor = BatchProcessor()

    def transform(self, sentences: List[str]):
        return self.get_bert_embeddings_batch(list(sentences))

    def fit_transform(self, train_sentences: List[str]):
        return self.get_bert_embeddings_batch(list(train_sentences))

    def _process_single_batch(self, batch):
        """Process a single batch of texts into BERT embeddings"""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model_embed(**inputs)
        # print(outputs)

        # Mean pooling across tokens
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def get_bert_embeddings_batch(self, texts, batch_size=32):
        """Get embeddings for all texts using batch processing"""
        return self.batch_processor.process_in_batches(
            data=texts,
            batch_size=batch_size,
            process_fn=self._process_single_batch
        )


class BertPreTrainedClassifier(BaseEmbedding):
    is_variable_length = False
    pre_compute = False
    def __init__(self,model):
        pipe = pipeline("text-classification", model=model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_embed = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3,ignore_mismatched_sizes=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_embed = self.model_embed.to(self.device)
        self.batch_processor = BatchProcessor()

    def transform(self, sentences: List[str]):
        return self.get_bert_embeddings_batch(list(sentences))

    def fit_transform(self, train_sentences: List[str]):
        return self.get_bert_embeddings_batch(list(train_sentences))

    def _process_single_batch(self, batch):
        """Process a single batch of texts into BERT embeddings"""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256 # was 512
        ).to(self.device)

        # training_args = TrainingArguments(...)  # Same as original
        # trainer = Trainer(...)  # Same as original
        # trainer.train()

        with torch.no_grad():
            outputs = self.model_embed(**inputs)
        # print(outputs)

        return outputs.logits.cpu().numpy()

    def get_bert_embeddings_batch(self, texts, batch_size=16):
        """Get embeddings for all texts using batch processing"""
        return self.batch_processor.process_in_batches(
            data=texts,
            batch_size=batch_size,
            process_fn=self._process_single_batch
        )


class BertTokenEmbedder(BaseEmbedding):
    is_variable_length = True
    pre_compute = True
    def __init__(self,model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.batch_processor = BatchProcessor()

    def transform(self, sentences: List[str]):
        return self.get_bert_embeddings_batch(list(sentences))

    def fit_transform(self, train_sentences: List[str]):
        return self.get_bert_embeddings_batch(list(train_sentences))

    def _process_single_batch(self, batch):
        """Process a single batch of texts into BERT embeddings"""
        encoding = self.tokenizer(
            batch,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256 # was 512
        ).to(self.device)
        ar = np.array([encoding['input_ids'].cpu().numpy(),encoding['attention_mask'].cpu().numpy()])
        return np.transpose(ar, (1, 0, 2))


    def get_bert_embeddings_batch(self, texts, batch_size=32):
        """Get embeddings for all texts using batch processing"""
        return self.batch_processor.process_in_batches(
            data=texts,
            batch_size=batch_size,
            process_fn=self._process_single_batch
        )

    def _unpack_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Unpack HF-formatted batch"""
        return (
            batch[0][:, 0].long(),
            batch[2],
            {'attention_mask': batch[0][:, 1].long().to(self.device)}
        )

    def precompute_embeddings(self, dataloader: DataLoader, val=False) -> DataLoader:
        """
        Runs every batch through BERT (in eval & no_grad mode),
        collects `pooler_output` into a TensorDataset, and returns
        a new DataLoader over (embeddings, labels).
        """
        self.model.eval()
        all_embs, all_labels = [], []

        pbar = tqdm(dataloader, desc=f"{'pre-computing'}",
                    unit='batch', leave=False)

        with torch.no_grad():
            for batch in pbar:
                x, y, kwargs = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                attention_mask = kwargs.get('attention_mask', None)
                if attention_mask is not None: attention_mask = attention_mask.to(self.device)

                # print(self.model)
                for attr in ("bert", "distilbert", "model", "roberta"): #This is to set the tokenizer correctly for different model architectures.
                    backbone = getattr(self.model, attr, None)
                    if backbone is not None:
                        break
                # backbone = getattr(self.model, "bert", None) or getattr(self.model, "distilbert", None) or getattr(self.model, "model",None)
                outputs = backbone( #.bert or distilbert
                    input_ids=x,
                    attention_mask=attention_mask
                )
                logits = outputs.last_hidden_state[:, 0] #.pooler_output  # shape (bsz, hidden_size)
                all_embs.append(logits.cpu())
                all_labels.append(y.cpu())

        embs = torch.cat(all_embs, dim=0).numpy()
        # embs = embs[:,np.newaxis, :]
        # print(embs.shape)
        labs = torch.cat(all_labels, dim=0).numpy()
        ds = custom_dataloader.EmbeddingDataset(embs, labs, variable_length=False)
        train_sampler = custom_dataloader.DynamicUnderSampler(labs, random_state=42)
        # reuse the same batch_size & shuffling as original

        if val:
            return DataLoader(
                ds,
                batch_size=16,
                collate_fn=custom_dataloader.collate_fn,
            )
        else:

            return DataLoader(
                ds,
                sampler=train_sampler,
                batch_size=16,
                collate_fn=custom_dataloader.collate_fn,
            )
    # return self._process_single_batch(texts)

# ***************************** Variable Length Embeddings *************************

class Word2VecEmbedding(BaseEmbedding):
    is_variable_length = True
    pre_compute = False
    
    def __init__(self, model_path, binary=True):
        """
        Initialize the Word2VecEmbedding class to load pre-trained Word2Vec model
        
        Args:
            model_path (str): Path to the pre-trained Word2Vec model file.
            binary (bool): Whether the model is in binary format (default: True)
        """
        self.model = self.load_word2vec_model(model_path, binary)
        self.vector_size = 300  # GoogleNews vectors have a size of 300
        
    def load_word2vec_model(self, file_path, binary=True):

        # word_vectors = {}
        
        # with open(file_path, 'rb' if binary else 'r') as f:
        #     if binary:
        #         # Read header for binary format (vocab_size and vector_size)
        #         header = f.readline()
        #         vocab_size, vector_size = map(int, header.split())
        #         for line in range(vocab_size):
        #             word = []
        #             while True:
        #                 char = f.read(1)
        #                 if char == b' ':
        #                     break
        #                 word.append(char)
        #             word = b''.join(word).decode('utf-8')
        #             vector = np.frombuffer(f.read(4 * vector_size), dtype=np.float32)
        #             word_vectors[word] = vector
        #     else:
        #         # Handle text format (space-separated values)
        #         for line in f:
        #             parts = line.split()
        #             word = parts[0]
        #             vector = np.array([float(x) for x in parts[1:]])
        #             word_vectors[word] = vector
                    
        # return word_vectors
        from gensim.models import KeyedVectors
        # Load the model
        return KeyedVectors.load_word2vec_format(file_path, binary=binary)
    
    def transform(self, sentences: List[str]):
        return self.get_word2vec_embeddings_batch(sentences)
    
    def fit_transform(self, train_sentences: List[str]):

        return self.get_word2vec_embeddings_batch(train_sentences)
    
    def get_word2vec_embeddings_batch(self, sentences, batch_size=32):
        """
        Get Word2Vec embeddings for a batch of sentences
        
        Args:
            sentences (list): List of sentences to embed
            batch_size: Number of sentences to process at once
        
        Returns:
            list[list[numpy.ndarray]]: List of word embeddings for each sentence
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i+batch_size]
            batch_word_embeddings = []
            
            for sentence in batch:
                word_embeddings = []
                
                # Get embeddings for each word in the sentence
                for word in sentence.split():
                    if word in self.model:
                        word_embeddings.append(self.model[word])
                    else:
                        # If the word isn't in the vocabulary, return a zero vector (or handle it differently)
                        word_embeddings.append(np.zeros(self.vector_size))
                
                batch_word_embeddings.append(np.array(word_embeddings))
            
            all_embeddings.extend(batch_word_embeddings)
        
        return all_embeddings


class Word2VecEmbedding2(BaseEmbedding):
    is_variable_length = True
    pre_compute = False

    def __init__(self,
                 model_path: str,
                 binary: bool = True,
                 oov_handler: Optional[Callable] = None,
                 batch_processor: Optional[Callable] = None):
        """
        Initialize Word2Vec embedding model with modular components.

        Args:
            model_path: Path to Word2Vec model file
            binary: Whether model is in binary format
            oov_handler: Function to handle out-of-vocabulary words
            batch_processor: Custom batch processing function
        """
        self.model = self._load_model(model_path, binary)
        self.vector_size = self.model.vector_size
        self.oov_handler = oov_handler or self._default_oov_handler
        self.batch_processor = batch_processor or self._default_batch_processor

    # Model Loading
    def _load_model(self, file_path: str, binary: bool) -> KeyedVectors:
        """Load Word2Vec model from file"""
        return KeyedVectors.load_word2vec_format(file_path, binary=binary)

    # OOV Handling
    def _default_oov_handler(self, word: str) -> np.ndarray:
        """Default OOV strategy: zero vector"""
        return np.zeros(self.vector_size)

    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get embedding for single word with OOV handling"""
        return self.model[word] if word in self.model else self.oov_handler(word)

    # Sentence Processing
    def _embed_sentence(self, sentence: str) -> np.ndarray:
        """Convert single sentence to word embeddings"""
        return np.array([self.get_word_embedding(word) for word in sentence.split()])

    # Batch Processing
    def _default_batch_processor(self, sentences: List[str], batch_size: int) -> List[np.ndarray]:
        """Process sentences in batches with progress bar"""
        results = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            results.extend([self._embed_sentence(sent) for sent in batch])
        return results

    # Public Interface
    def transform(self, sentences: List[str]) -> List[np.ndarray]:
        return self.get_word2vec_embeddings_batch(sentences)

    def fit_transform(self, train_sentences: List[str]) -> List[np.ndarray]:
        return self.get_word2vec_embeddings_batch(train_sentences)

    def get_word2vec_embeddings_batch(self,
                                      sentences: List[str],
                                      batch_size: int = 32) -> List[np.ndarray]:
        """Main batch processing method"""
        return self.batch_processor(sentences, batch_size)