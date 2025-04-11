
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    @abstractmethod
    def fit_transform(self, train_sentences):
        pass

    @abstractmethod
    def transform(self, sentences):
        pass


class BOW_baseline(BaseEmbedding):
    def __init__(self, ngram_range=(1, 2), max_features=10000, stop_words=None, min_df=1, max_df=1.0):
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df
        )
        self._is_fitted = False

    def transform(self, sentences):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
        return self.vectorizer.transform(sentences)

    def fit_transform(self, train_sentences):
        self._is_fitted = True
        return self.vectorizer.fit_transform(train_sentences)
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    

class TFIDF_Embedding(BaseEmbedding):
    def __init__(self, ngram_range=(1, 2), max_features=10000, stop_words=None, min_df=1, max_df=1.0):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df
        )
        self._is_fitted = False

    def fit_transform(self, train_sentences):
        self._is_fitted = True
        return self.vectorizer.fit_transform(train_sentences)

    def transform(self, sentences):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit_transform() first.")
        return self.vectorizer.transform(sentences)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class Other_BOW_Embedding(BaseEmbedding):
    def __init__(self, ngram_range=(1, 2), max_features=10000, stop_words='english', min_df=10, max_df=0.9):
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df
        )
        self._is_fitted = False

    def transform(self, sentences):
        if not self._is_fitted:
            raise ValueError("Vectorizer is not fitted yet. Call fit() first.")
        return self.vectorizer.transform(sentences)

    def fit_transform(self, train_sentences):
        self._is_fitted = True
        return self.vectorizer.fit_transform(train_sentences)
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    

class Bert_base_uncased(BaseEmbedding):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model_embed = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_embed = self.model_embed.to(self.device)

    def transform(self, sentences):
        return self.get_bert_embeddings_batch(sentences.tolist())

    def fit_transform(self, train_sentences):
        return self.get_bert_embeddings_batch(train_sentences.tolist())
    
    def get_bert_embeddings_batch(self, texts, batch_size=32):
        """
        Get BERT embeddings for a batch of texts
        
        Args:
            texts (list): List of texts to embed
            batch_size: Number of texts to process at once
        
        Returns:
            numpy.ndarray: Embeddings for all input texts
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model_embed(**inputs)
        
            last_hidden_states = outputs.last_hidden_state
            # print(embeddings.shape)
            batch_embeddings = last_hidden_states.mean(dim=1).cpu().numpy() # this is to mean pool
            
            # all_embeddings.append(embeddings.cpu().numpy())
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.concatenate(all_embeddings, axis=0)
        # return all_embeddings
