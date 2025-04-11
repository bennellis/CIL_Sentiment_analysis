
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    @abstractmethod
    def fit_transform(self, train_sentences):
        pass

    @abstractmethod
    def transform(self, sentences):
        pass

    @abstractmethod
    def get_feature_names_out(self):
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
