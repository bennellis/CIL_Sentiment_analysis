import re
import string
import contractions
#import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#import spacy

class Preprocessor:
    def __init__(self,
                 lowercase=False,
                 remove_urls=False,
                 expand_contractions=False,  # can't -> can not
                 remove_punctuation=False,
                 remove_stopwords=False,
                 normalize_repeated_chars=False,  # sooooo -> soo
                 lemmatize=False,
                 remove_numbers=False,
                 remove_extra_whitespace=True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.expand_contractions = expand_contractions
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.normalize_repeated_chars = normalize_repeated_chars
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.expand_contractions:
            text = contractions.fix(text)

        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|ftp\S+", "", text)

        if self.normalize_repeated_chars:
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # "soooo" -> "soo"

        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.remove_stopwords:
            tokens = text.split()
            tokens = [word for word in tokens if word not in self.stop_words]
            text = " ".join(tokens)

        if self.lemmatize:
            text = " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])


        if self.remove_extra_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def transform(self, texts):
        cleaned = []
        kept_indices = []

        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            if isinstance(cleaned_text, str) and cleaned_text.strip():
                cleaned.append(cleaned_text)
                kept_indices.append(i)

        # return indices for labels
        return cleaned, kept_indices
