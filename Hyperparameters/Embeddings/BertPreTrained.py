
from typing import List

from Hyperparameters.Embeddings.BaseEmbedding import BaseEmbedding
from Hyperparameters.Embeddings.BatchProcessor import BatchProcessor

import torch
from transformers import BertTokenizer, BertModel



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