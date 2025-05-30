
from typing import List

from Hyperparameters.Embeddings.BaseEmbedding import BaseEmbedding
from Hyperparameters.Embeddings.BatchProcessor import BatchProcessor

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline

import torch

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
