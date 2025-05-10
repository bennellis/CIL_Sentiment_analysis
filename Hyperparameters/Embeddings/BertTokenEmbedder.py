
from typing import List, Tuple

from torch.utils.data import DataLoader

from Hyperparameters.Dataloader.DynamicUnderSampler import DynamicUnderSampler
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn
from Hyperparameters.Embeddings.BaseEmbedding import BaseEmbedding
from Hyperparameters.Embeddings.BatchProcessor import BatchProcessor

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import numpy as np
from tqdm.auto import tqdm

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
        ds = EmbeddingDataset(embs, labs, variable_length=False)
        train_sampler = DynamicUnderSampler(labs, random_state=42)
        # reuse the same batch_size & shuffling as original

        if val:
            return DataLoader(
                ds,
                batch_size=16,
                collate_fn=collate_fn,
            )
        else:

            return DataLoader(
                ds,
                sampler=train_sampler,
                batch_size=16,
                collate_fn=collate_fn,
            )
    # return self._process_single_batch(texts)