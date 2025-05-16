import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from torch.utils.data import DataLoader, TensorDataset

from Hyperparameters.Embeddings.BertTokenEmbedder import BertTokenEmbedder
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn
from Hyperparameters.Models.BertPreTrainedClassifier import BertPreTrainedClassifier
from Hyperparameters.Training.ActiveLearningLoop import active_learning_loop
from Hyperparameters.Training.ActiveLearningLoop import query_entropy

from Hyperparameters.Utils.Misc import get_device


model_name="FacebookAI/roberta-large"
csv_path="data/Sentiment/training.csv"
seed = 42

lr = 1e-3
class_order = [0,1,2]
lr_top = 5e-5
lr_mid = 3e-5
lr_bot = 2e-5
dropout = 0.4
temperature = 0.5
ce_weight = 0.1

df = pd.read_csv(csv_path, index_col=0)
label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
df['label_encoded'] = df['label'].map(label_map)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'], df['label_encoded'],
    stratify=df['label_encoded'], test_size=0.1, random_state=seed
)
embedder = BertTokenEmbedder(model_name)
features = embedder.fit_transform(df['sentence'].to_list())
labels = df['label_encoded'].to_numpy()

if embedder.is_variable_length:
    feature_dataset = EmbeddingDataset(features, labels)

    cache_name= model_name.replace("/", "_")
    cache_path = "cache/" + cache_name
    emb_dataset_path = cache_path + "emb_dataset.pt"


    if os.path.exists(emb_dataset_path):
        embedded_feature_dataset = torch.load(emb_dataset_path, weights_only=False)
    else:
        feature_dataloader = DataLoader(feature_dataset, batch_size=8,collate_fn=collate_fn)
        embedded_feature_dataset = embedder.embed_dataset(feature_dataloader)
        os.makedirs("cache", exist_ok=True)
        torch.save(embedded_feature_dataset, emb_dataset_path)

else:
    raise Exception("blaalalal")

model = BertPreTrainedClassifier(
    model_name = model_name,
    lr = lr,
    pt_lr_bot = lr_bot,
    pt_lr_mid = lr_mid,
    pt_lr_top = lr_top,
    class_order = class_order,
    ce_weight = ce_weight,
    temperature = temperature,
    frozen = True,
    custom_ll = True
)



active_learning_loop(
        model,
        get_device(),
        embedded_feature_dataset,
        query_entropy,
        max_rounds=1000,
        query_batch_size=1000,
        train_epochs_per_round=3,
        initial_label_count=1000,
        val_split=0.2,
        batch_size=32
)

model_path = "roberta_large_active_loss"
torch.save(model.state_dict(), "cache/" + model_path + ".pt")
model.model.save_pretrained("cache/" + model_path + "pretrained")
model.model.config.save_pretrained("cache/" + model_path + "_config")
model.tokenizer.save_pretrained("cache/" + model_path + "tokenizer")