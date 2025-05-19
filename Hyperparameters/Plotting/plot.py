import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
import plotly.io as pio
pio.renderers.default = "browser"  # Ensure browser is used

from Hyperparameters.Embeddings.BertTokenEmbedder import BertTokenEmbedder
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn

from Hyperparameters.Plotting.PlotEmbeddings import extract_all_embeddings_and_labels, datashader_plot, project_umap, \
    project_tsne, project_pca, plotly_scatter_plot, generate_random_embeddings

seed = 42

## Model Parameters
prefix = "../"
model_name="FacebookAI/roberta-base"
csv_path= prefix + "data/Sentiment/training.csv"


embedder = BertTokenEmbedder(model_name)


embedded_feature_dataset = None

if embedder.is_variable_length:

    cache_name = model_name.replace("/", "_")
    cache_path = prefix + "cache/" + cache_name
    emb_dataset_path = cache_path + "emb_dataset.pt"

    if os.path.exists(emb_dataset_path):

        embedded_feature_dataset = torch.load(emb_dataset_path, weights_only=False)
    else:

        df = pd.read_csv(csv_path, index_col=0)
        label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['label_encoded'] = df['label'].map(label_map)

        features = embedder.fit_transform(df['sentence'].to_list())
        labels = df['label_encoded'].to_numpy()

        feature_dataset = EmbeddingDataset(features, labels)

        feature_dataloader = DataLoader(feature_dataset, batch_size=8, collate_fn=collate_fn)
        embedded_feature_dataset = embedder.embed_dataset(feature_dataloader)
        os.makedirs("cache", exist_ok=True)
        torch.save(embedded_feature_dataset, emb_dataset_path)

else:
    raise Exception("blaalalal")


embs, labels = extract_all_embeddings_and_labels(embedded_feature_dataset)


embs_2d, idx = project_umap(embs)
fig = plotly_scatter_plot(embs_2d, labels[idx])

fig.show()

print("here")
