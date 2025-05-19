import numpy as np
import umap
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.graph_objects as go

def extract_all_embeddings_and_labels(dataset):
    all_embeddings = []
    all_labels = []

    for item in dataset:
        all_embeddings.append(item['embeddings'].numpy())
        all_labels.append(item['label'])

    return np.stack(all_embeddings), np.array(all_labels)


def project_umap(embeddings, subset_size=None, random_state=42, **umap_kwargs):
    """
    Projects high-dimensional embeddings to 2D using UMAP.

    Args:
        embeddings (np.ndarray): The full embedding matrix (shape: [N, D]).
        subset_size (int, optional): If provided, randomly sample this many points before projection.
        random_state (int): For reproducible sampling (not for umap to allow for parallelism)
        umap_kwargs: Extra args passed to UMAP.

    Returns:
        (np.ndarray, np.ndarray): (2D projected subset, indices of subset in original array)
    """
    if subset_size is not None and subset_size < len(embeddings):
        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(len(embeddings), size=subset_size, replace=False)
        subset = embeddings[indices]
    else:
        indices = np.arange(len(embeddings))
        subset = embeddings

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, **umap_kwargs)
    projection = reducer.fit_transform(subset)

    return projection, indices


def project_pca(embeddings, subset_size=None, random_state=42, **pca_kwargs):
    """
    Projects embeddings to 2D using PCA.

    Returns:
        (2D projection, selected_indices)
    """
    if subset_size is not None and subset_size < len(embeddings):
        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(len(embeddings), size=subset_size, replace=False)
        subset = embeddings[indices]
    else:
        indices = np.arange(len(embeddings))
        subset = embeddings

    reducer = PCA(n_components=2, **pca_kwargs)
    projection = reducer.fit_transform(subset)

    return projection, indices




def project_tsne(embeddings, subset_size=None, random_state=42, **tsne_kwargs):
    """
    Projects embeddings to 2D using t-SNE.

    Returns:
        (2D projection, selected_indices)
    """
    if subset_size is not None and subset_size < len(embeddings):
        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(len(embeddings), size=subset_size, replace=False)
        subset = embeddings[indices]
    else:
        indices = np.arange(len(embeddings))
        subset = embeddings

    reducer = TSNE(n_components=2, **tsne_kwargs)
    projection = reducer.fit_transform(subset)

    return projection, indices



def datashader_plot(points_2d, labels, cmap='viridis'):
    df = pd.DataFrame({
        'x': points_2d[:, 0],
        'y': points_2d[:, 1],
        'label': labels
    })

    # ✅ Convert to categorical
    df['label'] = pd.Categorical(df['label'])

    canvas = ds.Canvas(plot_width=1024, plot_height=1024)
    agg = canvas.points(df, 'x', 'y', ds.count_cat('label'))

    img = tf.shade(agg, how='eq_hist')
    tf.set_background(img, "black").to_pil().show()



def generate_random_embeddings(n_points=1000, n_classes=3, seed=42):
    """
    Generate synthetic 2D embeddings and class labels for testing.

    Args:
        n_points (int): Total number of data points.
        n_classes (int): Number of distinct class labels.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: 2D points (n_points, 2)
        np.ndarray: Corresponding labels (n_points,)
    """
    np.random.seed(seed)
    points = []
    labels = []

    for i in range(n_classes):
        center = np.random.uniform(-10, 10, size=2)
        cluster = center + np.random.randn(n_points // n_classes, 2)
        points.append(cluster)
        labels.extend([i] * (n_points // n_classes))

    points = np.vstack(points)
    labels = np.array(labels)
    return points, labels


# def plotly_scatter_plot(points_2d, labels, hover_text=None, title="2D Projection"):
#     df = pd.DataFrame({
#         'x': points_2d[:, 0],
#         'y': points_2d[:, 1],
#         'label': labels
#     })
#
#     if hover_text is not None:
#         df['hover'] = hover_text
#     else:
#         df['hover'] = df['label'].astype(str)
#
#     # Ensure labels are categorical for grouping
#     df['label'] = df['label'].astype(str)
#
#     # Create one trace per label (class)
#     traces = []
#     for label in df['label'].unique():
#         class_df = df[df['label'] == label]
#         trace = go.Scattergl(  # Scattergl for better performance with large data
#             x=class_df['x'],
#             y=class_df['y'],
#             mode='markers',
#             name=str(label),
#             text=class_df['hover'],
#             hoverinfo='text',
#             marker=dict(size=5, opacity=0.6),
#         )
#         traces.append(trace)
#
#     layout = go.Layout(
#         title=title,
#         width=800,
#         height=800,
#         legend=dict(title='Class'),
#         margin=dict(l=20, r=20, t=40, b=20),
#         plot_bgcolor='white'
#     )
#
#     fig = go.Figure(data=traces, layout=layout)
#     return fig



def plotly_scatter_plot(points_2d, labels, hover_text=None, title="2D Projection", show=True):
    df = pd.DataFrame({
        'x': points_2d[:, 0],
        'y': points_2d[:, 1],
        'label': labels
    })

    if hover_text is not None:
        df['hover'] = hover_text
    else:
        df['hover'] = df['label'].astype(str)

    df['label'] = df['label'].astype(str)
    unique_labels = sorted(df['label'].unique())

    traces = []
    for label in unique_labels:
        class_df = df[df['label'] == label]
        trace = go.Scattergl(
            x=class_df['x'],
            y=class_df['y'],
            mode='markers',
            name=label,
            text=class_df['hover'],
            hoverinfo='text',
            marker=dict(size=5, opacity=0.6),
            visible=True  # will manage visibility via dropdown
        )
        traces.append(trace)

    # Create dropdown buttons
    dropdown_buttons = [
        {
            'label': 'All',
            'method': 'update',
            'args': [{'visible': [True] * len(unique_labels)},
                     {'title': title}]
        }
    ]

    for i, label in enumerate(unique_labels):
        visibility = [j == i for j in range(len(unique_labels))]
        dropdown_buttons.append(
            {
                'label': label,
                'method': 'update',
                'args': [{'visible': visibility},
                         {'title': f"{title} - {label}"}]
            }
        )

    layout = go.Layout(
        title=title,
        width=800,
        height=800,
        legend=dict(title='Class'),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.15,
            'xanchor': 'left',
            'y': 1,
            'yanchor': 'top'
        }]
    )

    fig = go.Figure(data=traces, layout=layout)



    return fig

