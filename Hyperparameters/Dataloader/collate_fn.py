
import torch


def collate_fn(batch):
    """
    Custom collate function to handle:
    - Variable-length embeddings
    - Corresponding labels
    Returns:
        padded_embeddings: Padded sequence tensor (batch_size, max_seq_len, embedding_dim)
        lengths: Original lengths of each sequence
        labels: Batch of labels
    """
    # Separate embeddings and labels
    embeddings = [item['embeddings'] for item in batch]
    labels = [item['label'] for item in batch]

    # Sort by sequence length (descending)
    sorted_indices = sorted(
        range(len(embeddings)),
        key=lambda i: embeddings[i].shape[0],
        reverse=True
    )
    embeddings = [embeddings[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Pad sequences
    lengths = torch.tensor([len(x) for x in embeddings])
    padded_embeddings = torch.nn.utils.rnn.pad_sequence(
        embeddings,
        batch_first=True,
        padding_value=0
    )

    # Stack labels
    labels = torch.stack(labels)

    return padded_embeddings, lengths, labels