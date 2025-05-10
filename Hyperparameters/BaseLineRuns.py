import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, confusion_matrix

from Hyperparameters.Dataloader.DynamicUnderSampler import DynamicUnderSampler
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn
from Hyperparameters.Embeddings.BertTokenEmbedder import BertTokenEmbedder
from Hyperparameters.Models.BertPreTrainedClassifier import BertPreTrainedClassifier

random_seed = 42
bert_model = "answerdotai/ModernBERT-base"
load_from_path = False
model_path = "roberta_875_04-27"
output_file = "test_predictions.csv"


def main():

    # Load Data
    training_data = pd.read_csv('data/training.csv', index_col=0)
    label_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    training_data['label_encoded'] = training_data['label'].map(label_mapping)

    # Create Validation Set

    sentences = training_data['sentence']
    labels = training_data['label_encoded']

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences,
                                                                                labels,
                                                                                test_size=0.1,
                                                                                stratify=labels,
                                                                                random_state=random_seed
                                                                                )
    train_sentences, val_sentences = list(train_sentences), list(val_sentences)
    train_labels, val_labels = np.array(train_labels), np.array(val_labels)

    ## Sample stuff

    train_data = list(zip(train_sentences, train_labels))

    # Separate by class
    class_minus1 = [x for x in train_data if x[1] == -1]
    class_0 = [x for x in train_data if x[1] == 0]
    class_1 = [x for x in train_data if x[1] == 1]

    min_size = min(len(class_minus1), len(class_0), len(class_1))

    class_minus1_bal = resample(class_minus1,
                                replace=False,
                                n_samples=min_size,
                                random_state=random_seed
                                )
    class_0_bal = resample(class_0,
                           replace=False,
                           n_samples=min_size,
                           random_state=random_seed
                           )
    class_1_bal = resample(class_1,
                           replace=False,
                           n_samples=min_size,
                           random_state=random_seed
                           )

    # Combine and shuffle
    balanced_train_data = class_minus1_bal + class_0_bal + class_1_bal
    np.random.shuffle(balanced_train_data)

    # Split back into sentences and labels
    train_sentences_bal, train_labels_bal = zip(*balanced_train_data)
    train_sentences_bal, train_labels_bal = list(train_sentences_bal), np.array(train_labels_bal)


    # Create Embedding Model
    embedding_model = BertTokenEmbedder(bert_model)
    n_samples = -1  # set to -1 to get all samples

    n_val_samples = int(n_samples / 10) if n_samples != -1 else -1
    X_train = embedding_model.fit_transform(train_sentences[:n_samples])  # For quick testing with less data
    X_val = embedding_model.transform(val_sentences[:n_val_samples])

    Y_train = train_labels[:n_samples]
    Y_val = val_labels[:n_val_samples]

    train_sampler = DynamicUnderSampler(Y_train, random_state=42)
    pre_compute = True
    if embedding_model.is_variable_length:
        dataset_train = EmbeddingDataset(X_train, Y_train)
        dataset_val = EmbeddingDataset(X_val, Y_val)
        # print(len(dataset_train))
        train_loader = DataLoader(
            dataset_train,
            sampler=train_sampler,
            batch_size=8,
            collate_fn=collate_fn,
        )
        train_loader_pred = DataLoader(
            dataset_train,
            batch_size=64,
            collate_fn=collate_fn,
        )
        train_loader_pred_shuf = DataLoader(
            dataset_train,
            batch_size=16,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=64,
            collate_fn=collate_fn
        )

        if pre_compute:
            emb_loader = embedding_model.precompute_embeddings(train_loader_pred)
            emb_val_loader = embedding_model.precompute_embeddings(val_loader, val=True)

    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), sampler=train_sampler, batch_size=32)
        train_loader_pred = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    model = BertPreTrainedClassifier(bert_model, lr=5e-05, frozen=True, class_order=[0, 1, 2],
                                            dropout=0.4, ce_weight=0.1, temperature=0.5, custom_ll=True,
                                            pt_lr_top=5e-05, pt_lr_mid=5e-06, pt_lr_bot=5e-07)


    ## Load and train the model
    if load_from_path:
        model_path = "models/distilbert_product_855_04-21.pt"
        print("Loading model from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))

    # print(X_train.shape)
    if model.use_dataloader:
        if model.frozen:
            model.fit(emb_loader, emb_val_loader, epochs=6, )
        else:
            model.fit(train_loader, val_loader, epochs=2)  # train_loader_pred_shuf has no class balancing
        # model.fit(train_loader,epochs=10)
    else:
        model.fit(X_train, Y_train)

    layers_to_freeze = 6  # set how many layers from the bottom you want to leave frozen when fine-tuning. 0 to unfreeze everything.

    for attr in ("encoder", "transformer",
                 "layers"):  # This is to set the "model layer" correctly for different model architectures.
        backbone = getattr(model.model, attr, None)
        if backbone is not None:
            if attr in ("encoder", "transformer"):
                backbone = backbone.layer
            break

    if model.frozen:
        print("model unfrozen")
        model.frozen = False
        for param in model.model.parameters():  # unfreeze all params
            param.requires_grad = True
        for param in model.model.embeddings.parameters():  # re-freeze the embedding layers (maybe we don't want this? Haven't tried yet)
            param.requires_grad = False
        for i in range(layers_to_freeze):  # freezes bottom most layers
            for param in backbone[i].parameters():
                param.requires_grad = False
    else:  # Toggle frozen again
        print("model frozen")
        model.frozen = True
        for param in model.model.parameters():
            param.requires_grad = False


    # Do validation


    if model.use_dataloader:
        Y_val_pred = model.predict(val_loader)

    else:
        Y_val_pred = model.predict(X_val)

    mae_val = mean_absolute_error(Y_val, Y_val_pred)
    L_score_val = 0.5 * (2 - mae_val)

    print(f'Evaluation Score (validation set): {L_score_val:.05f}')

    conf_matrix = confusion_matrix(Y_val, Y_val_pred, labels=[-1, 0, 1])
    print(conf_matrix)

    # save results

    torch.save(model.state_dict(), "models/" + model_path + ".pt")
    model.model.save_pretrained("models_pretrained/" + model_path)
    model.model.config.save_pretrained("configs/" + model_path)
    model.tokenizer.save_pretrained("models_pretrained/" + model_path)


    # run on test data

    test_data = pd.read_csv('data/test.csv', index_col=0)
    X_test = embedding_model.transform(test_data['sentence'])
    Y_test_fake_labels = np.ones(
        X_test.shape[0])  # This is just for the dataloader, too lazy to make it work without labels.

    if embedding_model.is_variable_length:
        dataset_test = EmbeddingDataset(X_test, Y_test_fake_labels)
        # print(len(dataset_train))
        test_loader = DataLoader(
            dataset_test,
            batch_size=64,
            collate_fn=collate_fn,
        )
    else:
        if hasattr(X_test, 'toarray'):
            X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        else:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test_tensor, X_test_tensor), batch_size=32)

    # print(X_test[0].shape)

    if model.use_dataloader:
        y_test = model.predict(test_loader)
    else:
        y_test = model.predict(X_test)

    y_labels = pd.Series(y_test).map({-1: 'negative', 0: 'neutral', 1: 'positive'})
    submission = pd.DataFrame({'id': test_data.index, 'label': y_labels})
    submission.to_csv(output_file, index=False)  # Update filename and path as needed
    print(f"Test predictions saved to {output_file}")


    ## Test Examples

    number_examples = 5
    label_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}

    misclassified = [
        (label_map[true.item()], label_map[pred.item()], text)
        for true, pred, text in zip(Y_val, Y_val_pred, val_sentences)
        if true != pred
    ]

    for true, pred, text in random.sample(misclassified, number_examples):
        print(f"True: {true}, Pred: {pred} → {text}")

if __name__ == "__main__":
    main()