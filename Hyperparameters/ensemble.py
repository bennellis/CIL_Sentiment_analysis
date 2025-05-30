import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import os

import mlflow
import optuna
from sklearn.metrics import mean_absolute_error, confusion_matrix

from Hyperparameters.Dataloader.DynamicUnderSampler import DynamicUnderSampler
from Hyperparameters.Dataloader.EmbeddingDataset import EmbeddingDataset
from Hyperparameters.Dataloader.collate_fn import collate_fn
from Hyperparameters.Embeddings.BertTokenEmbedder import BertTokenEmbedder
from Hyperparameters.Models.BertPreTrainedClassifier import BertPreTrainedClassifier
from Hyperparameters.Utils.Misc import get_device

from Hyperparameters.data.preprocessing import Preprocessor


def prepare_data(csv_path, model_name, head):
    """Prepares the data for the ensembling"""
    df = pd.read_csv(csv_path, index_col=0)
    label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['label_encoded'] = df['label'].map(label_map)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['sentence'], df['label_encoded'],
        stratify=df['label_encoded'], test_size=0.1, random_state=42
    )

    NUM_SAMPLES = -1
    NUM_VAR_SAMPLES = -1 if NUM_SAMPLES == -1 else int(NUM_SAMPLES / 10)
    # Create embedding model & features
    embedder = BertTokenEmbedder(model_name, head=head)
    # X_train = embedder.fit_transform(list(train_texts)[:NUM_SAMPLES])
    X_val = embedder.transform(list(val_texts)[:NUM_VAR_SAMPLES])

    # Y_train = np.array(train_labels)[:NUM_SAMPLES]
    Y_val = np.array(val_labels)[:NUM_VAR_SAMPLES]

    return embedder, (X_val, Y_val), list(val_texts)

def main(do_test:bool = False, do_validate:bool = True):
    """Runs the ensemble with testing or validation or both. Required that
    the user inputs the models they want to test in the following lists:
    model_names: name of the underlying encoder model used as the backbone of the model
    ensemble_list: list of indexes of the models you want to test
    model_vscores: scores of models from validation set (used for weighted ensemble)
    model_paths: paths to the weights of the models you want to ensemble
    heads: classification head of that model. Either mlp, cnn, or rnn
    """

    # these are model parameters to load the model with, only used for training so doesn't matter
    d_params = {"lr": 1e-4,"pt_lr_top": 1e-5,"pt_lr_mid": 5e-6,"pt_lr_bot": 3e-6,"dropout": 0.4,
                "temperature": 1.0,"ce_weight": 0.2,"margin": 0,"use_cdw": True}

    # names of the underlying encoder model used in the ensemble
    model_names = ['distilbert/distilbert-base-uncased', 'FacebookAI/roberta-base',
                   'google-bert/bert-base-uncased','FacebookAI/roberta-base',
                   'microsoft/deberta-v3-base','answerdotai/ModernBERT-base',
                   'microsoft/deberta-v3-base','FacebookAI/roberta-base',
                   'microsoft/deberta-v3-base','answerdotai/ModernBERT-base',
                   'FacebookAI/roberta-base','microsoft/deberta-v3-base',
                   'answerdotai/ModernBERT-base',
                   ]
    # names of the underlying encoder model used in the ensemble
    params = [d_params] * len(model_names)
    ensemble_list = [0]  # use this to identify which model indexes to ensemble for this run
    # ensemble_list = range(len(model_names)) # use this to run all models


    #this list is for weighting the different models in the ensemble. Right now using a scaling based on validation score
    model_vscores = np.array([0.855, 0.875, 0.864, 0.882, 0.902, 0.889, 0.904, 0.889, 0.904, 0.890, 0.889, 0.904, 0.890])
    model_vscores = model_vscores[ensemble_list] # This is if you are only running a subset of models
    max_score = max(model_vscores)
    model_weights = [(vscore - 0.8) / (max_score - 0.8) for vscore in model_vscores]
    t_weights = torch.tensor(model_weights).view(-1, 1, 1)
    #Path to model saved weights
    model_paths = ['saved_weights/distilbert/distilbert-base-uncased/baseline_no_warmup.pt',
                   'saved_weights/roberta_875_04-27.pt',
                   'saved_weights/google-bert/bert-base-uncased/baseline_1.pt',
                   'saved_weights/FacebookAI/roberta-base/baseline_1.pt',
                   'saved_weights/microsoft/deberta-v3-base/baseline_1.pt',
                   'saved_weights/answerdotai/ModernBERT-base/baseline_1.pt',
                   'saved_weights/microsoft/deberta-v3-base/baseline_cnn_plus_augmented.pt',
                   'saved_weights/FacebookAI/roberta-base/baseline_plus_augmented.pt',
                   'saved_weights/microsoft/deberta-v3-base/baseline_plus_augmented.pt',
                   'saved_weights/answerdotai/ModernBERT-base/baseline_plus_augmented.pt',
                   'saved_weights/FacebookAI/roberta-base/baseline_plus_augmentedbest_loss.pt',
                   'saved_weights/microsoft/deberta-v3-base/baseline_plus_augmentedbest_loss.pt',
                   'saved_weights/answerdotai/ModernBERT-base/baseline_plus_augmentedbest_loss.pt',
                   ]
    #classification head used, either mlp, cnn, or rnn
    heads = ['mlp','mlp','mlp','mlp','mlp','mlp','cnn','mlp','mlp','mlp','mlp','mlp','mlp']

    predictions = []
    test_predictions = []
    hard_predictions = []
    test_hard_predictions = []
    final_Y_val =  None
    for i in ensemble_list:
        #for each model, load it, and then validate and / or test it. save individual results to file and also for ensemble.
        model = BertPreTrainedClassifier(model_name=model_names[i],frozen=False,
                                                     **(params[i]), head = heads[i])
        print("Loading model from {}".format(model_paths[i]))
        model.load_state_dict(torch.load(model_paths[i]))

        embedder = BertTokenEmbedder(model_names[i], head=heads[i])

        if do_validate:
            _, val_data, val_texts = prepare_data("data/Sentiment/training.csv", model_names[i], heads[i])
            X_val, Y_val = val_data
            val_dataset = EmbeddingDataset(X_val, Y_val)
            val_loader_unfrozen = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

            Y_val = val_data[1]
            final_Y_val = Y_val
            Y_pred_sm = model.predict_sm(val_loader_unfrozen)
            predictions.append(Y_pred_sm)
            Y_pred = torch.argmax(Y_pred_sm, dim=1)-1
            hard_predictions.append(Y_pred)
            mae = mean_absolute_error(Y_val, Y_pred)
            score = 0.5 * (2 - mae)
            print(f"score: {score}")

            conf_matrix = confusion_matrix(Y_val, Y_pred, labels=[-1, 0, 1])

            print(f"Validation Confusion Matrix:\n{conf_matrix}")

        if do_test:
            test_data = pd.read_csv('data/Sentiment/test.csv', index_col=0)
            orig_size = len(test_data)
            X_test = embedder.transform(test_data['sentence'])
            Y_test_fake_labels = np.ones(X_test.shape[0])
            dataset_test = EmbeddingDataset(X_test, Y_test_fake_labels)
            # print(len(dataset_train))
            test_loader = DataLoader(
                dataset_test,
                batch_size=64,
                collate_fn=collate_fn,
            )
            y_test_sm = model.predict_sm(test_loader)
            test_predictions.append(y_test_sm)
            y_test_am = torch.argmax(y_test_sm, dim=1)-1
            test_hard_predictions.append(y_test_am)
            y_labels_unweighted = pd.Series(y_test_am).map(
                {-1: 'negative', 0: 'neutral', 1: 'positive'})
            submission_unweighted = pd.DataFrame({'id': test_data.index, 'label': y_labels_unweighted})
            csv_path = model_paths[i][:-2] + "csv"
            submission_unweighted.to_csv(csv_path, index=False)  # Update filename and path as needed
            print(f"Test predictions saved to '{csv_path}'")

    if do_validate:
        #ensemble on validation set
        stacked_outputs = torch.stack(predictions)
        weighted_sum = (t_weights * stacked_outputs).sum(dim=0)
        unweighted_sum = stacked_outputs.sum(dim=0)
        final_predictions_weighted = torch.argmax(weighted_sum, dim=1)-1
        final_predictions_unweighted = torch.argmax(unweighted_sum,dim=1)-1
        print("unweighted: ")
        mae_unweighted = mean_absolute_error(final_Y_val, final_predictions_unweighted)
        score = 0.5 * (2 - mae_unweighted)

        print(f"ensemble:")
        print(f"score: {score}")

        conf_matrix = confusion_matrix(final_Y_val, final_predictions_unweighted, labels=[-1, 0, 1])

        print(f"Validation Confusion Matrix:\n{conf_matrix}")

        print("weighted: ")
        mae_weighted = mean_absolute_error(final_Y_val, final_predictions_weighted)
        score = 0.5 * (2 - mae_weighted)

        print(f"ensemble:")
        print(f"score: {score}")

        conf_matrix = confusion_matrix(final_Y_val, final_predictions_weighted, labels=[-1, 0, 1])

        print(f"Validation Confusion Matrix:\n{conf_matrix}")

        print("hard voting:")
        hard_preds_tensor = torch.stack(hard_predictions).T
        final_predictions_hard = []
        for preds in hard_preds_tensor:
            counts = torch.bincount(preds + 1, minlength=3)  # Shift by +1 to make -1 be at index 0
            top_count = torch.max(counts)
            top_classes = (counts == top_count).nonzero(as_tuple=True)[0]

            if len(top_classes) > 1:
                final_predictions_hard.append(0)
            else:
                final_predictions_hard.append(int(top_classes[0]) - 1)  # Shift back by -1
        final_predictions_hard = torch.tensor(final_predictions_hard)

        mae_hard = mean_absolute_error(final_Y_val, final_predictions_hard)
        score = 0.5 * (2 - mae_hard)
        print(f"ensemble (hard voting):")
        print(f"score: {score}")

        conf_matrix = confusion_matrix(final_Y_val, final_predictions_hard, labels=[-1, 0, 1])
        print(f"Validation Confusion Matrix (hard voting):\n{conf_matrix}")

    if do_test:
        #ensemble on test set, and save three files. Weighted soft ensemble, unweighted soft ensemble, and hard ensemble
        stacked_test_outputs = torch.stack(test_predictions)
        unweighted_test_sum = stacked_test_outputs.sum(dim=0)
        weighted_test_sum = (t_weights * stacked_test_outputs).sum(dim=0)
        final_test_predictions_unweighted = torch.argmax(unweighted_test_sum, dim=1) - 1
        final_test_predictions_weighted = torch.argmax(weighted_test_sum, dim=1) - 1

        y_labels_unweighted = pd.Series(final_test_predictions_unweighted).map(
            {-1: 'negative', 0: 'neutral', 1: 'positive'})
        submission_unweighted = pd.DataFrame({'id': test_data.index, 'label': y_labels_unweighted})
        submission_unweighted.to_csv('test_predictions_unweighted.csv', index=False)  # Update filename and path as needed
        print("Test unweighted predictions saved to 'test_predictions_unweighted.csv'")

        y_labels_weighted = pd.Series(final_test_predictions_weighted).map({-1: 'negative', 0: 'neutral', 1: 'positive'})
        submission = pd.DataFrame({'id': test_data.index, 'label': y_labels_weighted})
        submission.to_csv('test_predictions_weighted.csv', index=False)  # Update filename and path as needed
        print("Test weighted predictions saved to 'test_predictions_weighted.csv'")

        print("hard voting on test predictions:")

        test_hard_preds_tensor = torch.stack(test_hard_predictions).T
        final_test_predictions_hard = []
        for preds in test_hard_preds_tensor:
            counts = torch.bincount(preds + 1, minlength=3)  # Shift to make -1 => 0, 0 => 1, 1 => 2
            top_count = torch.max(counts)
            top_classes = (counts == top_count).nonzero(as_tuple=True)[0]

            if len(top_classes) > 1:
                final_test_predictions_hard.append(0)
            else:
                final_test_predictions_hard.append(int(top_classes[0]) - 1)  # Shift back

        final_test_predictions_hard = torch.tensor(final_test_predictions_hard)
        y_labels_hard = pd.Series(final_test_predictions_hard).map({-1: 'negative', 0: 'neutral', 1: 'positive'})
        submission_hard = pd.DataFrame({'id': test_data.index, 'label': y_labels_hard})
        submission_hard.to_csv('test_predictions_hard.csv', index=False)

        print("Test hard predictions saved to 'test_predictions_hard.csv'")

if __name__ == "__main__":
    main(do_test=True, do_validate = False)

