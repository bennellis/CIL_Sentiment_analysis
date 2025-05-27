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
    df = pd.read_csv(csv_path, index_col=0)
    label_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['label_encoded'] = df['label'].map(label_map)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['sentence'], df['label_encoded'],
        stratify=df['label_encoded'], test_size=0.1, random_state=42
    )

    # if pre_process:  # pre-process data
    #     len_orig_train = len(train_texts)
    #     train_texts, train_kept_indices = self.pre.transform(train_texts)
    #     train_labels = train_labels.iloc[train_kept_indices].tolist()
    #     self.removed_train = len_orig_train - len(train_texts)
    #     print(f'removed training samples: {self.removed_train}')
    #
    #     len_orig_val = len(val_texts)
    #     all_indices = set(range(len_orig_val))
    #     val_texts_new, val_kept_indices = self.pre.transform(val_texts)
    #     kept_set = set(val_kept_indices)
    #     removed_indices = all_indices - kept_set
    #     sum_removed = sum(abs(val_labels[i]) for i in removed_indices)
    #     val_texts = val_texts_new
    #     val_labels = val_labels.iloc[val_kept_indices].tolist()
    #     self.removed_val = len_orig_val - len(val_texts)
    #     print(f'removed validation samples: {self.removed_val}')
    #
    #     self.added_val_score = float(sum_removed) / len_orig_val

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

    d_params = {"lr": 1e-4,"pt_lr_top": 1e-5,"pt_lr_mid": 5e-6,"pt_lr_bot": 3e-6,"dropout": 0.4,
                "temperature": 1.0,"ce_weight": 0.2,"margin": 0,"use_cdw": True}
    params = [d_params,d_params,d_params,d_params,d_params,d_params, d_params]
    model_names = ['distilbert/distilbert-base-uncased', 'FacebookAI/roberta-base',
                   'google-bert/bert-base-uncased','FacebookAI/roberta-base',
                   'microsoft/deberta-v3-base','answerdotai/ModernBERT-base',
                   'microsoft/deberta-v3-base',
                   ]

    model_vscores = np.array([0.855, 0.875, 0.864, 0.882, 0.902, 0.889, 0.904])
    # model_vscores = model_vscores[[4, 6]]
    max_score = max(model_vscores)
    model_weights = [(vscore - 0.8) / (max_score - 0.8) for vscore in model_vscores]
    t_weights = torch.tensor(model_weights).view(-1, 1, 1)
    model_paths = ['saved_weights/distilbert/distilbert-base-uncased/baseline_no_warmup.pt',
                   'saved_weights/roberta_875_04-27.pt',
                   'saved_weights/google-bert/bert-base-uncased/baseline_1.pt',
                   'saved_weights/FacebookAI/roberta-base/baseline_1.pt',
                   'saved_weights/microsoft/deberta-v3-base/baseline_1.pt',
                   'saved_weights/answerdotai/ModernBERT-base/baseline_1.pt',
                   'saved_weights/microsoft/deberta-v3-base/baseline_cnn_plus_augmented.pt',]
    heads = ['mlp','mlp','mlp','mlp','mlp','mlp','cnn']

    predictions = []
    test_predictions = []
    hard_predictions = []
    test_hard_predictions = []
    final_Y_val =  None
    for i in [4,6]:#range(len(model_names)):
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
            # if pre_process:
            #     sentences, kept_indices = self.pre.transform(test_data["sentence"])
            #     X_test = self.embedder.transform(sentences)
            # else:
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

            # if self.pre_process:
            #     full_labels = pd.Series(['neutral'] * len(test_data), index=test_data.index)
            #     full_labels.iloc[kept_indices] = y_labels.values
            #     y_labels = full_labels

    if do_validate:
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
        hard_preds_tensor = torch.stack(hard_predictions)
        final_predictions_hard, _ = torch.mode(hard_preds_tensor, dim=0)

        mae_hard = mean_absolute_error(final_Y_val, final_predictions_hard)
        score = 0.5 * (2 - mae_hard)
        print(f"ensemble (hard voting):")
        print(f"score: {score}")

        conf_matrix = confusion_matrix(final_Y_val, final_predictions_hard, labels=[-1, 0, 1])
        print(f"Validation Confusion Matrix (hard voting):\n{conf_matrix}")

    if do_test:
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
        test_hard_preds_tensor = torch.stack(test_hard_predictions)
        final_test_predictions_hard, _ = torch.mode(test_hard_preds_tensor, dim=0)

        y_labels_hard = pd.Series(final_test_predictions_hard).map({-1: 'negative', 0: 'neutral', 1: 'positive'})
        submission_hard = pd.DataFrame({'id': test_data.index, 'label': y_labels_hard})
        submission_hard.to_csv('test_predictions_hard.csv', index=False)
        print("Test hard predictions saved to 'test_predictions_hard.csv'")

if __name__ == "__main__":
    main(do_test=True, do_validate = False)

