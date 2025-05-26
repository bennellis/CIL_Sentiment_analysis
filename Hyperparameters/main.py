import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
from Objectives.BaseObjective import Objective
# main.py
from Hyperparameters.Models.ModelDummy import ModelDummy
from Utils.GitUtils import get_class_file_path, is_file_dirty, get_git_info
import subprocess
import webbrowser
import time
import mlflow

def launch_dashboards():
    # --- Config ---
    mlflow_uri = "http://127.0.0.1:5000"
    optuna_uri = "http://127.0.0.1:8080"
    optuna_db = "sqlite:///optuna_study.db"  # or your path

    # --- Start MLflow UI ---
    subprocess.Popen(["mlflow", "ui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- Start Optuna Dashboard ---
    subprocess.Popen(["optuna-dashboard", optuna_db], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- Give servers a second to start ---
    time.sleep(1.5)

    # --- Open dashboards in browser ---
    webbrowser.open(mlflow_uri)
    webbrowser.open(optuna_uri)

# MODEL_NAME = 'distilbert/distilbert-base-uncased'
STUDY_NAME = "baseline_testing"
HEAD_LIST = ['mlp','rnn','cnn']

def main(pre_process_name = 'NONE', model_name = 'distilbert/distilbert-base-uncased'):
    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage="sqlite:///optuna_study.db",  # File-based DB
        load_if_exists=True
    )

    mlflow.set_experiment(STUDY_NAME)
    balance_train_dataloader=False
    balance_val_dataloader=False
    use_frozen = False
    pre_process = False
    test_model = False
    use_augmented_data = True
    head=HEAD_LIST[0]
    mean_pool = False

    

    launch_dashboards()
    objective = Objective(model_name=model_name, balance_train_dataloader=balance_train_dataloader,
                          balance_val_dataloader=balance_val_dataloader, head=head, use_frozen = use_frozen,
                          test_model = test_model, pre_process = pre_process, pre_process_name = pre_process_name,
                          use_augmented_data = use_augmented_data, mean_pool = mean_pool)
    study.optimize(objective, n_trials=1)
    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":

    pp_list = ['lemmatize',
            'remove_numbers',
            'NONE']
    # pp_list = ['NONE']

    # model_list = [
    #     'google-bert/bert-base-uncased',
    #     'FacebookAI/roberta-base',
    #     'microsoft/deberta-v3-base',
    #     'answerdotai/ModernBERT-base',
    # ]
    # model_list = ['distilbert/distilbert-base-uncased',]
    # model_list = ['microsoft/deberta-v3-base']
    model_list = [
        'FacebookAI/roberta-large',
        'microsoft/deberta-v3-large',
        'answerdotai/ModernBERT-large',
    ]

    for model in model_list:
        main(model_name = model)
