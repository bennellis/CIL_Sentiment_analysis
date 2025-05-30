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
import yaml

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

def main():
    #load config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name=config['study_name'],
        direction="maximize",
        storage="sqlite:///optuna_study.db",  # File-based DB
        load_if_exists=True
    )

    #setup mlflow experiment
    mlflow.set_experiment(config['study_name'])

    #launch the mlflow and optuna dashboards
    launch_dashboards()
    objective = Objective(model_name=config['model']['name'],
                          balance_train_dataloader=config['dataloader']['balance_train'],
                          balance_val_dataloader=config['dataloader']['balance_val'],
                          head=config['model']['head'],
                          use_frozen = config['dataloader']['use_frozen'],
                          test_model = config['model']['test_model'],
                          pre_process = config['dataloader']['pre_process'],
                          pre_process_name = config['dataloader']['pre_process_name'],
                          use_augmented_data = config['dataloader']['use_augmented_data'],
                          mean_pool = config['model']['mean_pool'])
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
    main()
