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


STUDY_NAME = "sentiment-optim"

def main():
    # Create the optuna study which shares the experiment name
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage="sqlite:///optuna_study.db",  # File-based DB
        load_if_exists=True
    )

    mlflow.set_experiment(STUDY_NAME)

    launch_dashboards()
    objective = Objective()
    study.optimize(objective, n_trials=5)
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
