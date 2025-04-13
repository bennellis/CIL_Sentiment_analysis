import subprocess
import webbrowser
import time
import os

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
