import torch


def get_device(log_with_mlflow: bool = False):
    """
    Automatically selects the best available device.

    Priority: CUDA > MPS (Mac) > CPU

    Args:
        log_with_mlflow (bool): If True, logs the selected device to MLflow.

    Returns:
        torch.device: Selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS (Metal)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    if log_with_mlflow:
        try:
            import mlflow
            mlflow.log_param("device", device_name)
        except ImportError:
            print("MLflow not installed; skipping device logging.")

    print(f"[INFO] Using device: {device_name}")
    return device


def suggest_namespaced_params(trial, model_name, param_defs):
    """
    Suggest namespaced hyperparameters for a model using Optuna.

    Args:
        trial: The Optuna trial object.
        model_name (str): The name to namespace the hyperparameters under.
        param_defs (dict): A dictionary defining each hyperparameter and how to suggest it.
                           Example:
                           {
                               "hidden_dim": lambda trial: trial.suggest_int("hidden_dim", 64, 256),
                               "dropout": lambda trial: trial.suggest_float("dropout", 0.1, 0.5)
                           }

    Returns:
        dict: A dictionary of suggested hyperparameter values with keys like "ModelName_param".
    """
    params = {}
    for param_name, suggest_fn in param_defs.items():
        namespaced_name = f"{model_name}_{param_name}"
        params[param_name] = suggest_fn(trial, namespaced_name)
    return params
