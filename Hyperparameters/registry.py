import functools
from Hyperparameters.Utils.GitUtils import (
    get_model_file_path,
    is_file_dirty,
    get_git_info,
    get_github_link,
)
import mlflow

MODEL_REGISTRY = {}
PREPROCESSOR_REGISTRY = {}


def register_model(name, enforce_clean=True, verbose=False):
    def decorator(cls):
        file_path = get_model_file_path(cls)

        if name in MODEL_REGISTRY:
            print(f"⚠️ Model '{name}' already registered, skipping duplicate.")
            return cls

        print(f"🔍 Registering model: {name} (enforce_clean={enforce_clean})")

        if enforce_clean:
            if is_file_dirty(file_path):
                raise RuntimeError(
                    f"\nModel {name} has uncommitted changes.\nLocated in file '{file_path}'.\nPlease COMMIT and PUSH "
                    f"before registering the model"
                )

        # Get and log git info (always)
        git_info = get_git_info()
        file_link = get_github_link(git_info["remote_url"], file_path, git_info["commit_sha"])
        if verbose:
            print("✅ Git Info:")
            print(f"  User      : {git_info['user_name']} <{git_info['user_email']}>")
            print(f"  Commit    : {git_info['commit_sha']}")
            print(f"  Branch    : {git_info['branch']}")
            print(f"  File link : {file_link}")

        # Register the model
        MODEL_REGISTRY[name] = {
            "class": cls,
            "git_info": git_info,
            "file_path": file_path,
            "dirty": not enforce_clean
        }
        return cls

    return decorator


def register_preprocessor(name):
    def wrapper(cls):
        PREPROCESSOR_REGISTRY[name] = cls
        return cls

    return wrapper
