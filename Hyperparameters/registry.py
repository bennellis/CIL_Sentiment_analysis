import functools
from Hyperparameters.Utils.GitUtils import (
    get_class_file_path,
    is_file_dirty,
    get_git_info,
    get_github_link,
)
import mlflow

import functools
from Hyperparameters.Utils.GitUtils import (
    get_class_file_path,
    is_file_dirty,
    get_git_info,
    get_github_link,
)

class Registry:
    def __init__(self, name, enforce_git_clean=True, verbose=False):
        self.name = name
        self.registry = {}
        self.enforce_git_clean = enforce_git_clean
        self.verbose = verbose

    def register(self, name, enforce_clean=None, verbose=None):
        enforce_clean = self.enforce_git_clean if enforce_clean is None else enforce_clean
        verbose = self.verbose if verbose is None else verbose

        def decorator(cls):
            file_path = get_class_file_path(cls)

            if name in self.registry:
                print(f"{self.name} '{name}' already registered, skipping duplicate.")
                return cls

            print(f"Registering {self.name}: {name} (enforce_clean={enforce_clean})")

            if enforce_clean:
                if is_file_dirty(file_path):
                    raise RuntimeError(
                        f"\n{self.name} {name} has uncommitted changes.\nLocated in file '{file_path}'.\nPlease COMMIT and PUSH "
                        f"before registering."
                    )

            git_info = get_git_info()
            file_link = get_github_link(git_info["remote_url"], file_path, git_info["commit_sha"])

            if verbose:
                print("Git Info:")
                print(f"  User      : {git_info['user_name']} <{git_info['user_email']}>")
                print(f"  Commit    : {git_info['commit_sha']}")
                print(f"  Branch    : {git_info['branch']}")
                print(f"  File link : {file_link}")

            self.registry[name] = {
                "class": cls,
                "git_info": git_info,
                "file_path": file_path,
                "dirty": not enforce_clean
            }

            return cls

        return decorator

    def simple_register(self, name):
        """Simple registration without git checks (e.g., preprocessors)."""
        def decorator(cls):
            if name in self.registry:
                print(f"⚠️ {self.name} '{name}' already registered, skipping duplicate.")
                return cls

            print(f"🔍 Registering {self.name}: {name} (no git checks)")
            self.registry[name] = cls
            return cls

        return decorator


# Instantiate registries
model_registry = Registry("Model", enforce_git_clean=False, verbose=True)
embedding_registry = Registry("Embedding", enforce_git_clean=False, verbose=True)



# preprocessor_registry = Registry("Preprocessor", enforce_git_clean=False)  # Preprocessors don't require git checks


register_model = model_registry.register
register_embedding = embedding_registry.register
# register_preprocessor = preprocessor_registry.simple_register
