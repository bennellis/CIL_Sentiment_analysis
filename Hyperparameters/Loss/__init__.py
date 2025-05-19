import importlib
import os
import pathlib

# Get current directory
this_dir = pathlib.Path(__file__).parent

# Import all Python files in this directory (excluding __init__.py)
for file in os.listdir(this_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = f"{__name__}.{file[:-3]}"  # e.g., Hyperparameters.Loss.CustomLoss
        importlib.import_module(module_name)
