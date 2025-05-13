import os
import importlib

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        importlib.import_module(f"{__name__}.{filename[:-3]}")
