import torch
import mlflow
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from Hyperparameters.Utils.Misc import get_device
from Hyperparameters.Training.BasicTrainingLoop import train
from Hyperparameters.Training.BasicValidationLoop import validate
from Hyperparameters.Models.ModelDummy import ModelDummy
from Hyperparameters.Utils.GitUtils import log_model_git_info
from Hyperparameters.registry import model_registry


def get_mnist_dataloaders(batch_size=8):
    # Load the MNIST train and test datasets and save them to ./data
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    train_loader = torch.utils.data.DataLoader(mnist_train,
                                               batch_size=batch_size,
                                               shuffle=True)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    val_loader = torch.utils.data.DataLoader(mnist_test,
                                             batch_size=1000,
                                             shuffle=True)
    return train_loader, val_loader


def suggest_hyperparameters(trial):
    # Learning rate on a logarithmic scale
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Dropout ratio in the range from 0.0 to 0.9 with step size 0.1    # Optimizer to use as categorical value
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"])

    return lr, optimizer_name


def objective(trial):
    best_val_loss = float('Inf')

    with mlflow.start_run():
        # Get hyperparameters
        lr, optimizer_name = suggest_hyperparameters(trial)

        device = get_device(True)

        model_name = trial.suggest_categorical("model_name", list(model_registry.registry.keys()))
        model_info = model_registry.registry[model_name]
        log_model_git_info(model_name, model_info)

        model_class = model_info["class"]
        model_params = model_class.suggest_hyperparams(trial)

        mlflow.log_params(trial.params)
        print(trial.params)
        model = model_class(**model_params)
        model.to(device)

        # Optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

        # Data
        train_loader, val_loader = get_mnist_dataloaders()

        # Train/validate loop
        for epoch in range(5):
            avg_train_loss = train(model, device, train_loader, optimizer, epoch)
            avg_val_loss = validate(model, device, val_loader)

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            mlflow.log_metric("avg_train_losses", avg_train_loss, step=epoch)
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)

            scheduler.step()

    return best_val_loss

