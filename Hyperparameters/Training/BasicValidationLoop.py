import torch
import torch.nn.functional as F


def validate(model, device, val_loader):
    model.eval()
    val_set_size = len(val_loader.dataset)
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= val_set_size

    print(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{val_set_size} "
          f"({100. * correct / val_set_size:.0f}%)\n")
    return val_loss
