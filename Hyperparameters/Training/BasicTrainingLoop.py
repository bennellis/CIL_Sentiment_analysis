import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_set_size = len(train_loader.dataset)
    num_batches = len(train_loader)
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            batch_size = len(data)
            print(f"Train Epoch: {epoch} [{batch_idx * batch_size}/{train_set_size} "
                  f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}")
    avg_train_loss = train_loss / num_batches
    return avg_train_loss
