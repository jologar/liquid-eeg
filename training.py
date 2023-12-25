# Train NN
import datetime

import torch

EPOCHS = 15

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    total_start = datetime.datetime.now()
    correct, train_loss = 0, 0
     
    size = dataloader.dataset.__len__()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.unsqueeze(1).to(device), y.to(device)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += torch.eq(torch.argmax(pred, dim=1), y).type(torch.float).sum().item()
        train_loss += loss.item()
        
        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    print(f'Training time: {(datetime.datetime.now() - total_start).total_seconds()}')
    correct /= size
    return train_loss / batch, correct


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = dataloader.dataset.__len__()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.unsqueeze(1).to(device), y.to(device)
            pred, _ = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += torch.eq(torch.argmax(pred, dim=1), y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    return test_loss, correct
