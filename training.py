# Train NN
import datetime

import torch

from torch.utils.data import DataLoader

EPOCHS = 15

def train_loop(dataloader: DataLoader, model, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    model.to(device)
    total_start = datetime.datetime.now()
    correct, train_loss = 0, 0
    # size = dataloader.dataset.__len__()
    last_batch = 0
    start_batch_loading = datetime.datetime.now()
    for batch, (X, y) in enumerate(dataloader):
        start_batch_training = datetime.datetime.now()
        batch_loading_time = (start_batch_training - start_batch_loading).total_seconds()
  
        last_batch = batch
        X, y = X.to(device), y.to(device)


        # Compute prediction and loss
        pred, _ = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += torch.eq(torch.argmax(pred, dim=1), y).type(torch.float).sum().item()
        train_loss += loss.item()
        
        batch_training_time = (datetime.datetime.now() - start_batch_training).total_seconds()
        start_batch_loading = datetime.datetime.now()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}] training time ratio: {batch_training_time / (batch_training_time + batch_loading_time)}")
    last_batch += 1
    print(f'Training time: {(datetime.datetime.now() - total_start).total_seconds()}')
    correct /= last_batch*dataloader.batch_size
    return train_loss / last_batch, correct


def val_loop(dataloader: DataLoader, model, loss_fn, device):
    model.eval()
    model.to(device)
    test_loss, correct = 0, 0
    last_batch = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            last_batch = batch
            X, y = X.to(device), y.to(device)

            pred, _ = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += torch.eq(torch.argmax(pred, dim=1), y).type(torch.float).sum().item()

    last_batch += 1
    test_loss /= last_batch
    correct /= last_batch*dataloader.batch_size
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    return test_loss, correct
