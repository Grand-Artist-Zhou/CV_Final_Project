import torch
from torch import nn, optim
from torch.utils.data import random_split, Subset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
from three_dataset import ThreeDataset
from four_dataset import FourDataset
import matplotlib.pyplot as plt
import csv
import os

from utils import record_to_csv, record_to_csv2d
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.7564, 0.7176, 0.6869], [0.3128, 0.3258, 0.3480])
])

data_path = "../data/Plastics Classification/"

# generate different dataset
dataset = datasets.ImageFolder(data_path)
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=generator)
train_idx = train_dataset.indices
test_idx = test_dataset.indices

four_kind_dataset_train = FourDataset(
    split_idx=train_idx, data_path=data_path, transform=transform)
four_kind_dataset_test = FourDataset(
    split_idx=test_idx, data_path=data_path, transform=transform)

three_kind_dataset_train = ThreeDataset(
    split_idx=train_idx, data_path=data_path, transform=transform)
three_kind_dataset_test = ThreeDataset(
    split_idx=test_idx, data_path=data_path, transform=transform)


# train function with loss printed every 100 batches
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # break
    return losses

# test function with test accuracy and loss printed
def test(dataloader, model, loss_fn, kind_num):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        real_y_nums = [0 for _ in range(kind_num)]
        pred_y_nums = [0 for _ in range(kind_num)]
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            for real_y, pred_y in zip(y, pred.argmax(1)):     
                real_y_nums[real_y] += 1
                if real_y == pred_y:
                    pred_y_nums[real_y] += 1

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # break
    true_y_nums = [0 for _ in range(kind_num)]
    for i in range(len(real_y_nums)):
        true_y_nums[i] = pred_y_nums[i] / real_y_nums[i]
    # todo prepend label for true_y_nums
    
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct, true_y_nums

# multi kinds trains
def multi_kinds_train(kind_num, train_dataset, test_dataset):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, kind_num)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)

    num_epochs = 15

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    true_y_nums = []

    for t in range(num_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_losses = train(train_loader, model, criterion, optimizer)
        _, train_acc, _ = test(train_loader, model, criterion, kind_num)
        test_loss, test_acc, true_y_num = test(test_loader, model, criterion, kind_num)

        # save the loss and acc data
        train_losses.extend(train_losses)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        true_y_nums.append(true_y_num)
    
    torch.save(model.state_dict(), f'../models/myNetworkModel-{kind_num}')
    print("Training finished. Model saved.")

    record_to_csv('../' + str(kind_num) + '_train_loss.csv', train_losses)
    record_to_csv('../' + str(kind_num) + '_test_loss.csv', test_losses)
    record_to_csv('../' + str(kind_num) + '_train_acc.csv', train_accs)
    record_to_csv('../' + str(kind_num) + '_test_acc.csv', test_accs)
    record_to_csv2d('../' + str(kind_num) + '_true_Y.csv', true_y_nums)

# driver code for our test
multi_kinds_train(3, three_kind_dataset_train, three_kind_dataset_test)
multi_kinds_train(4, four_kind_dataset_train, four_kind_dataset_test)
