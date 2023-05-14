# -*- coding: utf-8 -*-
"""

An example for SVDD model fitting using different kernels

"""
import sys
sys.path.append("..")
from BaseSVDD import BaseSVDD
import torch
from torch import nn, optim
from torch.utils.data import random_split, Subset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import csv
from two_dataset import TwoDataset
import numpy as np

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.7564, 0.7176, 0.6869], [0.3128, 0.3258, 0.3480])
])

data_path = "../data/Plastics Classification/"

dataset = datasets.ImageFolder(data_path)
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=generator)
train_idx = train_dataset.indices
test_idx = test_dataset.indices

train_dataset = TwoDataset(train_idx, data_path, transform)
test_dataset = TwoDataset(test_idx, data_path, transform)

train_dataloader = DataLoader(train_dataset, batch_size=10000)
test_dataloader = DataLoader(test_dataset, batch_size=10000)

svdd = BaseSVDD(C=1.0, kernel='rbf', gamma='scale', display='on')

# train part
for X, y in train_dataloader:
    X = X.numpy()
    y = y.numpy()
    y = y.reshape(-1, 1)
    svdd.fit(X, y)
    # svdd.plot_boundary(X,  y)

# test part
for X, y in test_dataloader:
    X = X.numpy()
    y = y.numpy()
    y = y.reshape(-1, 1)
    svdd.predict(X, y)
