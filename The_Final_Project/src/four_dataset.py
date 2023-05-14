import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import cv2

# four class dataset
class FourDataset(Dataset):

    def __init__(self, split_idx, data_path, transform):
        self.dataset = ImageFolder(data_path)
        self.idx = split_idx
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, id):
        item = self.dataset.samples[self.idx[id]]
        img = cv2.imread(item[0])
        img = self.transform(img)
        return img, item[1]
