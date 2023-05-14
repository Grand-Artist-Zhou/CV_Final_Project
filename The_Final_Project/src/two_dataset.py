import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import cv2
from PIL import Image

# two dataset
class TwoDataset(Dataset):

    def __init__(self, split_idx, data_path, transform):
        self.dataset = ImageFolder(data_path)
        self.idx = split_idx
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, id):
        item = self.dataset.samples[self.idx[id]]
        img = cv2.imread(item[0])
        # print("path:", item[0])
        # print("img before", img)
        img = self.transform(img)
        # print("img after", img)
        label = -1 if item[1] == 1 else 1
        return img.view(-1), label
