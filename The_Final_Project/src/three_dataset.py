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

# three dataset
class ThreeDataset(Dataset):

    def __init__(self, split_idx, data_path, transform):
        self.dataset = ImageFolder(data_path)
        self.idx = [i for i, d in enumerate(self.dataset.targets) if
                    self.dataset.class_to_idx['no_image'] != d]
        self.idx = [x for x in self.idx if x in split_idx]
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
        label = 0 if item[1] == 0 else item[1] - 1
        return img, label
