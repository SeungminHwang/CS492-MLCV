import torch.utils.data as data
import os
import cv2
from torchvision import transforms
import torch
import numpy as np

class CGANDataSet(data.Dataset):
    def __init__(self, image_root, labels):
        self.image_path = image_root
        self.labels = labels
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    def __getitem__(self, index):
        image_path = os.path.join(self.image_path, "{}.png".format(index))
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape((32, 32, 1))
        return self.transform(image), torch.tensor(self.labels[index, 0])

    def __len__(self):
        folder = os.listdir(self.image_path)
        return len(folder)
