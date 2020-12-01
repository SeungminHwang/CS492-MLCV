import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import cv2


def main():
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    test_data = dsets.MNIST(root='../data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader =

if __name__ == '__main__':
    main()
