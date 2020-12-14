from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1 = nn.Conv2d(1, 6, (5,5))
        # Layer 2: Convolutional. Output = 10x10x16.
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.fc1   = nn.Linear(400, 120)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2   = nn.Linear(120, 84)
        # Layer 5: Fully Connected. Input = 84. Output = 10.
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # Activation. # Pooling. Input = 28x28x6. Output = 14x14x6.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
         # Activation. # Pooling. Input = 10x10x16. Output = 5x5x16.
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # Flatten. Input = 5x5x16. Output = 400.
        x = x.flatten(start_dim=1)
        # Activation.
        x = F.relu(self.fc1(x))
        # Activation.
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features