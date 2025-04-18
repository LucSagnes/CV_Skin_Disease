import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SmallNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=1, padding=0)
        self.act2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0)
        self.act3 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(in_features=10_368, out_features=6_400)
        self.fc2 = nn.Linear(in_features=6_400, out_features=1_280)
        self.fc3 = nn.Linear(in_features=1_280, out_features=7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output