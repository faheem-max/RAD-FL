import torch
import torch.nn as nn


class GNLeNet(nn.Module):
    """Same lightweight GNLeNet used by the PC simulation."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=32)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.gn1(self.conv1(x))))
        x = self.pool(self.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
