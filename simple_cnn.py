# simple_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define a simple CNN with one convolutional layer, pooling, and one fully-connected layer.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # CIFAR-100 images are 32x32, so after one pooling layer, they become 16x16.
        self.fc1 = nn.Linear(16 * 16 * 16, 100)  # 100 classes

    def forward(self, x):
        x = self.conv1(x)          # Apply convolution
        x = F.relu(x)              # Apply ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)            # Fully connected layer for classification
        return x

