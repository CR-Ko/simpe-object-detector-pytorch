import torch
import torch.nn as nn

class mini_VGG(nn.Module):
    def __init__(self, features, num_classes=20, init_weights=True):
        super(mini_VGG, self).__init__()
        self.num_classes = num_classes
        self.conv1    = nn.Conv2d(in_channels=3, out_channels=64, kernal_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernal_size=2, stride=2)
        self.conv2    = nn.Conv2d(in_channels=64, out_channels=128, kernal_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernal_size=2, stride=2)
        self.conv3    = nn.Conv2d(in_channels=128, out_channels=256, kernal_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernal_size=2, stride=2)
        self.conv4    = nn.Conv2d(in_channels=256, out_channels=512, kernal_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernal_size=2, stride=2) 
        self.fc1      = nn.Linear(14*14*512, 1000)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = self.conv4(x)
            x = self.maxpool4(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x




