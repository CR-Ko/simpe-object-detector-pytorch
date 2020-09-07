import torch
import torch.nn as nn
import numpy as np

class mini_VGG(nn.Module):
    def __init__(self):
        super(mini_VGG, self).__init__()
        #self.num_classes = num_classes
        self.conv1    = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2    = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3    = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4    = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
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


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('GPU state: ', device)
    img_size = (224, 224) 
    x = torch.zeros((1, 3) + img_size)
    x = x.to(device)
    with torch.no_grad():
        net = mini_VGG().to(device)
        #from torchsummary import summary
        #summary(net, (3, 224, 224))
        out = net(x)
        print(out.size())


if __name__ == '__main__':
    main()




