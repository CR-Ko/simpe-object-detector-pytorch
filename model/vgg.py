import torch
import torch.nn as nn

class mini_VGG(nn.Module):

    def __init__(self, features, num_classes=20, init_weights=True):
        super(mini_VGG, self).__init__()
        self.features   = features
        self.classifier = nn.Linear(?,?)
        if init_weights:
            self._initialize_weights()


    def forward(self,x):
        x = self.features(x)
        



