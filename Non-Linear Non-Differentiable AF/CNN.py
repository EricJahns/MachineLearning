from torch import nn
from Model import Model

class CNN(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels=1, 
                                               out_channels=32, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=2),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(in_channels=32, 
                                               out_channels=64, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=2),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Flatten(),
                                     nn.ReLU(),
                                     nn.Linear(4096, 10)
                                    )

    def forward(self, x):
        return self.network(x)