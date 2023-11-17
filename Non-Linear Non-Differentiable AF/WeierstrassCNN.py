from torch import nn
from Weierstrass import Weierstrass
from Model import Model

class WeierstrassCNN(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels=1, 
                                               out_channels=32, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=2,
                                               bias=False),
                                     Weierstrass(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2d(in_channels=32, 
                                               out_channels=64, 
                                               kernel_size=3, 
                                               stride=1, 
                                               padding=2),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Flatten(),
                                     nn.Linear(4096, 10)
                                    )

    def forward(self, x):
        return self.network(x)