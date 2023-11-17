from torch import nn
from Weierstrass import Weierstrass
from Model import Model

class WeierstrassNetwork(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            Weierstrass(),
            nn.LeakyReLU(),
            nn.Linear(100, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.layers(x)