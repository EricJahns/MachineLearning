from torch import nn
from Model import Model

class Network(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.network(x)