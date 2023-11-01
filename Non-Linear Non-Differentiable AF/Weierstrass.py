import torch
from torch import nn
import numpy as np
from cprint import cprint

class Weierstrass(nn.Module):
    def __init__(self, a=0.5, b=10):
        super(Weierstrass, self).__init__()
        self.a = a
        self.b = b
        self.aa = a ** 2
        self.bb = b ** 2

    def forward(self, x):
        return torch.cos(torch.pi * x) + self.a * torch.cos(self.b * torch.pi * x) + self.aa * torch.cos(self.bb * torch.pi * x)
    
    def test(self):
        a = 0.5
        b = 10
        f = Weierstrass(a, b)
        for i in range(0, 11, 1):
            print('-----------------------------------')
            print(f'Test {i}')
            print(f'x = {i}')
            x = torch.tensor([i, i])
            y = f(x)
            y_1 = torch.nn.ReLU()(x)
            print(f'y = {y}')
            print(f'y_1 = {y_1}')

if __name__ == '__main__':
    func = Weierstrass()
    func.test()
