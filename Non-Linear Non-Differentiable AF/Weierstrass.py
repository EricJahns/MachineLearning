import torch
from torch import nn

class Weierstrass(nn.Module):
    def __init__(self, a=0.5, b=10):
        super(Weierstrass, self).__init__()
        self.a = a
        self.b = b
        self.aa = a ** 2
        self.bb = b ** 2
        self.aaa = self.aa * self.a
        self.bbb = self.bb * self.b

    def forward(self, x):
        return torch.cos(torch.pi * x) + \
               self.a * torch.cos(self.b * torch.pi * x) + \
               self.aa * torch.cos(self.bb * torch.pi * x) + \
               self.aaa * torch.cos(self.bbb * torch.pi * x)

if __name__ == '__main__':
    func = Weierstrass()
    func.test()
