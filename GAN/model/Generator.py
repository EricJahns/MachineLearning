from torch import nn
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_noise_shape = (100, 1)
        self.output_image_shape = (1, 28, 28)

        self.loss = nn.BCELoss()

        self.model = nn.Sequential(
            nn.Linear(np.prod(self.input_noise_shape), 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, np.prod(self.output_image_shape))
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, x):
        print(x)
        out = self.model(x)
        
        return out.reshape(out.shape[0], *self.output_image_shape)
    
    def train(self, noise):
        self.optimizer.zero_grad()
        prediction = self.forward(noise)
        loss = self.loss(prediction, torch.ones(self.output_image_shape))
        loss.backward()
        self.optimizer.step()

        return loss.item()
