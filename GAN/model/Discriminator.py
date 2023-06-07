from torch import nn
import numpy as np
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_image_shape = (1, 28, 28)
        self.output_prob_shape = (1, )
        self.loss = nn.BCELoss()

        self.model = nn.Sequential(
            nn.Linear(np.prod(self.input_image_shape), 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, np.prod(self.output_prob_shape))
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, x):
        return self.model(x.flatten())
    
    def train(self, images):
        self.optimizer.zero_grad()
        prediction = self.forward(images)
        loss = self.loss(prediction, torch.ones(self.output_prob_shape))
        loss.backward()
        self.optimizer.step()

        return loss.item()