from torch import nn
import torch
from .Generator import Generator
from .Discriminator import Discriminator

class GAN():
    def __init__(self):
        self.generator = Generator().to('cuda')
        self.discriminator = Discriminator().to('cuda')

    def train(self, training_loader: torch.utils.data.DataLoader, epochs: int):
        half_batch_size = training_loader.batch_size // 2
        for epoch in range(epochs):
            for i, (images, _) in enumerate(training_loader):
                idx = torch.randint(0, images.shape[0], (half_batch_size, ))
                real_images = images[idx].to('cuda')
                

                noise = torch.randn(half_batch_size, 100).to('cuda')
                gen_images = self.generator(noise.unsqueeze(0))
                

                real_loss = self.discriminator.train(real_images)
                fake_loss = self.discriminator.train(gen_images)

                print(f'Epoch: {epoch}, Batch: {i}, Real Loss: {real_loss}, Fake Loss: {fake_loss}')