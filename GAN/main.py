from model.Generator import Generator
from model.Discriminator import Discriminator
from model.GAN import GAN
import torch
# import mnist from the torch library
from torchvision import datasets, transforms
from PIL import Image

def main():
    noise = torch.randn(100).to('cuda')

    generator = Generator().to('cuda')
    generator.eval()
    generated_image_raw = generator(noise.unsqueeze(0)).cpu().detach().numpy()
    generated_image = generated_image_raw.reshape(28, 28, 1)
    generated_image = Image.fromarray(generated_image, 'L')
    generated_image.save('generated_image.png')

    # discriminator = Discriminator().to('cuda')
    # discriminator.eval()
    # generated_image_raw = torch.from_numpy(generated_image_raw).to('cuda')
    
    # discriminator_output = discriminator(generated_image_raw).cpu().detach().numpy()

def train():
    training_set = datasets.MNIST(root='./dataset', train=True, download=False,
                                    transform=transforms.Compose([
                                        transforms.PILToTensor()
                                    ]))
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True)

    gan = GAN()
    gan.train(training_loader, 1)

if __name__ == '__main__':
    main()
    # train()