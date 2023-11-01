from torchvision import datasets, transforms
import torch
from torch import nn
from Network import Network

def main():
    # download mnist
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train = datasets.MNIST('', train=True, download=True, transform=transform)
    test = datasets.MNIST('', train=False, download=True, transform=transform)

    train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    validation_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    # create a network
    net = Network()
    net.weight_init()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # train the network
    epochs = 3
    net.train(epochs, train_set, validation_set, optimizer, criterion)


if __name__ == "__main__":
    main()