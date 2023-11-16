from torchvision import datasets, transforms
import torch
from torch import nn
from Network import Network
from WeierstrassNetwork import WeierstrassNetwork

def main():
    # download mnist
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train = datasets.MNIST('', train=True, download=True, transform=transform)
    test = datasets.MNIST('', train=False, download=True, transform=transform)

    train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=6)
    validation_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True, num_workers=2)

    # create networks
    net = Network()
    net.weight_init()
    net_optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    weierstrass_net = WeierstrassNetwork()
    # weierstrass_net.weight_init()
    weierstrass_optimizer = torch.optim.Adam(weierstrass_net.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    epochs = 3

    # train the network
    # net.train(epochs, train_set, validation_set, net_optimizer, criterion)

    # train the weierstrass network
    weierstrass_net.train(epochs, train_set, validation_set, weierstrass_optimizer, criterion)


if __name__ == "__main__":
    main()