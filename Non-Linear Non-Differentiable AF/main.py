from torchvision import datasets, transforms
import torch
from torch import nn
from Network import Network
from WeierstrassNetwork import WeierstrassNetwork
from CNN import CNN
from WeierstrassCNN import WeierstrassCNN
from torchsummary import summary

def train_net(net, epochs, training_set, validation_set, criterion):
    print(f'\n---- {type(net).__name__} ----\n')
    summary(net, (1, 28, 28))
    net.weight_init()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train(epochs, training_set, validation_set, optimizer, criterion)

def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    train = datasets.MNIST('', train=True, download=True, transform=transform)
    test = datasets.MNIST('', train=False, download=True, transform=transform)

    training_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=6)
    validation_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True, num_workers=2)

    # create networks
    criterion = nn.CrossEntropyLoss()
    epochs = 3
   
    net = Network()
    train_net(net, epochs, training_set, validation_set, criterion)

    weierstrass_net = WeierstrassNetwork()
    train_net(weierstrass_net, epochs, training_set, validation_set, criterion)

    cnn = CNN()
    train_net(cnn, epochs, training_set, validation_set, criterion)

    weierstrass_cnn = WeierstrassCNN()
    train_net(weierstrass_cnn, epochs, training_set, validation_set, criterion)

if __name__ == "__main__":
    main()