import torch
from torch import nn
from Weierstrass import Weierstrass

class Network(nn.Module):
    # create a simple network for mnist
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 10)
        self.weierstrass = Weierstrass()
        self.relu = nn.ReLU()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Linear):
                nn.init.xavier_uniform_(self._modules[m].weight)
                nn.init.zeros_(self._modules[m].bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def train(self, epochs, training_set, validation_set, optimizer, criterion):
        train_loss: float
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            train_loss = 0
            for data, target in training_set:
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Training Loss: {train_loss / len(training_set)}')
            self.validate(validation_set, criterion)

    def validate(self, validation_set, criterion):
        correct = 0
        total = 0
        validation_loss: float
        with torch.no_grad():
            validation_loss = 0
            for data, target in validation_set:
                output = self.forward(data)
                loss = criterion(output, target)
                validation_loss += loss.item()
                for idx, i in enumerate(output):
                    if torch.argmax(i) == target[idx]:
                        correct += 1
                    total += 1
        accuracy = round(correct/total, 3)
        print(f'Validation Loss: {validation_loss / len(validation_set)}')
        print(f'Accuracy: {accuracy * 100}%')
