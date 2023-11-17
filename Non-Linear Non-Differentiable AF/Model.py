import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Linear):
                nn.init.xavier_uniform_(self._modules[m].weight)
                nn.init.zeros_(self._modules[m].bias)
            elif isinstance(self._modules[m], nn.Conv2d):
                nn.init.xavier_uniform_(self._modules[m].weight)
                nn.init.zeros_(self._modules[m].bias)

    def forward(self, x):
        pass
    
    def train(self, epochs, training_set, validation_set, optimizer, criterion):
        train_loss: float
        for epoch in range(epochs):
            train_loss = 0
            for data, target in training_set:
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'---- Epoch: {epoch} ----')
            print(f'Training Loss: {train_loss / len(training_set) : .3f}')
            self.validate(validation_set, criterion)
            print('-----------------------')

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
        print(f'Validation Loss: {validation_loss / len(validation_set) : .3f}')
        print(f'Accuracy: {correct/total * 100 : .3f}%')
