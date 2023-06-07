import torch
from tqdm import tqdm, trange

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer():
    def __init__(self, model, training_loader, validation_loader, testing_loader, optimizer, criterion):
        self.model = model.to(DEVICE)
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.optimizer = optimizer
        self.criterion = criterion.to(DEVICE)

    def train(self, epochs):
        for epoch in trange(epochs):
            loss = self.step()
            tqdm.write(f'Epoch: {epoch + 1} | Loss: {loss : .6f}')

    def step(self):
        running_loss: float = 0

        for _, (image, mask) in enumerate(self.training_loader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            self.optimizer.zero_grad()

            output = self.model(image)

            loss = self.criterion(output, mask)
            loss.backward()

            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(self.training_loader)

    def validate(self):
        running_loss: float = 0

        with torch.no_grad():
            for _, (image, mask) in enumerate(self.validation_loader):
                image, mask = image.to(DEVICE), mask.to(DEVICE)

                output = self.model(image)

                loss = self.criterion(output, mask)

                running_loss += loss.item()

        return running_loss / len(self.validation_loader)
    
    def test(self):
        running_loss: float = 0

        with torch.no_grad():
            for _, (image, mask) in enumerate(self.testing_loader):
                image, mask = image.to(DEVICE), mask.to(DEVICE)

                output = self.model(image)

                loss = self.criterion(output, mask)

                running_loss += loss.item()

        return running_loss / len(self.testing_loader)