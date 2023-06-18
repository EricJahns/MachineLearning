import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from .UNet import UNet
from .Metrics import Metrics
from tqdm import tqdm, trange
import pandas as pd
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metrics = Metrics()

class UNetTrainer():
    def __init__(self, model:UNet, optimizer:Optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.log = {
            'epoch': [],
            'train_loss': [],
            'validation_loss': [],
            'train_pixel_accuracy': [],
            'validation_pixel_accuracy': [],
            'train_IoU': [],
            'validation_IoU': [],
            'train_dice_score': [],
            'validation_dice_score': []
        }

    def train(self, epochs:int, model_name:str, train_loader:DataLoader, validation_loader:DataLoader, test_loader:DataLoader=None, log_dir=None,):
        print(f"Training {model_name}")
        self.model = self.model.to(DEVICE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=epochs//2, T_mult=1, eta_min=1e-6, last_epoch=-1)

        for epoch in trange(1, epochs+1):
            tqdm.write(f'Epoch: {epoch}')
            self.log['epoch'].append(epoch)

            loss = self.step(train_loader)
            self.log['train_loss'].append(loss)

            scheduler.step(loss)

            validation_loss = self.validate(validation_loader)
            self.log['validation_loss'].append(validation_loss)

            tqdm.write(f'Train Loss: {loss : .6f} | Val Loss: {validation_loss : .6f}')
            tqdm.write("")

        if test_loader:
            loss = self.test(test_loader)
            tqdm.write(f'Test Loss: {loss : .6f}')

        self.model.save_weights(f'./model/weights/{model_name}.pth')

        if log_dir:
            self.save_log(log_dir)

    def step(self, train_loader:DataLoader):
        self.model.train()

        metrics = np.zeros(3, dtype=np.float64)

        running_loss: float = 0
        for _, (image, mask) in enumerate(train_loader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)

            output = self.model(image)
            loss = self.criterion(output, mask)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            metrics = np.add(metrics, self.step_metrics(mask.cpu(), output.cpu().detach()))

        self.record_training_metrics(np.divide(metrics, len(train_loader)))
        return running_loss / len(train_loader)

    def validate(self, validation_loader:DataLoader):
        self.model.eval()

        metrics = np.zeros(3)

        running_loss: float = 0
        with torch.no_grad():
            for _, (image, mask) in enumerate(validation_loader):
                image, mask = image.to(DEVICE), mask.to(DEVICE)

                output = self.model(image)

                loss = self.criterion(output, mask)

                running_loss += loss.item()

                metrics = np.add(metrics, self.step_metrics(mask.cpu(), output.cpu().detach()))

        self.record_validation_metrics(np.divide(metrics, len(validation_loader)))
        return running_loss / len(validation_loader)
    
    def test(self, test_loader:DataLoader):
        running_loss: float = 0

        with torch.no_grad():
            for _, (image, mask) in enumerate(test_loader):
                image, mask = image.to(DEVICE), mask.to(DEVICE)

                output = self.model(image)

                loss = self.criterion(output, mask)

                running_loss += loss.item()

        return running_loss / len(test_loader)

    def step_metrics(self, y_true, y_pred):
        self.model.eval()

        with torch.no_grad():
            pixel_accuracy = metrics.pixel_accuracy(y_true, y_pred)
            IoU = metrics.IoU(y_true, y_pred)
            dice_score = metrics.dice_score(y_true, y_pred)

        return np.array([pixel_accuracy, IoU, dice_score])
    
    def record_training_metrics(self, metrics: np.ndarray):
        self.log['train_pixel_accuracy'].append(metrics[0])
        self.log['train_IoU'].append(metrics[1])
        self.log['train_dice_score'].append(metrics[2])

        tqdm.write(f'Train Pixel Accuracy: {metrics[0] : .3f} | Train IoU: {metrics[1] : .3f} | Train Dice Score: {metrics[2] : .3f}')

    def record_validation_metrics(self, metrics: np.ndarray):
        self.log['validation_pixel_accuracy'].append(metrics[0])
        self.log['validation_IoU'].append(metrics[1])
        self.log['validation_dice_score'].append(metrics[2])

        tqdm.write(f'Val   Pixel Accuracy: {metrics[0] : .3f} | Val   IoU: {metrics[1] : .3f} | Val   Dice Score: {metrics[2] : .3f}')
    
    def save_log(self, log_dir):
        log = pd.DataFrame(self.log)
        log.to_csv(log_dir, index=False)