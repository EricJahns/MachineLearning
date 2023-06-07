import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from .UNet import UNet
from .Metrics import Metrics
from tqdm import tqdm, trange
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metrics = Metrics()

class Trainer():
    def __init__(self, model:UNet, training_loader:DataLoader, validation_loader:DataLoader, testing_loader:DataLoader, optimizer:Optimizer, criterion):
        self.model = model
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler()

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

    def train(self, epochs, model_name, log_dir=None):
        print(f"Training {model_name}")
        self.model = self.model.to(DEVICE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, verbose=True)

        for epoch in trange(1, epochs+1):
            tqdm.write(f'Epoch: {epoch}')
            self.log['epoch'].append(epoch)

            loss = self.step()
            scheduler.step(loss)
            self.log['train_loss'].append(loss)

            validation_loss = self.validate()
            self.log['validation_loss'].append(validation_loss)

            tqdm.write(f'Train Loss: {loss : .6f} | Val Loss: {validation_loss : .6f}')
            tqdm.write("")

        self.model.save_weights(f'./model/weights/{model_name}.pth')

        if log_dir:
            self.save_log(log_dir)

    def step(self):
        self.model.train()

        running_loss: float = 0
        for _, (image, mask) in enumerate(self.training_loader):
            image, mask = image.to(DEVICE), mask.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                output = self.model(image)
                loss = self.criterion(output, mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        self.step_training_metrics(mask.cpu().numpy(), output.cpu().detach().numpy())
        
        return running_loss / len(self.training_loader)

    def validate(self):
        self.model.eval()

        running_loss: float = 0
        with torch.no_grad():
            for _, (image, mask) in enumerate(self.validation_loader):
                image, mask = image.to(DEVICE), mask.to(DEVICE)

                output = self.model(image)

                loss = self.criterion(output, mask)

                running_loss += loss.item()

        self.step_validation_metrics(mask.cpu().numpy(), output.cpu().detach().numpy())

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
    
    def step_validation_metrics(self, y_true, y_pred):
        self.model.eval()

        with torch.no_grad():
            pixel_accuracy = metrics.pixel_accuracy(y_true, y_pred)
            IoU = metrics.IoU(y_true, y_pred)
            dice_score = metrics.dice_score(y_true, y_pred)

        self.log['validation_pixel_accuracy'].append(pixel_accuracy)
        self.log['validation_IoU'].append(IoU)
        self.log['validation_dice_score'].append(dice_score)

        tqdm.write(f'Val   Pixel Accuracy: {pixel_accuracy : .3f} | Val   IoU: {IoU : .3f} | Val   Dice Score: {dice_score : .3f}')

    def step_training_metrics(self, y_true, y_pred):
        self.model.eval()

        with torch.no_grad():
            pixel_accuracy = metrics.pixel_accuracy(y_true, y_pred)
            IoU = metrics.IoU(y_true, y_pred)
            dice_score = metrics.dice_score(y_true, y_pred)

        self.log['train_pixel_accuracy'].append(pixel_accuracy)
        self.log['train_IoU'].append(IoU)
        self.log['train_dice_score'].append(dice_score)

        tqdm.write(f'Train Pixel Accuracy: {pixel_accuracy : .3f} | Train IoU: {IoU : .3f} | Train Dice Score: {dice_score : .3f}')
    
    def save_log(self, log_dir):
        log = pd.DataFrame(self.log)
        log.to_csv(log_dir, index=False)