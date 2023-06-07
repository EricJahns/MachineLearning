import pandas as pd
from matplotlib import pyplot as plt

class GraphCreator():
    def __init__(self, log: pd.DataFrame):
        self.log = log



    def create_train_metrics_graph(self, name: str):
        # 2x2 graph
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # plot train loss
        axs[0, 0].plot(self.log['epoch'], self.log['train_loss'], label='Train Loss')
        axs[0, 0].set_title('Train Loss')
        axs[0, 0].set_xlabel('Epoch')

        # plot train pixel accuracy
        axs[0, 1].plot(self.log['epoch'], self.log['train_pixel_accuracy'], label='Train Pixel Accuracy')
        axs[0, 1].set_title('Train Pixel Accuracy')
        axs[0, 1].set_xlabel('Epoch')

        # plot train IoU
        axs[1, 0].plot(self.log['epoch'], self.log['train_IoU'], label='Train IoU')
        axs[1, 0].set_title('Train IoU')
        axs[1, 0].set_xlabel('Epoch')

        # plot train dice score
        axs[1, 1].plot(self.log['epoch'], self.log['train_dice_score'], label='Train Dice Score')
        axs[1, 1].set_title('Train Dice Score')
        axs[1, 1].set_xlabel('Epoch')

        # set graph title
        fig.suptitle('Training Metrics')

        # save graph
        plt.savefig(f'./model/graphs/{name}')

    def create_val_metrics_graph(self, name: str):
        # 2x2 graph
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # plot validation loss
        axs[0, 0].plot(self.log['epoch'], self.log['validation_loss'], label='Validation Loss')
        axs[0, 0].set_title('Validation Loss')
        axs[0, 0].set_xlabel('Epoch')

        # plot validation pixel accuracy
        axs[0, 1].plot(self.log['epoch'], self.log['val_pixel_accuracy'], label='Validation Pixel Accuracy')
        axs[0, 1].set_title('Validation Pixel Accuracy')
        axs[0, 1].set_xlabel('Epoch')

        # plot validation IoU
        axs[1, 0].plot(self.log['epoch'], self.log['val_IoU'], label='Validation IoU')
        axs[1, 0].set_title('Validation IoU')
        axs[1, 0].set_xlabel('Epoch')

        # plot validation dice score
        axs[1, 1].plot(self.log['epoch'], self.log['val_dice_score'], label='Validation Dice Score')
        axs[1, 1].set_title('Validation Dice Score')
        axs[1, 1].set_xlabel('Epoch')

        # set graph title
        fig.suptitle('Validation Metrics')

        # save graph
        plt.savefig(f'./model/graphs/{name}')

if __name__ == "__main__":
    graph_creator = GraphCreator(pd.read_csv('./model/logs/UUNet.csv'))
    graph_creator.create_train_metrics_graph('UUNet_Train_Metrics.png')
    graph_creator.create_val_metrics_graph('UUNet_Val_Metrics.png')

