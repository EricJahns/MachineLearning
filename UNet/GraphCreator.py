import pandas as pd
from matplotlib import pyplot as plt

class GraphCreator():
    def __init__(self):
        pass

    def create_metrics(self, name: str, title: str, df: pd.DataFrame):
        # put both train and validation metrics in one graph
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # plot train loss
        axs[0, 0].plot(self.df['epoch'], self.df['train_loss'], label='Train Loss')
        axs[0, 0].plot(self.df['epoch'], self.df['validation_loss'], label='Validation Loss')

        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend()

        # plot train pixel accuracy
        axs[0, 1].plot(self.df['epoch'], self.df['train_pixel_accuracy'], label='Train Pixel Accuracy')
        axs[0, 1].plot(self.df['epoch'], self.df['validation_pixel_accuracy'], label='Validation Pixel Accuracy')

        axs[0, 1].set_title('Pixel Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend()

        # plot train IoU
        axs[1, 0].plot(self.df['epoch'], self.df['train_IoU'], label='Train IoU')
        axs[1, 0].plot(self.df['epoch'], self.df['validation_IoU'], label='Validation IoU')

        axs[1, 0].set_title('IoU')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend()

        # plot train dice score
        axs[1, 1].plot(self.df['epoch'], self.df['train_dice_score'], label='Train Dice Score')
        axs[1, 1].plot(self.df['epoch'], self.df['validation_dice_score'], label='Validation Dice Score')

        axs[1, 1].set_title('Dice Score')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend()

        # set graph title
        fig.suptitle(title)

        # save graph
        plt.savefig(f'./model/graphs/{name}')

if __name__ == "__main__":
    df = pd.read_csv('./model/dfs/UNet.csv')
    graph_creator = GraphCreator()
    graph_creator.create_metrics('UNet.png', 'UNet', df)


