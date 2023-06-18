from model.UNet import UNet
from model.Trainer import UNetTrainer
from model.DiceLoss import DiceLoss
from dataset.Dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
import torch

torch.manual_seed(42)

def main():
    """ Main function """
    # Create UNet model
    model_name = "UNet3"
    
    model = UNet3()
    # model.load_weights('./model/weights/UNet.pth')

    training_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.2),
        transforms.Resize((576, 576)),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((576, 576)),
        transforms.ToTensor()
    ])

    training_set = Dataset(type='train',  transforms=training_transforms)
    validation_set = Dataset(type='val', transforms=val_transforms)
    # test_set = Dataset(type='test', transforms=transform)

    training_loader = DataLoader(training_set, batch_size=2, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_set, batch_size=2, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=0)

    trainer = UNetTrainer(model=model,
                          optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999)), 
                          criterion=DiceLoss())
    
    trainer.train(epochs=20, model_name=model_name, train_loader=training_loader, validation_loader=validation_loader, log_dir=f'./model/logs/{model_name}.csv')

def predict():
    # model = UUNet()
    model = UNet()
    model.load_weights('./model/weights/UNet.pth')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((576, 576)),
        transforms.ToTensor()
    ])

    training_set = Dataset(type='train',  transforms=transform)

    image = training_set[1][0]

    image_mask = training_set[1][1]

    mask = model.predict(image, out_threshold=0.7)

    # show original image, mask and predicted mask
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0), cmap='gray')
    ax[1].imshow(image_mask.permute(1, 2, 0), cmap='gray')
    ax[2].imshow(mask, cmap='gray')

    # set titles
    ax[0].set_title('Original Image')
    ax[1].set_title('Mask')
    ax[2].set_title('Predicted Mask')

    plt.savefig("UUNet_Test.png")

def compare(model_A: nn.Module, model_B: nn.Module):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ColorJitter(brightness=0.2),
        transforms.Resize((576, 576)),
        transforms.ToTensor()
    ])

    training_set = Dataset(type='val',  transforms=transform)

    datapoint = training_set[132]

    image = datapoint[0]

    image_mask = datapoint[1]

    threadhold = 0.2

    model_A_mask = model_A.predict(image, threadhold)
    model_B_mask = model_B.predict(image, threadhold)

    # show original image, mask and predicted mask
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0), cmap='gray')
    ax[1].imshow(image_mask.permute(1, 2, 0), cmap='gray')
    ax[2].imshow(model_A_mask, cmap='gray')
    ax[3].imshow(model_B_mask, cmap='gray')

    # set titles
    ax[0].set_title('Original Image')
    ax[1].set_title('Mask')
    ax[2].set_title(f'Predicted Mask {model_A.name}')
    ax[3].set_title(f'Predicted Mask {model_B.name}')


    plt.savefig("Comparison.png")

if __name__ == "__main__":
    main()
    # predict()
    # compare()
    