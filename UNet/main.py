from model.UNet import UNet
from model.UUNet import UUNet
from model.Trainer import Trainer
from model.DiceLoss import DiceLoss
from dataset.Dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import torch

torch.manual_seed(42)

def main():
    """ Main function """
    # Create UNet model
    model_name = "UNet"
    
    model = UNet()
    # model.load_weights('./model/weights/UUNet.pth')

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

    training_loader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_set, batch_size=4, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=0)

    trainer = Trainer(model=model, 
                      training_loader=training_loader, 
                      validation_loader=validation_loader, 
                      testing_loader=None, 
                      optimizer=torch.optim.Adam(model.parameters()), 
                      criterion=DiceLoss())
    
    trainer.train(epochs=20, model_name=model_name, log_dir=f'./model/logs/{model_name}.csv')

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

    image = training_set[5][0]

    image_mask = training_set[5][1]

    mask = model.predict(image)

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

def compare():
    uunet = UUNet()
    unet = UNet()

    unet.load_weights('./model/weights/UNet.pth')
    uunet.load_weights('./model/weights/UUNet.pth')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ColorJitter(brightness=0.2),
        transforms.Resize((576, 576)),
        transforms.ToTensor()
    ])

    training_set = Dataset(type='train',  transforms=transform)

    datapoint = training_set[152]

    image = datapoint[0]

    image_mask = datapoint[1]

    threadhold = 0.7

    unet_mask = unet.predict(image, threadhold)
    uunet_mask = uunet.predict(image, threadhold)

    # show original image, mask and predicted mask
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0), cmap='gray')
    ax[1].imshow(image_mask.permute(1, 2, 0), cmap='gray')
    ax[2].imshow(unet_mask, cmap='gray')
    ax[3].imshow(uunet_mask, cmap='gray')

    # set titles
    ax[0].set_title('Original Image')
    ax[1].set_title('Mask')
    ax[2].set_title('Predicted Mask UNet')
    ax[3].set_title('Predicted Mask UUNet')


    plt.savefig("Comparison.png")

if __name__ == "__main__":
    main()
    # predict()
    # compare()
    