import torch
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import os

VALID_TYPES = ['train', 'val', 'test']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, type: str, transforms=None):
        if type not in VALID_TYPES:
            raise ValueError(f'Invalid type. Must be one of {VALID_TYPES}')
        
        self.type = type
        self.transforms = transforms

        self.base_path = os.path.join(os.getcwd(), 'dataset', 'archive')
        self.image_path = os.path.join(self.base_path, 'Images')
        self.mask_path = os.path.join(self.base_path, 'Masks')

        self.csv = pd.read_csv(os.path.join(self.base_path, 'metadata_prepped.csv'))
        self.csv = self.csv[self.csv['dataset'] == self.type]

    def __len__(self):
        return int(len(self.csv))

    def __getitem__(self, index):
        image_name = self.csv.iloc[index, 0]
        mask_name = self.csv.iloc[index, 1]

        image = read_image(os.path.join(self.image_path, image_name), ImageReadMode.RGB)
        mask = read_image(os.path.join(self.mask_path, mask_name), ImageReadMode.GRAY)

        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)

        return image, mask