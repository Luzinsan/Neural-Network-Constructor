
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import os


class DataModule(LightningDataModule):
    def __init__(self, dataset_class, transforms=None, **params):
        super().__init__()
        self.train = dataset_class(f"{os.getcwd()}/datasets/", train=True, download=True, transform=transforms)
        self.val = dataset_class(f"{os.getcwd()}/datasets/", train=False, download=True, transform=transforms)
    
        for attr, value in params.items():
            setattr(self, attr, value)
        
        

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=7)
    
