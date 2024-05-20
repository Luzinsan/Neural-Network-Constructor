from lightning import LightningDataModule
from torch.utils.data import DataLoader
import os
import re


class DataModule(LightningDataModule):
    def __init__(self, dataset_class, **kwargs):
        super().__init__()
        
        params = dict(root=f"{os.getcwd()}/datasets/", 
                      download=True)
        transform = kwargs.get('Transform')
        if transform: params.update({'transform': transform})
        
        self.train = dataset_class(train=True, **params)
        self.val = dataset_class(train=False, **params)
        
        if transform and (resize := re.search('(?<=Resize\(size=)(\[.*\])', transform.extra_repr())):
            self.shape = eval(resize[0])
        else: self.shape = self.train.data.shape[-2:]
    
        for attr, value in kwargs.items():
            setattr(self, attr, value)
            
        assert hasattr(self, 'batch_size')
        
    
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=7)
    
