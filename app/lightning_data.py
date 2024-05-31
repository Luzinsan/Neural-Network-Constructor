from lightning import LightningDataModule
from torch.utils.data import DataLoader
import os
import re


class DataModule(LightningDataModule):
    def __init__(self, dataset_class, **kwargs):
        super().__init__()
        args: tuple = dataset_class.__init__.__code__.co_varnames
        has_train_arg = 'train' in args 
        has_split_arg = 'split' in args

        params = dict(root=f"{os.getcwd()}/datasets/", 
                      download=True)
        
        name_transform = next((key for key in ['transform', 'transforms'] if key in args), None)
        if name_transform:
            transform = kwargs.get('Transform', None)
            if transform: params.update({name_transform: transform})
        
        if has_train_arg: params.update({'train': True})
        elif has_split_arg: params.update({'split':'train'})
        self.train = dataset_class(**params)
        
        if has_train_arg: params.update({'train': False})
        elif has_split_arg: params.update({'split':'val'})
        self.val = dataset_class(**params)
        
        if transform and (resize := re.search('(?<=Resize\(size=)(\[.*\])', transform.extra_repr())):
            self.shape = eval(resize[0])
        else: self.shape = self.train[0][0].size
    
        for attr, value in kwargs.items():
            setattr(self, attr, value)
            
        assert hasattr(self, 'batch_size')
        
    
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=7)
    
