from __future__ import annotations
from torch import nn
from collections import namedtuple, defaultdict
from typing import DefaultDict, NamedTuple
from nodes import layer, dataset
from app.trash import CustomDataset

import torch
import torchvision.datasets as ds # type: ignore
from torchvision.transforms import v2 # type: ignore


Module: NamedTuple = namedtuple('Module', ['generator', 'func', 'params', 'default_params','tooltip'], defaults=(None, None, None, None, None))
modules: DefaultDict[str, NamedTuple] = defaultdict(lambda: Module("Not present"))

_dtypes = {'float32': v2.ToDtype(torch.float32, scale=True), 
                  'int32': v2.ToDtype(torch.int32), 
                  'int64': v2.ToDtype(torch.int64)} 
transforms_img = [
    {"label":"Resize", "type":'text/tuple', "default_value":"[224, 224]", "user_data": v2.Resize},
    {"label":"ToImage", "type":'blank', "user_data": v2.ToImage},
    {"label":"ToDtype", "type":'combo', "default_value":"float32",
        "items":tuple(_dtypes.keys()), "user_data": _dtypes},
    {"label":"AutoAugment", "type":'blank', "user_data": v2.AutoAugment },
    {"label":"RandomIoUCrop", "type":'blank', "user_data": v2.RandomIoUCrop},
    {"label":"ElasticTransform", "type":'blank', "user_data": v2.ElasticTransform},
    {"label":"Grayscale", "type":'blank', "user_data": v2.Grayscale},
    # {"label":"RandomCrop", "type":'blank', "user_data": v2.RandomCrop},
    {"label":"RandomVerticalFlip", "type":'blank', "user_data": v2.RandomVerticalFlip},
    {"label":"RandomHorizontalFlip", "type":'blank', "user_data": v2.RandomHorizontalFlip}
                    ] 
transforms_setting_img = {"label":"Transform", "type":'collaps', "items":transforms_img}


params = {
    "img_transforms": transforms_setting_img,
    "batch_size" :   {"label":"batch_size", "type":'int', "default_value":64},
    "val_size":     {"label":"val_size", "type":'float', 
                     "max_value": 0.9999999, "max_clamped":True, "default_value":0.2},
    "button_load":  {"label":"Load Dataset", "type":"path", "default_value":"/home/luzinsan/Environments/petrol/data/"},
    "default_train":{'Loss':'L1 Loss','Optimizer':'SGD'},
    "out_features": {"label":"out_features", "type":'int', "default_value":1},
    "out_channels": {"label":"out_channels", "type":'int', "default_value":6},
    "num_features": {"label":"num_features", "type":'int', "default_value":6},
    "output_size":{"label":"output_size", "type":'text/tuple', "default_value":'[1, 2]'},
    "kernel_size":{"label":"kernel_size", "type":'int', "default_value":5},
    "stride":{"label":"stride", "type":'int', "default_value":1},
    "padding":{"label":"padding", "type":'int', "default_value":2},
    "eps":          {"label":"eps", "type":'float', "default_value":0.00001},
    "momentum":{"label":"momentum", "type":'float', "default_value":0.01},
    "affine":{"label":"affine", "type":'bool'},
    "p":{"label":"p", "type":'float', "default_value":0.5},
}

default_params = {
    'Loss':'Cross Entropy Loss',
    'Optimizer':'SGD',
}
fromkeys = lambda d, keys: {x:d.get(x) for x in keys}
modules.update({
    "FashionMNIST":       Module(dataset.DataNode.factory, ds.FashionMNIST, 
                                 (params['img_transforms'], params['batch_size']), 
                                 fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Caltech101":       Module(dataset.DataNode.factory, ds.Caltech101, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Caltech256":       Module(dataset.DataNode.factory, ds.Caltech256, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CarlaStereo":       Module(dataset.DataNode.factory, ds.CarlaStereo, 
                                (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CelebA":       Module(dataset.DataNode.factory, ds.CelebA, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CIFAR10":       Module(dataset.DataNode.factory, ds.CIFAR10, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Cityscapes":       Module(dataset.DataNode.factory, ds.Cityscapes, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CLEVRClassification":       Module(dataset.DataNode.factory, ds.CLEVRClassification, 
                                        (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "EMNIST":       Module(dataset.DataNode.factory, ds.EMNIST, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CocoCaptions":       Module(dataset.DataNode.factory, ds.CocoCaptions, 
                                 (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "EuroSAT":       Module(dataset.DataNode.factory, ds.EuroSAT, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Flowers102":       Module(dataset.DataNode.factory, ds.Flowers102, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Food101":       Module(dataset.DataNode.factory, ds.Food101, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "ImageNet":       Module(dataset.DataNode.factory, ds.ImageNet, 
                             (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "SUN397":       Module(dataset.DataNode.factory, ds.SUN397, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Dataset from File":       Module(dataset.DataNode.factory, CustomDataset, 
                                      (params['val_size'], params['button_load']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    
    "LazyLinear":       Module(layer.LayerNode.factory, nn.LazyLinear, (params['out_features'],)),
    "LazyBatchNorm1d":  Module(layer.LayerNode.factory, nn.LazyBatchNorm1d, (params['eps'], params['momentum'], params['affine'])),
    "LazyBatchNorm2d":  Module(layer.LayerNode.factory, nn.LazyBatchNorm2d, (params['eps'], params['momentum'], params['affine'])),
    "LazyBatchNorm3d":  Module(layer.LayerNode.factory, nn.LazyBatchNorm3d, (params['eps'], params['momentum'], params['affine'])),
    "LazyConv1d":       Module(layer.LayerNode.factory, nn.LazyConv1d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding'])),
    "LazyConv2d":       Module(layer.LayerNode.factory, nn.LazyConv2d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding'])),
    "LazyConv3d":       Module(layer.LayerNode.factory, nn.LazyConv3d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding'])),
    "BatchNorm1d":      Module(layer.LayerNode.factory, nn.BatchNorm1d, (params['num_features'], params['eps'], params['momentum'], params['affine'])),
    "BatchNorm2d":      Module(layer.LayerNode.factory, nn.BatchNorm2d, (params['num_features'], params['eps'], params['momentum'], params['affine'])),
    "BatchNorm3d":      Module(layer.LayerNode.factory, nn.BatchNorm3d, (params['num_features'], params['eps'], params['momentum'], params['affine'])),
    "Flatten":          Module(layer.LayerNode.factory, nn.Flatten),
    "AvgPool2d":        Module(layer.LayerNode.factory, nn.AvgPool2d, (params['kernel_size'], params['stride'])),
    "MaxPool2d":        Module(layer.LayerNode.factory, nn.MaxPool2d, (params['kernel_size'], params['stride'])),
    "AdaptiveAvgPool2d":Module(layer.LayerNode.factory, nn.AdaptiveAvgPool2d, (params['output_size'], )),
    "Dropout":          Module(layer.LayerNode.factory, nn.Dropout, (params['p'], )),
    "ReLU":             Module(layer.LayerNode.factory, nn.ReLU),
    "Softmax":          Module(layer.LayerNode.factory, nn.Softmax),
    "Tanh":             Module(layer.LayerNode.factory, nn.Tanh),
    "GELU":             Module(layer.LayerNode.factory, nn.GELU),
    
    

    "LeNet":            Module(layer.ModuleNode.factory),
    "VGG":              Module(layer.ModuleNode.factory),
    "AlexNet":          Module(layer.ModuleNode.factory),
    "NiN":              Module(layer.ModuleNode.factory),
    "NiN Net":          Module(layer.ModuleNode.factory, tooltip='Архитектура NiN Net')
})

