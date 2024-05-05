import dearpygui.dearpygui as dpg

from torch import nn
import torch
import torchvision.datasets as ds
from torchvision.transforms import v2 

import sys, os
sys.path.insert(1, os.getcwd())


from config.settings import *
from core.node_editor import NodeEditor
from core.dragndrop import DragSource, DragSourceContainer
from nodes.dataset import DataNode
from nodes.layer import LayerNode, ModuleNode
from nodes.tools import ViewNode_2D
from app.pipeline import Pipeline
from app.trash import CustomDataset


    
class App:

    def __init__(self):

        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.right_panel = dpg.generate_uuid()
        self.node_editor = NodeEditor()
        self.plugins = []
        WIDTH = 150

        #region datasets
        self.dataset_container = DragSourceContainer("Датасеты", 150, -500)
        
        _dtypes = {'float32': v2.ToDtype(torch.float32, scale=True), 
                  'int32': v2.ToDtype(torch.int32), 
                  'int64': v2.ToDtype(torch.int64)} 
        
        transforms = [{"label":"Resize", "type":'text/tuple', "default_value":"28, 28", "user_data": v2.Resize},
                      {"label":"ToImage", "type":'blank', "user_data": v2.ToImage},
                      {"label":"ToDtype", "type":'combo', "default_value":"float32",
                       "items":tuple(_dtypes.keys()), "user_data": _dtypes},
        ]
        
        transforms_img = transforms + [
            {"label":"AutoAugment", "type":'blank', "user_data": v2.AutoAugment },
            {"label":"RandomIoUCrop", "type":'blank', "user_data": v2.RandomIoUCrop},
            {"label":"ElasticTransform", "type":'blank', "user_data": v2.ElasticTransform},
            {"label":"Grayscale", "type":'blank', "user_data": v2.Grayscale},
            # {"label":"RandomCrop", "type":'blank', "user_data": v2.RandomCrop},
            {"label":"RandomVerticalFlip", "type":'blank', "user_data": v2.RandomVerticalFlip},
            {"label":"RandomHorizontalFlip", "type":'blank', "user_data": v2.RandomHorizontalFlip}
                          ] 
        transforms_setting = {"label":"Transforms", "type":'collaps', "width":WIDTH, "items":transforms}
        transforms_setting_img = {"label":"Transforms", "type":'collaps', "width":WIDTH, "items":transforms_img}
        datasets = {
            "FashionMNIST": DragSource("FashionMNIST", 
                                        DataNode.factory, 
                                        ds.FashionMNIST,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Caltech101": DragSource("Caltech101", 
                                        DataNode.factory, 
                                        ds.Caltech101,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Caltech256": DragSource("Caltech256", 
                                        DataNode.factory, 
                                        ds.Caltech256,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CarlaStereo": DragSource("CarlaStereo", 
                                        DataNode.factory, 
                                        ds.CarlaStereo,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CelebA": DragSource("CelebA", 
                                        DataNode.factory, 
                                        ds.CelebA,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CIFAR10": DragSource("CIFAR10", 
                                        DataNode.factory, 
                                        ds.CIFAR10,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Cityscapes": DragSource("Cityscapes", 
                                        DataNode.factory, 
                                        ds.Cityscapes,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CLEVRClassification": DragSource("CLEVRClassification", 
                                        DataNode.factory, 
                                        ds.CLEVRClassification,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "EMNIST": DragSource("EMNIST", 
                                        DataNode.factory, 
                                        ds.EMNIST,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CocoCaptions": DragSource("CocoCaptions", 
                                        DataNode.factory, 
                                        ds.CocoCaptions,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "EuroSAT": DragSource("EuroSAT", 
                                        DataNode.factory, 
                                        ds.EuroSAT,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Flowers102": DragSource("Flowers102", 
                                        DataNode.factory, 
                                        ds.Flowers102,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Food101": DragSource("Food101", 
                                        DataNode.factory, 
                                        ds.Food101,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "ImageNet": DragSource("ImageNet", 
                                        DataNode.factory, 
                                        ds.ImageNet,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "SUN397": DragSource("SUN397", 
                                        DataNode.factory, 
                                        ds.SUN397,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Dataset from File": DragSource("FileData", 
                                        DataNode.factory, 
                                        CustomDataset,
                                        (
                                            {"label":"val_size", "type":'float', "step":1, "width":WIDTH, "min_value":0.00000001, "min_clamped":True, 
                                             "max_value": 0.9999999, "max_clamped":True, "default_value":0.2},
                                            {"label":"Load Dataset", "type":"path", "default_value":"/home/luzinsan/Environments/petrol/data/"},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'L1 Loss','Optimizer':'SGD'}),                            
            }
        self.dataset_container.add_drag_source(datasets.values())
        #endregion
        #region layers
        init_params = {'Default':None, 'Normal': Pipeline.init_normal, 'Xavier': Pipeline.init_xavier } 
        init_setting = {"label":"Initialization", "type":'combo', "default_value":'Default', "width":WIDTH,
                "items":tuple(init_params.keys()), "user_data":init_params}
        
        layers = {
            "LazyLinear":   DragSource("LazyLinear", 
                                        LayerNode.factory, 
                                        nn.LazyLinear,
                                        (
                                            {"label":"out_features", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":1},
                                            init_setting
                                        )),
            "LazyBatchNorm1d":   DragSource("LazyBatchNorm1d", 
                                        LayerNode.factory, 
                                        nn.LazyBatchNorm1d,
                                        (
                                            {"label":"eps", "type":'float', "step":0.00001, "width":WIDTH, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "width":WIDTH, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                            init_setting
                                        )),
            "LazyBatchNorm2d":   DragSource("LazyBatchNorm2d", 
                                        LayerNode.factory, 
                                        nn.LazyBatchNorm2d,
                                        (
                                            {"label":"eps", "type":'float', "step":0.00001, "width":WIDTH, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "width":WIDTH, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                            init_setting
                                        )),
            "LazyBatchNorm3d":   DragSource("LazyBatchNorm3d", 
                                        LayerNode.factory, 
                                        nn.LazyBatchNorm3d,
                                        (
                                            {"label":"eps", "type":'float', "step":0.00001, "width":WIDTH, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "width":WIDTH, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                            init_setting
                                        )),
            "LazyConv1d":   DragSource("LazyConv1d", 
                                        LayerNode.factory, 
                                        nn.LazyConv1d,
                                        (
                                            {"label":"out_channels", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":5},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":1},
                                            {"label":"padding", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            init_setting
                                        )),
            "LazyConv2d":   DragSource("LazyConv2d", 
                                        LayerNode.factory, 
                                        nn.LazyConv2d,
                                        (
                                            {"label":"out_channels", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":5},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":1},
                                            {"label":"padding", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            init_setting
                                        )),
            "LazyConv3d":   DragSource("LazyConv3d", 
                                        LayerNode.factory, 
                                        nn.LazyConv3d,
                                        (
                                            {"label":"out_channels", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":5},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":1},
                                            {"label":"padding", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            init_setting
                                        )),
            "BatchNorm1d":   DragSource("BatchNorm1d", 
                                        LayerNode.factory, 
                                        nn.BatchNorm1d,
                                        (
                                            {"label":"num_features", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"eps", "type":'float', "step":0.00001, "width":WIDTH, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "width":WIDTH, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                            init_setting
                                        )),
            "BatchNorm2d":   DragSource("BatchNorm2d", 
                                        LayerNode.factory, 
                                        nn.BatchNorm2d,
                                        (
                                            {"label":"num_features", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"eps", "type":'float', "step":0.00001, "width":WIDTH, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "width":WIDTH, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                            init_setting
                                        )),
            "BatchNorm3d":   DragSource("BatchNorm3d", 
                                        LayerNode.factory, 
                                        nn.BatchNorm3d,
                                        (
                                            {"label":"num_features", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"eps", "type":'float', "step":0.00001, "width":WIDTH, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "width":WIDTH, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                            init_setting
                                        )),
        
            "Flatten":        DragSource("Flatten",
                                        LayerNode.factory,
                                        nn.Flatten),
            "AvgPool2d":      DragSource("MaxPool2d",
                                        LayerNode.factory,
                                        nn.AvgPool2d,
                                        (
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "MaxPool2d":      DragSource("MaxPool2d",
                                        LayerNode.factory,
                                        nn.MaxPool2d,
                                        (
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "AdaptiveAvgPool2d":DragSource("AdaptiveAvgPool2d",
                                        LayerNode.factory,
                                        nn.AdaptiveAvgPool2d,
                                        (
                                            {"label":"output_size", "type":'text/tuple', "width":WIDTH, "default_value":'(1, 2)'},
                                        )),
            "Dropout":      DragSource("Dropout",
                                        LayerNode.factory,
                                        nn.Dropout,
                                        (
                                            {"label":"p", "type":'float', "width":WIDTH, "default_value":0.5},
                                        )),
            "ReLU":         DragSource("ReLU",
                                        LayerNode.factory,
                                        nn.ReLU,
                                        ),
            "Softmax":      DragSource("Softmax",
                                        LayerNode.factory,
                                        nn.Softmax),
            "Tanh":         DragSource("Tanh",
                                        LayerNode.factory,
                                        nn.Tanh),
            "GELU":         DragSource("GELU",
                                        LayerNode.factory,
                                        nn.GELU),
        }
        self.layer_container = DragSourceContainer("Слои|ф.активации", 150, 0)
        self.layer_container.add_drag_source(layers.values())
        #endregion
        #region architectures
    
        archs = {
            'LeNet': DragSource("LeNet", 
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':6,"kernel_size":5,"stride":1,"padding":3,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":5,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':120, "Initialization":"Xavier"}), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':84, "Initialization":"Xavier"}), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':10, "Initialization":"Xavier"}),
                                ),
                                node_params={"node_editor":self.node_editor}),
            'VGG': DragSource("VGG", 
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':32,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':64,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':64,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':120, "Initialization":"Xavier"}), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':84, "Initialization":"Xavier"}), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':10, "Initialization":"Xavier"}),
                                ),
                                node_params={"node_editor":self.node_editor}),
            'AlexNet': DragSource("AlexNet", 
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':96,"kernel_size":11,"stride":4,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':256,"kernel_size":5,"stride":1,"padding":2,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':256,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),(layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':4096, "Initialization":"Xavier"}), (layers['ReLU'], ),(layers['Dropout'], {'p':0.5}),
                                    (layers['LazyLinear'], {'out_features':84, "Initialization":"Xavier"}), (layers['ReLU'], ),(layers['Dropout'], {'p':0.5}),
                                    (layers['LazyLinear'], {'out_features':10, "Initialization":"Xavier"}),
                                ),
                                node_params={"node_editor":self.node_editor}),
            'NiN': DragSource("NiN",
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':11, 'stride':4,'padding':0}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0}),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0}),(layers['ReLU'], )
                                ),
                                node_params={"node_editor":self.node_editor}),
        }
        archs['NiN Net'] = DragSource("NiN Net",
                                ModuleNode.factory,
                                (
                                    (datasets['FashionMNIST'], {"batch_size": 128}),
                                    (archs['NiN'], {'out_channels':[96]*3, 'kernel_size':[11,1,1], 'stride':[4,1,1],'padding':[0]*3}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['NiN'], {'out_channels':[256]*3, 'kernel_size':[5,1,1], 'stride':[1]*3,'padding':[2,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['NiN'], {'out_channels':[384]*3, 'kernel_size':[3,1,1], 'stride':[1]*3,'padding':[1,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (layers['Dropout'], {'p':0.5}),
                                    (archs['NiN'], {'out_channels':[10]*3, 'kernel_size':[3,1,1], 'stride':[1]*3,'padding':[1,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (layers['AdaptiveAvgPool2d'], {'output_size':'(1,1)'}),(layers['Flatten'],)
                                ),
                                node_params={"node_editor":self.node_editor})
        
        self.archs_container = DragSourceContainer("Модули", 150, 0)
        self.archs_container.add_drag_source(archs.values())
        #endregion
        #region tools
        # tools = {
        #     "Progress Board":DragSource("Progress Board", 
        #                                 ViewNode_2D.factory),
        # }
        # self.tool_container = DragSourceContainer("Tools", 150, 0)
        # self.tool_container.add_drag_source(tools.values())
        #endregion
        
        

        
    def update(self):

        with dpg.mutex():
            dpg.delete_item(self.left_panel, children_only=True)
            self.dataset_container.submit(self.left_panel)
            self.layer_container.submit(self.left_panel)

            dpg.delete_item(self.right_panel, children_only=True)
            self.archs_container.submit(self.right_panel)
            # self.tool_container.submit(self.right_panel)


    def add_plugin(self, name, callback):
        self.plugins.append((name, callback))
                

   

    def start(self):
        #dpg.setup_registries()
        dpg.set_viewport_title("Deep Learning Constructor")
        dpg.show_viewport()

        with dpg.window() as main_window:

            with dpg.menu_bar():
                with dpg.menu(label="Операции"):
                    dpg.add_menu_item(label="Сбросить", callback=self.node_editor.clear)

                with dpg.menu(label="Плагины"):
                    for plugin in self.plugins:
                        dpg.add_menu_item(label=plugin[0], callback=plugin[1])

            with dpg.group(horizontal=True) as group:
                # left panel
                with dpg.group(id=self.left_panel):
                    self.dataset_container.submit(self.left_panel)
                    self.layer_container.submit(self.left_panel)

                # center panel
                self.node_editor.submit(group)

                # right panel
                with dpg.group(id=self.right_panel):
                    self.archs_container.submit(self.right_panel)
                    # self.tool_container.submit(self.right_panel)
                    
        
        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()
        




app = App()
app.start()
