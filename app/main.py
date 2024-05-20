import dearpygui.dearpygui as dpg
import torchvision.datasets as ds # type: ignore
from torchvision.transforms import v2 # type: ignore

from torch import nn
import torch

import sys, os
sys.path.insert(1, os.getcwd())

from config.setup import *
from config.settings import Configs

from core.node_editor import NodeEditor
from core.dragndrop import DragSource, DragSourceContainer

from nodes.dataset import DataNode
from nodes.layer import LayerNode, ModuleNode
from app.trash import CustomDataset


    
class App:

    def __init__(self):

        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.center_panel = dpg.generate_uuid()
        self.right_panel = dpg.generate_uuid()
        self.node_editor = NodeEditor()
    
        
        #region datasets
        self.dataset_container = DragSourceContainer("Датасеты", 150, -500)
        
        _dtypes = {'float32': v2.ToDtype(torch.float32, scale=True), 
                  'int32': v2.ToDtype(torch.int32), 
                  'int64': v2.ToDtype(torch.int64)} 
        
        transforms = [{"label":"Resize", "type":'text/tuple', "default_value":"224, 224", "user_data": v2.Resize},
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
        transforms_setting = {"label":"Transform", "type":'collaps', "items":transforms}
        transforms_setting_img = {"label":"Transform", "type":'collaps', "items":transforms_img}
        datasets = {
            "FashionMNIST": DragSource("FashionMNIST", 
                                        DataNode.factory, 
                                        ds.FashionMNIST,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Caltech101": DragSource("Caltech101", 
                                        DataNode.factory, 
                                        ds.Caltech101,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Caltech256": DragSource("Caltech256", 
                                        DataNode.factory, 
                                        ds.Caltech256,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CarlaStereo": DragSource("CarlaStereo", 
                                        DataNode.factory, 
                                        ds.CarlaStereo,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CelebA": DragSource("CelebA", 
                                        DataNode.factory, 
                                        ds.CelebA,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CIFAR10": DragSource("CIFAR10", 
                                        DataNode.factory, 
                                        ds.CIFAR10,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Cityscapes": DragSource("Cityscapes", 
                                        DataNode.factory, 
                                        ds.Cityscapes,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CLEVRClassification": DragSource("CLEVRClassification", 
                                        DataNode.factory, 
                                        ds.CLEVRClassification,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "EMNIST": DragSource("EMNIST", 
                                        DataNode.factory, 
                                        ds.EMNIST,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "CocoCaptions": DragSource("CocoCaptions", 
                                        DataNode.factory, 
                                        ds.CocoCaptions,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "EuroSAT": DragSource("EuroSAT", 
                                        DataNode.factory, 
                                        ds.EuroSAT,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Flowers102": DragSource("Flowers102", 
                                        DataNode.factory, 
                                        ds.Flowers102,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Food101": DragSource("Food101", 
                                        DataNode.factory, 
                                        ds.Food101,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "ImageNet": DragSource("ImageNet", 
                                        DataNode.factory, 
                                        ds.ImageNet,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "SUN397": DragSource("SUN397", 
                                        DataNode.factory, 
                                        ds.SUN397,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "min_value":2, "min_clamped":True, "default_value":64},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            "Dataset from File": DragSource("FileData", 
                                        DataNode.factory, 
                                        CustomDataset,
                                        (
                                            {"label":"val_size", "type":'float', "step":1, "min_value":0.00000001, "min_clamped":True, 
                                             "max_value": 0.9999999, "max_clamped":True, "default_value":0.2},
                                            {"label":"Load Dataset", "type":"path", "default_value":"/home/luzinsan/Environments/petrol/data/"},
                                            transforms_setting_img
                                        ),
                                        default_params={'Loss':'L1 Loss','Optimizer':'SGD'}),                            
            }
        self.dataset_container.add_drag_source(datasets.values())
        #endregion
        #region layers
        
        
        layers = {
            "LazyLinear":   DragSource("LazyLinear", 
                                        LayerNode.factory, 
                                        nn.LazyLinear,
                                        (
                                            {"label":"out_features", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":1},
                                        )),
            "LazyBatchNorm1d":   DragSource("LazyBatchNorm1d", 
                                        LayerNode.factory, 
                                        nn.LazyBatchNorm1d,
                                        (
                                            {"label":"eps", "type":'float', "step":0.00001, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                        )),
            "LazyBatchNorm2d":   DragSource("LazyBatchNorm2d", 
                                        LayerNode.factory, 
                                        nn.LazyBatchNorm2d,
                                        (
                                            {"label":"eps", "type":'float', "step":0.00001, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                        )),
            "LazyBatchNorm3d":   DragSource("LazyBatchNorm3d", 
                                        LayerNode.factory, 
                                        nn.LazyBatchNorm3d,
                                        (
                                            {"label":"eps", "type":'float', "step":0.00001, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                        )),
            "LazyConv1d":   DragSource("LazyConv1d", 
                                        LayerNode.factory, 
                                        nn.LazyConv1d,
                                        (
                                            {"label":"out_channels", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"kernel_size", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":5},
                                            {"label":"stride", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":1},
                                            {"label":"padding", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "LazyConv2d":   DragSource("LazyConv2d", 
                                        LayerNode.factory, 
                                        nn.LazyConv2d,
                                        (
                                            {"label":"out_channels", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"kernel_size", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":5},
                                            {"label":"stride", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":1},
                                            {"label":"padding", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "LazyConv3d":   DragSource("LazyConv3d", 
                                        LayerNode.factory, 
                                        nn.LazyConv3d,
                                        (
                                            {"label":"out_channels", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"kernel_size", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":5},
                                            {"label":"stride", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":1},
                                            {"label":"padding", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "BatchNorm1d":   DragSource("BatchNorm1d", 
                                        LayerNode.factory, 
                                        nn.BatchNorm1d,
                                        (
                                            {"label":"num_features", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"eps", "type":'float', "step":0.00001, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                        )),
            "BatchNorm2d":   DragSource("BatchNorm2d", 
                                        LayerNode.factory, 
                                        nn.BatchNorm2d,
                                        (
                                            {"label":"num_features", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"eps", "type":'float', "step":0.00001, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                        )),
            "BatchNorm3d":   DragSource("BatchNorm3d", 
                                        LayerNode.factory, 
                                        nn.BatchNorm3d,
                                        (
                                            {"label":"num_features", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":6},
                                            {"label":"eps", "type":'float', "step":0.00001, "min_value":0.00001, "min_clamped":True, "default_value":0.00001},
                                            {"label":"momentum", "type":'float', "step":0.001, "min_value":0.000001, "min_clamped":True, "default_value":0.01},
                                            {"label":"affine", "type":'bool'},
                                        )),
        
            "Flatten":        DragSource("Flatten",
                                        LayerNode.factory,
                                        nn.Flatten),
            "AvgPool2d":      DragSource("MaxPool2d",
                                        LayerNode.factory,
                                        nn.AvgPool2d,
                                        (
                                            {"label":"kernel_size", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                            {"label":"stride", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "MaxPool2d":      DragSource("MaxPool2d",
                                        LayerNode.factory,
                                        nn.MaxPool2d,
                                        (
                                            {"label":"kernel_size", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                            {"label":"stride", "type":'int', "step":1, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "AdaptiveAvgPool2d":DragSource("AdaptiveAvgPool2d",
                                        LayerNode.factory,
                                        nn.AdaptiveAvgPool2d,
                                        (
                                            {"label":"output_size", "type":'text/tuple', "default_value":'1, 2'},
                                        )),
            "Dropout":      DragSource("Dropout",
                                        LayerNode.factory,
                                        nn.Dropout,
                                        (
                                            {"label":"p", "type":'float', "default_value":0.5},
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
                                    (layers['LazyConv2d'], {'out_channels':6,"kernel_size":5,"stride":1,"padding":3}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":5,"stride":1,"padding":1}),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':120}), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':84}), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':10}),
                                ),
                                node_editor=self.node_editor),
            'VGG': DragSource("VGG", 
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':32,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':64,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':64,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':120,  }), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':84,  }), (layers['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':10,  }),
                                ),
                                node_editor=self.node_editor),
            'AlexNet': DragSource("AlexNet", 
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':96,"kernel_size":11,"stride":4,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':256,"kernel_size":5,"stride":1,"padding":2, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':256,"kernel_size":3,"stride":1,"padding":1, }),(layers['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),(layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':4096,  }), (layers['ReLU'], ),(layers['Dropout'], {'p':0.5}),
                                    (layers['LazyLinear'], {'out_features':84,  }), (layers['ReLU'], ),(layers['Dropout'], {'p':0.5}),
                                    (layers['LazyLinear'], {'out_features':10,  }),
                                ),
                                node_editor=self.node_editor),
            'NiN': DragSource("NiN",
                                ModuleNode.factory,
                                (
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':11, 'stride':4,'padding':0, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0,  }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0,  }),(layers['ReLU'], )
                                ),
                                node_editor=self.node_editor)
        }
        archs['NiN Net'] = DragSource("NiN Net",
                                ModuleNode.factory,
                                (
                                    (datasets['FashionMNIST'], {"batch_size": 128, "Resize": '224, 224'}),
                                    (archs['NiN'], {'out_channels':[96, 96, 96], 'kernel_size':[11,1,1], 'stride':[4, 1, 1],'padding':[0, 0, 0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['NiN'], {'out_channels':[256, 256, 256], 'kernel_size':[5,1,1], 'stride':[1, 1, 1],'padding':[2,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['NiN'], {'out_channels':[384, 384, 384], 'kernel_size':[3,1,1], 'stride':[1, 1, 1],'padding':[1,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (layers['Dropout'], {'p':0.5}),
                                    (archs['NiN'], {'out_channels':[10, 10, 10], 'kernel_size':[3,1,1], 'stride':[1,1,1],'padding':[1,0,0]}),
                                    (layers['AdaptiveAvgPool2d'], {'output_size':'1, 1'}),(layers['Flatten'],)
                                ),
                                node_editor=self.node_editor)
        self.archs_container = DragSourceContainer("Модули", 150, 0)
        self.archs_container.add_drag_source(archs.values())
        
        

        
    def update(self):

        with dpg.mutex():
            dpg.delete_item(self.left_panel, children_only=True)
            self.dataset_container.submit(self.left_panel)
            self.layer_container.submit(self.left_panel)

            dpg.delete_item(self.right_panel, children_only=True)
            self.archs_container.submit(self.right_panel)

                

    def start(self):
        dpg.set_viewport_title("Deep Learning Constructor")
        dpg.show_viewport()
        try:
            with dpg.item_handler_registry(tag="hover_handler"):
                dpg.add_item_hover_handler(callback=lambda s, a, u: dpg.configure_item('hover_logger', 
                                        default_value=f"Текущий элемент: {dpg.get_item_label(a)}"))        
        except SystemError as err: print("Удаление узла")
            
        with dpg.window() as main_window:

            with dpg.menu_bar():
                with dpg.menu(label="Файл"):
                    dpg.add_menu_item(label="Сбросить", callback=self.node_editor.clear)

                with dpg.menu(label="Настройки"):
                    dpg.add_menu_item(label="Логирование", check=True, callback=lambda s,check_value,u:Configs.set_logger)
                    with dpg.menu(label="Инструменты"):
                        dpg.add_menu_item(label="Show Metrics", callback=lambda:dpg.show_tool(dpg.mvTool_Metrics))
                        dpg.add_menu_item(label="Show Documentation", callback=lambda:dpg.show_tool(dpg.mvTool_Doc))
                        dpg.add_menu_item(label="Show Debug", callback=lambda:dpg.show_tool(dpg.mvTool_Debug))
                        dpg.add_menu_item(label="Show Style Editor", callback=lambda:dpg.show_tool(dpg.mvTool_Style))
                        dpg.add_menu_item(label="Show Font Manager", callback=lambda:dpg.show_tool(dpg.mvTool_Font))
                        dpg.add_menu_item(label="Show Item Registry", callback=lambda:dpg.show_tool(dpg.mvTool_ItemRegistry))
                        dpg.add_menu_item(label="Show About", callback=lambda:dpg.show_tool(dpg.mvTool_About))
                
                with dpg.menu(tag='menu_message_logger', label='---Сообщения---'):
                    dpg.add_child_window(tag='message_logger', height=200, delay_search=True, width=1600)

            with dpg.group(tag='panel', horizontal=True):
                # left panel
                with dpg.group(tag=self.left_panel):
                    self.dataset_container.submit(self.left_panel)
                    self.layer_container.submit(self.left_panel)

                # center panel
                with dpg.group(tag=self.center_panel):
                    self.node_editor.submit(self.center_panel)
                    dpg.add_text(tag='hover_logger', default_value="Текущий элемент: ", 
                                 parent=self.center_panel)

                # right panel
                with dpg.group(tag=self.right_panel):
                    self.archs_container.submit(self.right_panel)
                    
        
        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()
        

app = App()
app.start()
