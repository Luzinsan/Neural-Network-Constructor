import dearpygui.dearpygui as dpg

import sys, os
sys.path.insert(1, os.getcwd())

from config.setup import *
from config.settings import Configs

from core import node_editor
from core.dragndrop import DragSource, DragSourceContainer


    
class App:

    def __init__(self):

        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.center_panel = dpg.generate_uuid()
        self.right_panel = dpg.generate_uuid()
        self.node_editor = node_editor.NodeEditor()
    
        
        #region datasets
        datasets = {
            data: DragSource(data)
            for data
            in ["FashionMNIST","Caltech101","Caltech256","CarlaStereo","CelebA",
                "CIFAR10","Cityscapes","CLEVRClassification","EMNIST","CocoCaptions",
                "EuroSAT","Flowers102","Food101","ImageNet","SUN397","Dataset from File",]
        }
        self.dataset_container = DragSourceContainer("Датасеты", 150, -500)
        self.dataset_container.add_drag_source(datasets.values())
        #endregion
        
        #region layers
        layers = {
            layer: DragSource(layer)
            for layer
            in ["LazyLinear","LazyBatchNorm1d","LazyBatchNorm2d","LazyBatchNorm3d",
                "LazyConv1d","LazyConv2d","LazyConv3d",
                "BatchNorm1d","BatchNorm2d","BatchNorm3d",
                "Flatten","AvgPool2d","MaxPool2d","AdaptiveAvgPool2d",
                "Dropout","ReLU","Softmax","Tanh","GELU",]
         }
        self.layer_container = DragSourceContainer("Слои|ф.активации", 150, 0)
        self.layer_container.add_drag_source(layers.values())
        #endregion
        
        #region architectures
        archs = {
            'LeNet': DragSource("LeNet",
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
                                 ),
            'VGG': DragSource("VGG",
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
                                 ),
            'AlexNet': DragSource("AlexNet",
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
                                 ),
            'NiN': DragSource("NiN",
                                (
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':11, 'stride':4,'padding':0, }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0,  }),(layers['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0,  }),(layers['ReLU'], )
                                ),
                                 )
        }
        archs['NiN Net'] = DragSource("NiN Net",
                                (
                                    (archs['NiN'], {'out_channels':[96, 96, 96], 'kernel_size':[11,1,1], 'stride':[4, 1, 1],'padding':[0, 0, 0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['NiN'], {'out_channels':[256, 256, 256], 'kernel_size':[5,1,1], 'stride':[1, 1, 1],'padding':[2,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['NiN'], {'out_channels':[384, 384, 384], 'kernel_size':[3,1,1], 'stride':[1, 1, 1],'padding':[1,0,0]}),(layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (layers['Dropout'], {'p':0.5}),
                                    (archs['NiN'], {'out_channels':[10, 10, 10], 'kernel_size':[3,1,1], 'stride':[1,1,1],'padding':[1,0,0]}),
                                    (layers['AdaptiveAvgPool2d'], {'output_size':'[1, 1]'}),(layers['Flatten'],)
                                ))
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
        
            
        with dpg.window() as main_window:

            with dpg.menu_bar():
                with dpg.menu(label="Файл"):
                    dpg.add_menu_item(label="Открыть", callback=lambda:self.node_editor.callback_file(self.node_editor.open))
                    dpg.add_menu_item(label="Сохранить", callback=lambda:self.node_editor.callback_file(self.node_editor.save))
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
