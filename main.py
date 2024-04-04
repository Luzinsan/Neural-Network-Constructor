
import dearpygui.dearpygui as dpg
import config.settings
from torch import nn
import torchvision.datasets as ds
from  core.node_editor import NodeEditor
from core.dragndrop import DragSource, DragSourceContainer
from nodes.dataset import DataNode
from nodes.utility import UtilityNode
from nodes.layer import LayerNode, ModuleNode
from pipline import Pipline
from nodes.tools import ViewNode_2D
import torchvision.datasets as ds




    
class App:

    def __init__(self):

        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.right_panel = dpg.generate_uuid()
        self.node_editor = NodeEditor()
        self.plugins = []
        WIDTH = 70

        #region datasets
        self.dataset_container = DragSourceContainer("Datasets", 150, -500)
        datasets = {
            "FashionMNIST": DragSource("FashionMNIST", 
                                        DataNode.factory, 
                                        ds.FashionMNIST,
                                        (
                                            {"label":"batch_size", "type":'int', "step":2, "width":WIDTH, "min_value":2, "min_clamped":True, "default_value":64},
                                        ),
                                        default_params={'Loss':'Cross Entropy Loss','Optimizer':'SGD'}),
            }
        self.dataset_container.add_drag_source(datasets.values())
        #endregion
        #region utilities
        utilities = {
            "Flatten":      DragSource("Flatten",
                                        UtilityNode.factory,
                                        nn.Flatten),
            "AvgPool2d":      DragSource("MaxPool2d",
                                        UtilityNode.factory,
                                        nn.AvgPool2d,
                                        (
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "MaxPool2d":      DragSource("MaxPool2d",
                                        UtilityNode.factory,
                                        nn.MaxPool2d,
                                        (
                                            {"label":"kernel_size", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                            {"label":"stride", "type":'int', "step":1, "width":WIDTH, "min_value":1, "min_clamped":True, "default_value":2},
                                        )),
            "AdaptiveAvgPool2d":DragSource("AdaptiveAvgPool2d",
                                        UtilityNode.factory,
                                        nn.AdaptiveAvgPool2d,
                                        (
                                            {"label":"output_size", "type":'text/tuple', "width":WIDTH, "default_value":'(1, 2)'},
                                        )),
            "Dropout":      DragSource("Dropout",
                                        UtilityNode.factory,
                                        nn.Dropout,
                                        (
                                            {"label":"p", "type":'float', "width":WIDTH, "default_value":0.5},
                                        )),
            "ReLU":         DragSource("ReLU",
                                        UtilityNode.factory,
                                        nn.ReLU,
                                        ),
            "Softmax":      DragSource("Softmax",
                                        UtilityNode.factory,
                                        nn.Softmax),
            "Tanh":         DragSource("Tanh",
                                        UtilityNode.factory,
                                        nn.Tanh),
            "GELU":         DragSource("GELU",
                                        UtilityNode.factory,
                                        nn.GELU),
        }
        self.utility_container = DragSourceContainer("Utilities", 150, -500)
        self.utility_container.add_drag_source(utilities.values())
        #endregion
        #region layers
        init_params = {'Default':None, 'Normal': Pipline.init_normal, 'Xavier': Pipline.init_xavier } 
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
            
            }
        

        layers['LeNet'] = DragSource("LeNet", 
                                    ModuleNode.factory,
                                    (
                                        
                                        (layers['LazyConv2d'], {'out_channels':6,"kernel_size":5,"stride":1,"padding":3,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (utilities['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                        (layers['LazyConv2d'], {'out_channels':16,"kernel_size":5,"stride":1,"padding":1,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (utilities['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                        (utilities['Flatten'], ),
                                        (layers['LazyLinear'], {'out_features':120, "Initialization":"Xavier"}), (utilities['ReLU'], ),
                                        (layers['LazyLinear'], {'out_features':84, "Initialization":"Xavier"}), (utilities['ReLU'], ),
                                        (layers['LazyLinear'], {'out_features':10, "Initialization":"Xavier"}),
                                    ),
                                    node_params={"node_editor":self.node_editor})
        layers['AlexNet'] = DragSource("AlexNet", 
                                    ModuleNode.factory,
                                    (
                                        (layers['LazyConv2d'], {'out_channels':96,"kernel_size":11,"stride":4,"padding":1,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (utilities['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                        (layers['LazyConv2d'], {'out_channels':256,"kernel_size":5,"stride":1,"padding":2,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (utilities['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                        (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':256,"kernel_size":3,"stride":1,"padding":1,"Initialization":"Xavier"}),(utilities['ReLU'], ),
                                        (utilities['MaxPool2d'], {"kernel_size":3,"stride":2}),(utilities['Flatten'], ),
                                        (layers['LazyLinear'], {'out_features':4096, "Initialization":"Xavier"}), (utilities['ReLU'], ),(utilities['Dropout'], {'p':0.5}),
                                        (layers['LazyLinear'], {'out_features':84, "Initialization":"Xavier"}), (utilities['ReLU'], ),(utilities['Dropout'], {'p':0.5}),
                                        (layers['LazyLinear'], {'out_features':10, "Initialization":"Xavier"}),
                                    ),
                                    node_params={"node_editor":self.node_editor})
        layers['NiN'] = DragSource("NiN",
                                    ModuleNode.factory,
                                    (
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':11, 'stride':4,'padding':0}),(utilities['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0}),(utilities['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0}),(utilities['ReLU'], )
                                    ),
                                    node_params={"node_editor":self.node_editor})
        layers['NiN Net'] = DragSource("NiN Net",
                                   ModuleNode.factory,
                                   (
                                       (datasets['FashionMNIST'], {"batch_size": 128}),
                                       (layers['NiN'], {'out_channels':[96]*3, 'kernel_size':[11,1,1], 'stride':[4,1,1],'padding':[0]*3}),(utilities['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                       (layers['NiN'], {'out_channels':[256]*3, 'kernel_size':[5,1,1], 'stride':[1]*3,'padding':[2,0,0]}),(utilities['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                       (layers['NiN'], {'out_channels':[384]*3, 'kernel_size':[3,1,1], 'stride':[1]*3,'padding':[1,0,0]}),(utilities['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                       (utilities['Dropout'], {'p':0.5}),
                                       (layers['NiN'], {'out_channels':[10]*3, 'kernel_size':[3,1,1], 'stride':[1]*3,'padding':[1,0,0]}),(utilities['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                       (utilities['AdaptiveAvgPool2d'], {'output_size':'(1,1)'}),(utilities['Flatten'],)
                                   ),
                                   node_params={"node_editor":self.node_editor})
        self.layer_container = DragSourceContainer("Layers", 150, -1)
        self.layer_container.add_drag_source(layers.values())
        #endregion
        #region tools
        tools = {
            "Progress Board":DragSource("Progress Board", 
                                        ViewNode_2D.factory),
        }
        self.tool_container = DragSourceContainer("Tools", 150, -30)
        self.tool_container.add_drag_source(tools.values())
        #endregion
        
        

        
    def update(self):

        with dpg.mutex():
            dpg.delete_item(self.left_panel, children_only=True)
            self.dataset_container.submit(self.left_panel)
            self.layer_container.submit(self.left_panel)

            dpg.delete_item(self.right_panel, children_only=True)
            self.utility_container.submit(self.right_panel)
            self.tool_container.submit(self.right_panel)


    def add_plugin(self, name, callback):
        self.plugins.append((name, callback))
                

   

    def start(self):
        #dpg.setup_registries()
        dpg.set_viewport_title("Deep Learning Constructor")
        dpg.show_viewport()

        with dpg.window() as main_window:

            with dpg.menu_bar():
                with dpg.menu(label="Operations"):
                    dpg.add_menu_item(label="Reset", callback=self.node_editor.clear)

                with dpg.menu(label="Plugins"):
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
                    self.utility_container.submit(self.right_panel)
                    self.tool_container.submit(self.right_panel)
                    
        
        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()
        




app = App()
app.start()
