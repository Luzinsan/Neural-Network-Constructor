from typing import Union
import dearpygui.dearpygui as dpg
import torch
from d2l import torch as d2l
from torch import nn
import torchvision.datasets as ds
import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
# Setup
########################################################################################################################
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

########################################################################################################################
# Themes
########################################################################################################################

with dpg.theme() as _source_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, [25, 119, 0])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [25, 255, 0])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [25, 119, 0])

with dpg.theme() as _completion_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [37, 28, 138], category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, [37, 28, 138], category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, [37, 28, 138], category=dpg.mvThemeCat_Nodes)




########################################################################################################################
# Node DPG Wrappings
########################################################################################################################
class OutputNodeAttribute:

    def __init__(self, label: str = "output"):

        self._label = label
        self.uuid = dpg.generate_uuid()
        self._children = []  # output attributes
        self._data = None

    def add_child(self, parent, child):
        child.set_parent(self)
        self._children.append(child)

    def remove_child(self, parent, child):
        self._children.remove(child)
        child.reset_parent(self)

    def execute(self, data):
        self._data = data
        for child in self._children:
            child._data = self._data

    def submit(self, parent):

        with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                user_data=self, tag=self.uuid):
            dpg.add_text(self._label)


class InputNodeAttribute:

    def __init__(self, label: str = "input", data=None):

        self._label = label
        self.uuid = dpg.generate_uuid()
        self._parent = None  # output attribute
        self._data = data

    def get_data(self):
        return self._data

    def set_parent(self, parent: OutputNodeAttribute):
        self._parent = parent

    def reset_parent(self, parent: OutputNodeAttribute):
        self._parent = None

    def submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, id=self.uuid):
            dpg.add_text(self._label)

class LinkNode:

    def __init__(self, input_uuid=None, output_uuid=None, parent=None):

        self.uuid = dpg.generate_uuid()
        self._input_attr = input_uuid
        self._output_attr = output_uuid
        self._parent = parent

    def get_attrs(self):
        return self._input_attr, self._output_attr

    @staticmethod
    def _link_callback(node_editor_uuid, app_data, user_data=None):
        output_attr_uuid, input_attr_uuid = app_data

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        link_node = LinkNode(input_attr_uuid, output_attr_uuid, node_editor_uuid)
        dpg.add_node_link(*link_node.get_attrs(), parent=node_editor_uuid, user_data=link_node, tag=link_node.uuid)
        output_attr.add_child(node_editor_uuid, input_attr)


    @staticmethod
    def _delink_callback(node_editor_id, link_id):
        link: LinkNode = dpg.get_item_user_data(link_id)
        input_attr_uuid, output_attr_uuid = link.get_attrs()

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        output_attr.remove_child(node_editor_id, input_attr)
        dpg.delete_item(link.uuid)
        del link


def select_path(sender, app_data, user_data):
    dpg.set_item_user_data('file_dialog', user_data)
    dpg.show_item('file_dialog')


def set_path(sender, app_data):
    tag_path = dpg.get_item_user_data('file_dialog')
    dpg.configure_item(tag_path, default_value=app_data['file_path_name'])


with dpg.file_dialog(directory_selector=False, show=False, callback=set_path, tag="file_dialog",
                     width=700, height=400, modal=True):
    # dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255), custom_text="[Calc]")
    dpg.add_file_extension(".params", color=(0, 255, 0, 255), custom_text="[Params]")


class ParamNode:

    def __init__(self, label: str, type: str, **params):

        self.uuid = dpg.generate_uuid()
        self._label: str = label
        self._type: str = type
        self._params = params


    def submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Static):
            match self._type:
                case 'int':
                    dpg.add_input_int(**self._params, label=self._label, tag=self.uuid)
                case 'float':
                    dpg.add_input_float(**self._params, label=self._label, tag=self.uuid)
                case 'combo':
                    dpg.add_combo(**self._params, label=self._label, tag=self.uuid)
                case 'button':
                    dpg.add_button(**self._params, label=self._label, tag=self.uuid)
                case 'file':
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(width=150, no_spaces=True, tag=self.uuid)
                        dpg.add_button(label="Path", user_data=self.uuid, callback=select_path)
                    dpg.add_button(**self._params, label=self._label, user_data=(dpg.get_item_user_data(parent), self.uuid))
                    
                    

class Node:

    def __init__(self, label: str, data=None, **node_params):

        self.label = label
        self.uuid = dpg.generate_uuid()
        self.static_uuid = dpg.generate_uuid()
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = []
        self._data = data
        self.node_params = node_params #if len(node_params) else None 

    def finish(self):
        dpg.bind_item_theme(self.uuid, _completion_theme)

    def add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)

    def add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)

    def add_params(self, params: list[dict]):
        if params:
            self._params += [ParamNode(**param) for param in params] 

    def custom(self):
        pass
        
    def execute(self):
        for attribute in self._output_attributes:
            attribute.execute(self._data)
        self.finish()


    def submit(self, parent):

        with dpg.node(**self.node_params, parent=parent, label=self.label, tag=self.uuid, user_data=self):

            for attribute in self._input_attributes:
                attribute.submit(self.uuid)

            for attribute in self._params:
                attribute.submit(self.uuid)
            
            with dpg.node_attribute(parent=self.uuid, attribute_type=dpg.mvNode_Attr_Static,
                                    user_data=self, tag=self.static_uuid):
                self.custom()

            for attribute in self._output_attributes:
                attribute.submit(self.uuid)


class NodeEditor:

    
    def __init__(self):

        self._nodes: list[Node] = []
        self.uuid = dpg.generate_uuid()

    def add_node(self, node: Node):
        self._nodes.append(node)

    def on_drop(self, sender, app_data, user_data):
        source, generator, data, params, default_params = app_data
        node: Node = generator(source.label, data, params, default_params)
        node.submit(self.uuid)
        self.add_node(node)

    def clear(self):
        dpg.delete_item(self.uuid, children_only=True)
        self._nodes.clear()


    def submit(self, parent):
        
        with dpg.child_window(width=-160, parent=parent, user_data=self, 
                              drop_callback=lambda s, a, u: dpg.get_item_user_data(s).on_drop(s, a, u)):
            with dpg.node_editor(callback=LinkNode._link_callback,
                                 delink_callback=LinkNode._delink_callback,
                                 tag=self.uuid, width=-1, height=-1):
                for node in self._nodes:
                    node.submit(self.uuid)


########################################################################################################################
# Drag & Drop
########################################################################################################################
class DragSource:

    def __init__(self, label: str, node_generator, data=None, params: list[dict]=None, default_params: dict[str, str]= None):

        self.label = label
        self._generator = node_generator
        self._data = data
        self._params = params
        self._default_params = default_params

    def submit(self, parent):

        dpg.add_button(label=self.label, parent=parent, width=-1)
        dpg.bind_item_theme(dpg.last_item(), _source_theme)

        with dpg.drag_payload(parent=dpg.last_item(), drag_data=(self, self._generator, self._data, self._params, self._default_params)):
            dpg.add_text(f"Name: {self.label}")



class DragSourceContainer:

    def __init__(self, label: str, width: int = 150, height: int = -1):

        self._label = label
        self._width = width
        self._height = height
        self._uuid = dpg.generate_uuid()
        self._children: list[DragSource] = []  # drag sources

    def add_drag_source(self, source: DragSource):
        self._children.append(source)

    def submit(self, parent):

        with dpg.child_window(parent=parent, width=self._width, height=self._height, tag=self._uuid, menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label, enabled=False)

            for child in self._children:
                child.submit(child_parent)



########################################################################################################################
# Utilities
########################################################################################################################
class UtilityNode(Node):

    @staticmethod
    def factory(name, data, params: list[dict]=None, default_params: dict[str,str]=None,**node_params):
        node = UtilityNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params: list[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)

        self.add_input_attribute(InputNodeAttribute("data", self))
        self.add_output_attribute(OutputNodeAttribute("processed data"))
        self.add_params(params)


class TrainParamsNode(Node):

    @staticmethod
    def factory(name, data=None, train_params: list[dict]=None, **node_params):
        node = TrainParamsNode(name, data, train_params, **node_params)
        return node

    def __init__(self, label: str, data=None, train_params: list[dict]=None, **node_params):
        super().__init__(label, data, **node_params)

        self.pipline: Pipline = None
        self.add_input_attribute(InputNodeAttribute("train dataset", self))
        self.add_params(train_params)


    def set_pipline(self, pipline):
        self.pipline = pipline




########################################################################################################################
# Layers
########################################################################################################################
class LayerNode(Node):

    @staticmethod
    def factory(name, data, params:list[dict]=None, default_params: dict[str,str]=None, **node_params):
        node = LayerNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:list[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)

        self.add_input_attribute(InputNodeAttribute("data", self))
        self.add_output_attribute(OutputNodeAttribute("weighted data"))
        self.add_params(params)

        

########################################################################################################################
# Tools
########################################################################################################################
class ViewNode_2D(Node):

    @staticmethod
    def factory(name, data=None, params:list[dict]=None, default_params: dict[str,str]=None, **node_params):
        node = ViewNode_2D(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data=None, params:list[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)

        self.add_input_attribute(InputNodeAttribute("full dataset", self))
        self.add_params(params)

        self.x_axis = dpg.generate_uuid()
        self.y_axis = dpg.generate_uuid()
        self.plot = dpg.generate_uuid()
        

    def custom(self):

        with dpg.plot(height=400, width=400, no_title=True, tag=self.plot):
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag=self.x_axis)
            # dpg.set_axis_limits(dpg.last_item(), 0, 10)
            dpg.add_plot_axis(dpg.mvYAxis, label="estimates", tag=self.y_axis)
            dpg.set_axis_limits(dpg.last_item(), -0.1, 1)
            dpg.add_plot_legend()


    def execute(self, plt_lines=None, labels=None):

        x_axis_id = self.x_axis
        y_axis_id = self.y_axis
        dpg.delete_item(y_axis_id, children_only=True)
        for idx, line in enumerate(plt_lines):
            x_orig_data, y_orig_data = line.get_xdata(), line.get_ydata()
            dpg.add_line_series(x_orig_data, y_orig_data, parent=y_axis_id, label=labels[idx])
        dpg.fit_axis_data(x_axis_id)
        dpg.fit_axis_data(y_axis_id)
        self.finish()


class DataNode(Node):

    @staticmethod
    def factory(name, data, params:list[dict]=None, default_params: dict[str, str]=None,**node_params):
        node = DataNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:list[dict]=None, default_params: dict[str, str]=None, **node_params):
        super().__init__(label, data, **node_params)
        self.add_output_attribute(OutputNodeAttribute("data"))
        self.add_output_attribute(OutputNodeAttribute("train graph"))
        self.add_output_attribute(OutputNodeAttribute("train params"))
        self.add_params(params)
        self.train_params: TrainParamsNode = None
        self.uuid = dpg.generate_uuid()
        self._default_params = default_params


    def submit(self, parent):
        super().submit(parent)
        losses = {"MSE (squared L2)": nn.MSELoss, "Cross Entropy Loss": nn.CrossEntropyLoss, "L1 Loss": nn.L1Loss}
        tasks = {
                #  "Regression": d2l.Module, 
                 "Classification":d2l.Classifier}
        
        


        self.train_params = TrainParamsNode('Train Params',
                                       train_params=[
                                            {"label":"Loss", "type":'combo', "default_value":self._default_params['Loss'], "width":150,
                                             "items":tuple(losses.keys()), "user_data":losses},
                                            {"label":"Task", "type":'combo', "default_value":self._default_params['Task'], "width":150,
                                             "items":tuple(tasks.keys()), "user_data":tasks},
                                            {"label":"Learning Rate", "type":'float', "default_value":0.05, "width":150},
                                            {"label":"Max Epoches", "type":'int', "default_value":2, "width":150},
                                            {"label":"Save Weights", "type":"file", "callback":Pipline.save_weight},
                                            {"label":"Load Weights", "type":"file", "callback":Pipline.load_weight},
                                            {"label":"Train", "type":"button", "callback":Pipline.flow, "user_data":self},
                                            {"label":"Continue Train", "type":"button", "callback":Pipline.keep_train, "user_data":self}
                                           ],
                                       pos=(100, 500))
        
        self.train_params.submit(parent)
        LinkNode._link_callback(parent, (self._output_attributes[2].uuid, self.train_params._input_attributes[0].uuid))




class Pipline:

    @staticmethod
    def flow(sender=None, app_data=None, data_node=None, fake=False):
        assert data_node
        try:
            for model_init in data_node._output_attributes[0]._children:
                self = Pipline(data_node)
                self.collect_layers(model_init._data)
                self.train(fake)
        except BaseException as err:
            raise LookupError("Error in flow")
        return self
    
    @staticmethod
    def keep_train(sender=None, app_data=None, data_node:DataNode=None):
        self = data_node.train_params.pipline
        if self:
            try:
                self.max_epoches = Pipline.get_params(data_node.train_params)['Max Epoches']
                self.train()
            except BaseException as err:
                raise RuntimeError("Error in keep training")
        else:
            Pipline.flow(data_node=data_node)
        
        

    def __init__(self, init_node: DataNode):
        self.pipline = [Pipline.init_layer(init_node)]
        init_node.train_params.set_pipline(self)
        self.train_params = Pipline.get_params(init_node.train_params)
        print("\n\n\ttrain params: ", self.train_params)

        self.progress_board: list = init_node._output_attributes[1]._children
        self.progress_board: Union[ViewNode_2D, None] = self.progress_board[0]._data if len(self.progress_board) else None
        
    @staticmethod
    def init_normal(module: nn.Module):
        if type(module) == nn.Linear:
            nn.init.normal_(module.weight, mean=0, std=0.01)
            nn.init.zeros_(module.bias)

    @staticmethod
    def init_xavier(module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight)
    

    @staticmethod
    def init_layer(layer: Node) -> any:
        params = dict()
        if len(layer._params):
            init_dict = dict()
            for param in layer._params:
                key = param._label
                if key == 'Initialization':
                    init_dict = dpg.get_item_user_data(param.uuid)
                params[key] = dpg.get_value(param.uuid)
            
            if 'Initialization' in params.keys() and (init_func := init_dict[params.pop("Initialization")]):
                return layer._data(**params).apply(init_func)
            return layer._data(**params)
        return layer._data()
        

    @staticmethod
    def get_params(params_node: TrainParamsNode) -> dict:
        train_params = dict()
        for param in params_node._params[:-4]:
            choices = dpg.get_item_user_data(param.uuid)
            if isinstance(choices, dict):
                train_params[param._label] = choices[dpg.get_value(param.uuid)]
            else:
                train_params[param._label] = dpg.get_value(param.uuid)
        return train_params
    
    
    @staticmethod
    def save_weight(sender, app_data, train_params__file: tuple[2]):
        train_params, filepath_uuid = train_params__file
        self = train_params.pipline
        if self:
            filepath = dpg.get_value(filepath_uuid)
            torch.save(self.net.state_dict(), filepath)

    @staticmethod
    def load_weight(sender, app_data, train_params__file: tuple[2]):
        train_params, filepath_uuid = train_params__file
        self: Pipline = train_params.pipline
        if not self:
            data_node = dpg.get_item_user_data(
                            dpg.get_item_parent(train_params._input_attributes[0]._parent.uuid))  
            self = Pipline.flow(data_node=data_node, fake=True)
            
        filepath = dpg.get_value(filepath_uuid)
        try:
            self.net.load_state_dict(torch.load(filepath))
        except BaseException as err:
            raise FileNotFoundError("Файл параметров не найден")
        
    

    def collect_layers(self, node: Node):
        self.pipline.append(Pipline.init_layer(node))
        
        while len(node := node._output_attributes[0]._children):
            node = node[0]._data
            self.pipline.append(Pipline.init_layer(node))

        net = self.train_params.pop('Task')
        self.max_epoches = self.train_params.pop('Max Epoches')
        self.net = net(self.train_params.pop('Learning Rate'), self.train_params.pop('Loss'), 
                       self.pipline[1:], widget=self.progress_board)
        print("pipline: ", self.net)
        

    def train(self, fake=False):
        axes = dpg.get_item_children(self.progress_board.plot, 1)
        for axis in axes:
            dpg.delete_item(axis, children_only=True, slot=1)
        self.net.board = d2l.ProgressBoard(widget=self.progress_board)

        self.trainer = d2l.Trainer(max_epochs=1 if fake else self.max_epoches)
        self.trainer.fit(self.net, self.pipline[0])


    
    
    
    
########################################################################################################################
# Application
########################################################################################################################
class App:

    def __init__(self):

        self.dataset_container = DragSourceContainer("Datasets", 150, -500)
        self.layer_container = DragSourceContainer("Layers", 150, -1)
        self.utility_container = DragSourceContainer("Utilities", 150, -500)
        self.tool_container = DragSourceContainer("Tools", 150, -30)

        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.right_panel = dpg.generate_uuid()

        self.node_editor = NodeEditor()
        

        self.plugins = []
        
        self.dataset_container.add_drag_source(DragSource("FashionMNIST", 
                                                          DataNode.factory, 
                                                          d2l.FashionMNIST,
                                                          [{"label":"batch_size", "type":'int', "step":2, "width":150, "min_value":2, "min_clamped":True, "default_value":64}],
                                                          default_params={'Loss':'Cross Entropy Loss','Task':'Classification'}))
        
        self.utility_container.add_drag_source(DragSource("Flatten",
                                                          UtilityNode.factory,
                                                          nn.Flatten))
        self.utility_container.add_drag_source(DragSource("ReLU",
                                                          UtilityNode.factory,
                                                          nn.ReLU))
        self.utility_container.add_drag_source(DragSource("Softmax",
                                                          UtilityNode.factory,
                                                          nn.Softmax))
        self.utility_container.add_drag_source(DragSource("Tanh",
                                                          UtilityNode.factory,
                                                          nn.Tanh))
        self.utility_container.add_drag_source(DragSource("GELU",
                                                          UtilityNode.factory,
                                                          nn.GELU))

        init_params = {'Default':None, 'Normal': Pipline.init_normal, 'Xavier': Pipline.init_xavier } 
        self.layer_container.add_drag_source(DragSource("Linear Layer", 
                                                        LayerNode.factory, 
                                                        nn.LazyLinear,
                                                        [{"label":"out_features", "type":'int', "step":1, "width":150, "min_value":1, "min_clamped":True, "default_value":1},
                                                         {"label":"Initialization", "type":'combo', "default_value":'Default', "width":150,
                                                        "items":tuple(init_params.keys()), "user_data":init_params}]))
        # self.layer_container.add_drag_source(DragSource("LazyConv1d", 
        #                                                 LayerNode.factory, 
        #                                                 nn.LazyConv1d,
        #                                                 [ParamNode("out_channels", 'int', step=1, width=150, min_value=1, min_clamped=True, default_value=1)]
        #                                                 ))
        # self.layer_container.add_drag_source(DragSource("LazyConv2d", 
        #                                                 LayerNode.factory, 
        #                                                 nn.LazyConv2d,
        #                                                 [ParamNode("out_channels", 'int', step=1, width=150, min_value=1, min_clamped=True, default_value=1)]
        #                                                 ))
        # self.layer_container.add_drag_source(DragSource("LazyConv3d", 
        #                                                 LayerNode.factory, 
        #                                                 nn.LazyConv3d,
        #                                                 [ParamNode("out_channels", 'int', step=1, width=150, min_value=1, min_clamped=True, default_value=1)]
        #                                                 ))
        
        self.tool_container.add_drag_source(DragSource("Progress Board", 
                                                        ViewNode_2D.factory))
        

        
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
                    # with dpg.group(horizontal=True):
                    #     dpg.add_button(label='Train', callback=self.piplines)
                    #     dpg.add_button(label='Continue Train', callback=self.)
                    

        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()


if __name__ == "__main__":

    app = App()
    app.start()
