import dearpygui.dearpygui as dpg
import torch
from d2l import torch as d2l
from torch import nn
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

        dpg.add_node_link(self.uuid, child.uuid, parent=parent)
        child.set_parent(self)
        self._children.append(child)

    def execute(self, data):
        self._data = data
        for child in self._children:
            child._data = self._data

    def submit(self, parent):

        with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                user_data=self, id=self.uuid):
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

    def submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, id=self.uuid):
            dpg.add_text(self._label)

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
                    
                    

class Node:

    def __init__(self, label: str, data):

        self.label = label
        self.uuid = dpg.generate_uuid()
        self.static_uuid = dpg.generate_uuid()
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = []
        self._data = data

    def finish(self):
        dpg.bind_item_theme(self.uuid, _completion_theme)

    def add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)

    def add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)

    def add_params(self, params: list[ParamNode]):
        if params:
            self._params += params

    def custom(self):
        pass
        
    def execute(self):
        for attribute in self._output_attributes:
            attribute.execute(self._data)
        self.finish()


    def submit(self, parent):

        with dpg.node(parent=parent, label=self.label, tag=self.uuid):

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

    @staticmethod
    def _link_callback(sender, app_data, user_data):
        output_attr_uuid, input_attr_uuid = app_data

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        output_attr.add_child(sender, input_attr)

    def __init__(self):

        self._nodes: list[Node] = []
        self.uuid = dpg.generate_uuid()

    def add_node(self, node: Node):
        self._nodes.append(node)

    def on_drop(self, sender, app_data, user_data):
        source, generator, data, params = app_data
        node: Node = generator(source.label, data, params)
        node.submit(self.uuid)
        self.add_node(node)

    def submit(self, parent):
        
        with dpg.child_window(width=-160, parent=parent, user_data=self, 
                              drop_callback=lambda s, a, u: dpg.get_item_user_data(s).on_drop(s, a, u)):
            with dpg.node_editor(tag=self.uuid, callback=NodeEditor._link_callback, width=-1, height=-1):
                for node in self._nodes:
                    node.submit(self.uuid)


########################################################################################################################
# Drag & Drop
########################################################################################################################
class DragSource:

    def __init__(self, label: str, node_generator, data=None, params: list[ParamNode]=None):

        self.label = label
        self._generator = node_generator
        self._data = data
        self._params = params

    def submit(self, parent):

        dpg.add_button(label=self.label, parent=parent, width=-1)
        dpg.bind_item_theme(dpg.last_item(), _source_theme)

        with dpg.drag_payload(parent=dpg.last_item(), drag_data=(self, self._generator, self._data, self._params)):
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

class ParamSource:

    def __init__(self, label: str, choices:dict, **params):

        self.uuid = dpg.generate_uuid()
        self.label = label
        self.choices = choices
        self.params=params

    def submit(self, parent):
        dpg.add_combo(**self.params, label=self.label, parent=parent, 
                      items=list(self.choices.keys()), 
                      tag=self.uuid)
        dpg.bind_item_theme(dpg.last_item(), _source_theme)



class ParamContainer:

    def __init__(self, label: str, width: int = 150, height: int = -1):

        self._label = label
        self._width = width
        self._height = height
        self._uuid = dpg.generate_uuid()
        self._children: list[ParamSource] = []  # params

    def add_param(self, source: ParamSource):
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
    def factory(name, data, params=None):
        node = UtilityNode(name, data, params)
        return node

    def __init__(self, label: str, data, params=None):
        super().__init__(label, data)

        self.add_input_attribute(InputNodeAttribute("data", self))
        self.add_output_attribute(OutputNodeAttribute("processed data"))
        self.add_params(params)




########################################################################################################################
# Layers
########################################################################################################################
class LayerNode(Node):

    @staticmethod
    def factory(name, data, params=None):
        node = LayerNode(name, data, params)
        return node

    def __init__(self, label: str, data, params=None):
        super().__init__(label, data)

        self.add_input_attribute(InputNodeAttribute("data", self))
        self.add_output_attribute(OutputNodeAttribute("weighted data"))
        self.add_params(params)

        

########################################################################################################################
# Tools
########################################################################################################################
class ViewNode_2D(Node):

    @staticmethod
    def factory(name, data=None, params=None):
        node = ViewNode_2D(name, data, params)
        return node

    def __init__(self, label: str, data=None, params=None):
        super().__init__(label, data)

        self.add_input_attribute(InputNodeAttribute("full dataset", self))
        self.add_params(params)

        self.x_axis = dpg.generate_uuid()
        self.y_axis = dpg.generate_uuid()
        

    def custom(self):

        with dpg.plot(height=400, width=400, no_title=True):
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
    def factory(name, data, params=None):
        node = DataNode(name, data, params)
        return node

    def __init__(self, label: str, data, params=None):
        super().__init__(label, data)
        self.add_output_attribute(OutputNodeAttribute("data"))
        self.add_output_attribute(OutputNodeAttribute("train graph"))
        self.add_params(params)
        self.uuid = dpg.generate_uuid()


class RegressNet(d2l.Module):
    def __init__(self, lr, sequential: list, Loss, widget=None):
        super().__init__(widget=widget)
        self.save_hyperparameters()
        self.net = nn.Sequential(*sequential)
        self.loss_func=Loss

    def loss(self, y_hat, y):
        fn= self.loss_func()
        return fn(y_hat, y)
    
    
class ClassifierNet(d2l.Classifier):
    def __init__(self, lr, sequential: list, Loss, widget=None):
        super().__init__(widget=widget)
        self.save_hyperparameters()
        self.net = nn.Sequential(*sequential)
        self.loss_func=Loss

    def loss(self, y_hat, y):
        fn= self.loss_func()
        return fn(y_hat, y)
    
########################################################################################################################
# Application
########################################################################################################################
class App:

    def __init__(self):

        self.dataset_container = DragSourceContainer("Datasets", 150, -500)
        self.layer_container = DragSourceContainer("Layers", 150, -1)
        self.utility_container = DragSourceContainer("Utilities", 150, -500)
        self.tool_container = DragSourceContainer("Tools", 150, -150)
        self.param_container = ParamContainer("Parameters", 150, -30)
        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.node_editor = NodeEditor()
        self.right_panel = dpg.generate_uuid()

        self.plugins = []
        self.dataset_container.add_drag_source(DragSource("Synthetic Regression Data", 
                                                          DataNode.factory, 
                                                          d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)))
        self.dataset_container.add_drag_source(DragSource("FashionMNIST", 
                                                          DataNode.factory, 
                                                          d2l.FashionMNIST,
                                                          [ParamNode("batch_size", 'int', step=2, width=150, min_value=2, min_clamped=True, default_value=64)]))
        
        self.utility_container.add_drag_source(DragSource("Flatten",
                                                          UtilityNode.factory,
                                                          nn.Flatten))
                                               
        self.layer_container.add_drag_source(DragSource("Linear Layer", 
                                                        LayerNode.factory, 
                                                        nn.LazyLinear,
                                                        [ParamNode("out_features", 'int', step=1, width=150, min_value=1, min_clamped=True, default_value=1)]))
        
        self.tool_container.add_drag_source(DragSource("Progress Board", 
                                                        ViewNode_2D.factory))
        
        self.param_container.add_param(ParamSource("Loss",
                                                   {'MSE': nn.MSELoss, 'Cross Entropy Loss':nn.CrossEntropyLoss},
                                                   default_value='MSE'))
        self.param_container.add_param(ParamSource("Task",
                                                   {'Regression': RegressNet, 'Classifier': ClassifierNet},
                                                   default_value='Regression'))
        

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

    def train(self, data, pipline: list[Node], **train_params):
        sequential = []
        for layer in pipline:
            params = dict()
            for param in layer._params:
                params[param._label] = dpg.get_value(param.uuid)
            sequential.append(layer._data(**params))  
        
        net = train_params.pop('Task')
        net = net(0.03, sequential, **train_params)
        print("pipline: ", net)
        trainer = d2l.Trainer(max_epochs=10)
        trainer.fit(net, data)


    def piplines(self):
        print("\tTrainig!")
        train_params = dict()
        for param in self.param_container._children:
            train_params[param.label] = param.choices[dpg.get_value(param.uuid)]
        print("\n\n\ttrain params: ", train_params)


        for node in self.node_editor._nodes:
            if isinstance(node, DataNode):
                data_params = dict()
                if len(node._params):
                    for data_param in node._params:
                        data_params[data_param._label] = dpg.get_value(data_param.uuid)
                    data = node._data(**data_params)
                else:
                    data = node._data 
                for model_init in node._output_attributes[0]._children:
                    model_init = model_init._data
                    pipline: list[Node] = [data, model_init]
                    while len(linked_node := model_init._output_attributes[0]._children):
                        pipline.append(linked_node[0]._data)
                        model_init = linked_node[0]._data
                    progress_board = node._output_attributes[1]._children
                    print('\n\n\t\tpipline: ', pipline)
                    if len(progress_board):
                        self.train(pipline[0], pipline[1:], widget=progress_board[0]._data, **train_params)
                    else: 
                        self.train(pipline[0], pipline[1:], **train_params)
                

    def start(self):
        #dpg.setup_registries()
        dpg.set_viewport_title("Deep Learning Constructor")
        dpg.show_viewport()

        with dpg.window() as main_window:

            with dpg.menu_bar():

                with dpg.menu(label="Operations"):
                    dpg.add_menu_item(label="Reset", callback=lambda: dpg.delete_item(self.node_editor.uuid, children_only=True))

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
                    self.param_container.submit(self.right_panel)
                    dpg.add_button(label='Train', callback=self.piplines)
                    

        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()


if __name__ == "__main__":

    app = App()
    app.start()
