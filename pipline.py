from typing import Union
from torch import nn
import torch
import dearpygui.dearpygui as dpg
from d2l import torch as d2l
from nodes.tools import ViewNode_2D
from core.node import Node



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
    def keep_train(sender=None, app_data=None, data_node=None):
        self = data_node.train_params.pipline
        if self:
            try:
                self.max_epoches = Pipline.get_params(data_node.train_params)['Max Epoches']
                self.train()
            except BaseException as err:
                raise RuntimeError("Error in keep training")
        else:
            Pipline.flow(data_node=data_node)
        
        

    def __init__(self, init_node):
        self.pipline = [Pipline.init_layer(init_node)]
        init_node.train_params.set_pipline(self)
        self.train_params = Pipline.get_params(init_node.train_params)
        print("\n\n\ttrain params: ", self.train_params)

        self.progress_board: list = init_node._output_attributes[1]._children
        self.progress_board: Union[ViewNode_2D, None] = self.progress_board[0]._data if len(self.progress_board) else None
        
    @staticmethod
    def init_normal(module: nn.Module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, mean=0, std=0.01)
            nn.init.zeros_(module.bias)

    @staticmethod
    def init_xavier(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
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
    def get_params(params_node) -> dict:
        train_params = dict()
        for param in params_node._params:
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
                       self.train_params.pop('Optimizer'), 
                       self.pipline[1:], widget=self.progress_board)
        print("pipline: ", self.net)
        

    def train(self, fake=False):
        if self.progress_board:
            self.net.board = d2l.ProgressBoard(widget=self.progress_board)

        self.trainer = d2l.Trainer(max_epochs=1 if fake else self.max_epoches, num_gpus=1)
        self.trainer.fit(self.net, self.pipline[0])

