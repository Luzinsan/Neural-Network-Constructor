
import dearpygui.dearpygui as dpg
from torch import nn
import torch.optim
import torch.nn.functional as F

from core.node import Node
from core.output_node_attr import OutputNodeAttribute

from core.link_node import LinkNode
from core import utils

from app.pipeline import Pipeline
from nodes.train_params_node import TrainParamsNode
from lightning import LightningDataModule
import os
from torch.utils.data import DataLoader



class DataNode(Node):

    @staticmethod
    def factory(name, data, params:tuple[dict]=None, default_params: dict[str, str]=None,**node_params):
        node = DataNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:tuple[dict]=None, default_params: dict[str, str]=None, **node_params):
        super().__init__(label, data, **node_params)
        self._add_output_attribute(OutputNodeAttribute("data"))
        self._add_output_attribute(OutputNodeAttribute("train graph"))
        self._add_output_attribute(OutputNodeAttribute("train params"))
        self._add_params(params)
        self._default_params = default_params


    def _submit(self, parent):
        super()._submit(parent)
        losses = {"MSE (squared L2)": nn.MSELoss, "Cross Entropy Loss": F.cross_entropy, "L1 Loss": nn.L1Loss}
        optimizers = {"SGD": torch.optim.SGD, "Adam":torch.optim.Adam, "Adadelta":torch.optim.Adadelta, "Adamax":torch.optim.Adamax}
        init_params = {'Default':None, 'Normal': utils.init_normal, 'Xavier': utils.init_xavier } 


        self.train_params: TrainParamsNode = TrainParamsNode('Train Params',
                                       train_params=(
                                            {"label":"Project name", "type":'text', "default_value":'Project', "width":150},
                                            {"label":"Task name", "type":'text', "default_value":'Experiment', "width":150},
                                            {"label":"Loss", "type":'combo', "default_value":self._default_params['Loss'], "width":150,
                                             "items":tuple(losses.keys()), "user_data":losses},
                                            {"label":"Optimizer", "type":'combo', "default_value":self._default_params['Optimizer'], "width":150,
                                             "items":tuple(optimizers.keys()), "user_data":optimizers},
                                            
                                            {"label":"Initialization", "type":'combo', "default_value":'Xavier', "width":150,
                                             "items":tuple(init_params.keys()), "user_data":init_params},
                                            
                                            {"label":"Learning Rate", "type":'float', "default_value":0.05, "width":150},
                                            {"label":"Max Epoches", "type":'int', "default_value":2, "width":150},
                                            {"label":"Save Weights", "type":"file", "callback":Pipeline.save_weight},
                                            {"label":"Load Weights", "type":"file", "callback":Pipeline.load_weight},
                                            {"label":"Train", "type":"button", "callback":Pipeline.flow, "user_data":self},
                                            {"label":"Continue Train", "type":"button", "callback":Pipeline.keep_train, "user_data":self}
                                       ),
                                       pos=(100, 250))
        
        self.train_params._submit(parent)
        LinkNode._link_callback(parent, (self._output_attributes[2]._uuid, self.train_params._input_attributes[0]._uuid))

