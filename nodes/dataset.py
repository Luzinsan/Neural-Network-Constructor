
import dearpygui.dearpygui as dpg
from torch import nn
import torch.optim
from d2l import torch as d2l

from core.node import Node
from core.output_node_attr import OutputNodeAttribute

from core.link_node import LinkNode
from pipline import Pipline
from nodes.train_params_node import TrainParamsNode




class DataNode(Node):

    @staticmethod
    def factory(name, data, params:tuple[dict]=None, default_params: dict[str, str]=None,**node_params):
        node = DataNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:tuple[dict]=None, default_params: dict[str, str]=None, **node_params):
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
                 "Regression": d2l.Module, 
                 "Classification":d2l.Classifier}
        optimizers = {"SGD": torch.optim.SGD, "Adam":torch.optim.Adam, "Adadelta":torch.optim.Adadelta, "Adamax":torch.optim.Adamax}
        
        


        self.train_params = TrainParamsNode('Train Params',
                                       train_params=(
                                            {"label":"Task", "type":'combo', "default_value":self._default_params['Task'], "width":150,
                                             "items":tuple(tasks.keys()), "user_data":tasks},
                                            {"label":"Loss", "type":'combo', "default_value":self._default_params['Loss'], "width":150,
                                             "items":tuple(losses.keys()), "user_data":losses},
                                            {"label":"Optimizer", "type":'combo', "default_value":self._default_params['Optimizer'], "width":150,
                                             "items":tuple(optimizers.keys()), "user_data":optimizers},
                                            {"label":"Learning Rate", "type":'float', "default_value":0.05, "width":150},
                                            {"label":"Max Epoches", "type":'int', "default_value":2, "width":150},
                                            {"label":"Save Weights", "type":"file", "callback":Pipline.save_weight},
                                            {"label":"Load Weights", "type":"file", "callback":Pipline.load_weight},
                                            {"label":"Train", "type":"button", "callback":Pipline.flow, "user_data":self},
                                            {"label":"Continue Train", "type":"button", "callback":Pipline.keep_train, "user_data":self}
                                       ),
                                       pos=(100, 450))
        
        self.train_params.submit(parent)
        LinkNode._link_callback(parent, (self._output_attributes[2].uuid, self.train_params._input_attributes[0].uuid))

