from __future__ import annotations

from typing import Any, Optional
from torch import nn
import torch.optim
import torch.nn.functional as F

from core.node import Node
from core.input_node_attr import InputNodeAttribute
from core import utils

from app.pipeline import Pipeline
from nodes import dataset




class TrainParamsNode(Node):
    
    WIDTH=150
    __params: dict[str, dict[str, Any]] = dict(
        Loss= 
            {"MSE (squared L2)": nn.MSELoss, 
            "Cross Entropy Loss": F.cross_entropy, 
            "L1 Loss": nn.L1Loss},
        Optimizer=
            {"SGD":torch.optim.SGD, 
            "Adam":torch.optim.Adam, 
            "Adadelta":torch.optim.Adadelta, 
            "Adamax":torch.optim.Adamax},
        Initialization=
            {'Default':None, 
            'Normal': utils.init_normal, 
            'Xavier': utils.init_xavier},
    )


    @staticmethod
    def factory(name, data_node: dataset.DataNode, default_params: Optional[dict[str, str]]=None, **node_params):
        node = TrainParamsNode(name, data_node, default_params, **node_params)
        return node

    def __init__(self, label: str, data_node: dataset.DataNode, default_params: Optional[dict[str, str]]=None, **node_params):
        super().__init__(label, data_node, **node_params)

        self._add_input_attribute(InputNodeAttribute("train dataset", self))
        if not default_params: 
            default_params = {key:list(choices.keys())[0] for key, choices in TrainParamsNode.__params.items()}
        
        def get_default(value):
            return default_params.get(value, list(TrainParamsNode.__params[value].keys())[0])
        
        train_params: list[dict[str, object]] = [
            {"label":"Project name", "type":'text', "default_value":default_params.get('Project name','DLC'), "width":TrainParamsNode.WIDTH},
            {"label":"Task name", "type":'text', "default_value":default_params.get('Task name', data_node._label), "width":TrainParamsNode.WIDTH},
            {
                "label":"Loss", "type":'combo', "default_value":get_default('Loss'), 
                "items":tuple(TrainParamsNode.__params['Loss'].keys()), "user_data":TrainParamsNode.__params['Loss'], "width":TrainParamsNode.WIDTH
            },
            {
                "label":"Optimizer", "type":'combo', "default_value":get_default('Optimizer'), 
                "items":tuple(TrainParamsNode.__params['Optimizer'].keys()), "user_data":TrainParamsNode.__params['Optimizer'], "width":TrainParamsNode.WIDTH
            },
            {
                "label":"Initialization", "type":'combo', "default_value":get_default('Initialization'), 
                "items":tuple(TrainParamsNode.__params['Initialization'].keys()), "user_data":TrainParamsNode.__params['Initialization'], "width":TrainParamsNode.WIDTH
            },
            {"label":"Learning Rate", "type":'float', "default_value":default_params.get('Learning Rate', 0.05), "width":TrainParamsNode.WIDTH},
            {"label":"Max Epoches", "type":'int', "default_value":default_params.get('Max Epoches', 2), "width":TrainParamsNode.WIDTH},
            {"label":"Save Weights", "type":"file", "callback":Pipeline.save_weight},
            {"label":"Load Weights", "type":"file", "callback":Pipeline.load_weight},
            {"label":"Continue Train", "type":"button", "callback":Pipeline.keep_train, "user_data":data_node},
            {"label":"Terminate", "type":"button", "callback":Pipeline.terminate, "user_data":data_node},
            {"label":"Train", "type":"button", "callback":Pipeline.flow, "user_data":data_node},
        ]
        self._add_params(train_params)
    
    
    
    def set_pipline(self, pipeline: Pipeline):
        self.pipeline = pipeline
        
    def set_datanode(self, datanode: dataset.DataNode):
        self.datanode = datanode

