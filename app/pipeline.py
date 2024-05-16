import pdb

import dearpygui.dearpygui as dpg


from typing import Union, Optional, Any
from torch import nn
import torch
from lightning import Trainer
import clearml

from core.node import Node
from core.utils import init_xavier
from app.lightning_module import Module
from app.lightning_data import DataModule



class Pipeline:

    debug = True

    @staticmethod
    def flow(sender=None, app_data=None, data_node: "nodes.DataNode"=None, fake=False):
        assert data_node
        try:
            for model_init in data_node._output_attributes[0]._children:
                print("Инициализация пайплайна") if Pipeline.debug else None
                self = Pipeline(data_node)
                print("Сбор слоёв") if Pipeline.debug else None
                self.collect_layers(model_init.get_node())
                print("Тренировка: ", fake) if Pipeline.debug else None
                self.train(fake)
            return self
        except BaseException as err:
            raise LookupError("Error in flow")
    
    @staticmethod
    def keep_train(sender=None, app_data=None, data_node: "nodes.DataNode"=None):
    
        if hasattr(data_node.train_params, "pipeline"):
            try:
                self = data_node.train_params.pipeline 
                self.max_epochs = Pipeline.get_params(data_node.train_params)['Max Epoches']
                self.train()
            except BaseException as err:
                raise RuntimeError("Error in keep training")
        else:
            Pipeline.flow(data_node=data_node)
        
        

    def __init__(self, init_node: "nodes.DataNode"):
        self.pipeline = [Pipeline.init_dataloader(init_node)]
        init_node.train_params.set_pipline(self)
        self.train_params = Pipeline.get_params(init_node.train_params)
        self.task: clearml.Task = clearml.Task.init(
                    project_name=self.train_params.pop("Project name"),
                    task_name=self.train_params.pop("Task name")
                )
        print("\n\n\ttrain params: ", self.train_params) if Pipeline.debug else None
        self.task.connect(self.train_params)

      
    
    @staticmethod
    def init_dataloader(data_node: Node):
        try:
            params = Pipeline.get_params(data_node)
            print("\tПараметры датасета: ", params) if Pipeline.debug else None
            transforms = params.pop('transforms') if 'transforms' in params else None
            return DataModule(data_node._data, transforms=transforms, **params)
        except: raise RuntimeError("Ошибка инициализации датасета")
        


    @staticmethod
    def init_layer(layer: Node) -> Any:
        try: 
            params = Pipeline.get_params(layer)
            # print("\nСлой: ", layer._data) if Pipeline.debug else None
            # print("Параметры слоя: ", params) if Pipeline.debug else None
            return layer._data(**params)
        except: raise RuntimeError("Ошибка инициализации слоя")
        

    @staticmethod
    def get_params(params_node: Node) -> dict:
        params = {}
        for param in params_node._params:
            returned = param.get_value()
            params.update(returned) if returned else None
        return params
    
    
    @staticmethod
    def save_weight(sender, app_data, train_params__file):
        train_params, filepath_uuid = train_params__file
        self = train_params.pipeline
        if self:
            filepath = dpg.get_value(filepath_uuid)
            torch.save(self.net.state_dict(), filepath)

    @staticmethod
    def load_weight(sender, app_data, train_params__file):
        train_params, filepath_uuid = train_params__file
        self: Pipeline = train_params.pipeline
        if not self:
            data_node = dpg.get_item_user_data(
                            dpg.get_item_parent(train_params._input_attributes[0]._linked_out_attr._uuid))  
            self = Pipeline.flow(data_node=data_node, fake=True)
            
        filepath = dpg.get_value(filepath_uuid)
        try:
            self.net.load_state_dict(torch.load(filepath))
        except BaseException as err:
            raise FileNotFoundError("Файл параметров не найден")
        
    

    def collect_layers(self, node: Node):
        self.pipeline.append(Pipeline.init_layer(node))
        
        while len(node := node._output_attributes[0]._children):
            node: Node = node[0].get_node()
            self.pipeline.append(Pipeline.init_layer(node))
        
        self.max_epochs = self.train_params.pop('Max Epoches')
        try:
            self.net = Module(sequential=nn.Sequential(*self.pipeline[1:]), optimizer=self.train_params.pop('Optimizer'),
                              lr=self.train_params.pop('Learning Rate'), loss_func=self.train_params.pop('Loss'))
            self.net.apply_init(self.pipeline[0], self.train_params.pop('Initialization'))
            print('\n\t\t\tИнициализированная сеть: ', self.net) if Pipeline.debug else None
            self.net.layer_summary((1,1,*self.pipeline[0].shape)) if Pipeline.debug else None
        except: raise RuntimeError("Возникла ошибка во время инициализации модели")
        

    def train(self, fake=False):
        try: self.trainer = Trainer(max_epochs=1 if fake else self.max_epochs, accelerator='gpu')
        except: raise RuntimeError("Ошибка инициализации тренировочного класса")
        try: self.trainer.fit(model=self.net, datamodule=self.pipeline[0])
        except: raise RuntimeError("Возникла ошибка во время обучения модели")

