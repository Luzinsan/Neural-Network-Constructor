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
import dearpygui.dearpygui as dpg




class TrainParamsNode(Node):
    
    WIDTH=150
    __params: dict[str, dict[str, Any]] = dict(
        Loss= 
            {"MSE": nn.MSELoss, 
            "Cross Entropy": F.cross_entropy, 
            "L1": nn.L1Loss},
        Optimizer=
            {"SGD":torch.optim.SGD, 
            "Adam":torch.optim.Adam, 
            "RMSprop":torch.optim.RMSprop,
            "Adagrad":torch.optim.Adagrad
            },
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

        self._add_input_attribute(InputNodeAttribute("train dataset"))
        if not default_params: 
            default_params = {key:list(choices.keys())[0] for key, choices in TrainParamsNode.__params.items()}
        
        def get_default(value):
            return default_params.get(value, list(TrainParamsNode.__params[value].keys())[0])
        
        train_params: list[dict[str, object]] = [
            {"label":"Название проекта", "type":'text', "default_value":default_params.get('Название задачи', data_node._label), "width":TrainParamsNode.WIDTH},
            {"label":"Название задачи", "type":'text', "default_value":default_params.get('Название проекта','DLC'), "width":TrainParamsNode.WIDTH},
            {
                "label":"Функция потерь", "type":'combo', "default_value":get_default('Loss'), 'tooltip':"""
### Функция потерь
___
+ MSE - критерий, который измеряет среднеквадратичную ошибку (квадрат нормы L2) между каждым элементом во входном x и целевом y
+ Cross Entropy - критерий вычисляет потерю перекрестной энтропии между входными логитами и таргетом. Используется для задачи классификации
+ L1 - критерий измеряет среднюю абсолютную ошибку (MAE) между каждым элементом во входном x и целевом y
                """,
                "items":tuple(TrainParamsNode.__params['Loss'].keys()), "user_data":TrainParamsNode.__params['Loss'], "width":TrainParamsNode.WIDTH
            },
            {
                "label":"Optimizer", "type":'combo', "default_value":get_default('Optimizer'), 'tooltip':"""
### Оптимизатор
+ SGD - стохастический градиентный спуск - итерационный метод оптимизации целевой функции с подходящими свойствами гладкости (например, дифференцируемость или субдифференцируемость)
+ Adam - сочетает в себе идеи RMSProp и оптимизатора импульса
+ RMSprop - среднеквадратичное распространение корня - это экспоненциально затухающее среднее значение. RMSprop вносит свой вклад в экспоненциально затухающее среднее значение прошлых «квадратичных градиентов»
+ Adagrad (алгоритм адаптивного градиента) - регулирует скорость обучения для каждого параметра на основе его предыдущих градиентов
                """,
                "items":tuple(TrainParamsNode.__params['Optimizer'].keys()), "user_data":TrainParamsNode.__params['Optimizer'], "width":TrainParamsNode.WIDTH
            },
            {
                "label":"Initialization", "type":'combo', "default_value":get_default('Initialization'), 'tooltip':"""
### Инициализация параметров
+ Normal - из нормального распределения (среднее=0, отклонение=0.01)
+ Xavier - из Ксавье распределения - подойдет для симметричных относительно нуля функций активации (например, Tanh), оставляет дисперсию весов одинаковой
                """,
                "items":tuple(TrainParamsNode.__params['Initialization'].keys()), "user_data":TrainParamsNode.__params['Initialization'], "width":TrainParamsNode.WIDTH
            },
            {"label":"Скорость обучения", "type":'float', "default_value":default_params.get('Скорость обучения', 0.05), "width":TrainParamsNode.WIDTH},
            {"label":"Эпохи", "type":'int', "default_value":default_params.get('Эпохи', 2), "width":TrainParamsNode.WIDTH},
            {"label":"Сохранить веса", "type":"file", "callback":Pipeline.save_weight},
            {"label":"Загрузить веса", "type":"file", "callback":Pipeline.load_weight},
            {"label":"Дообучить", "type":"button", "callback":Pipeline.keep_train, "user_data":data_node},
            {"label":"Прервать", "type":"button", "callback":Pipeline.terminate, "user_data":data_node},
            {"label":"Запустить обучение", "type":"button", "callback":Pipeline.flow, "user_data":data_node},
        ]
        self._add_params(train_params)
    
    
    
    def set_pipline(self, pipeline: Pipeline):
        self.pipeline = pipeline
        
    def set_datanode(self, datanode: dataset.DataNode):
        self.datanode = datanode

