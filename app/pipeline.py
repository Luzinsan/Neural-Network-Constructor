
from __future__ import annotations
import pdb

import dearpygui.dearpygui as dpg
import clearml # type: ignore

from typing import Optional, Any, Union
from torch import nn
import torch
from lightning import Trainer

from core.node import Node
from app.lightning_module import Module
from app.lightning_data import DataModule
from nodes import dataset
from nodes import train_params_node
from core.utils import send_message, terminate_thread
import webbrowser
import threading


class Pipeline:

    debug = True

    def __init__(self, datanode: dataset.DataNode):
        send_message("Инициализация пайплайна", 'log') 
        datanode.train_params.set_pipline(self)
        self.train_params = datanode.train_params.get_params()
        send_message(self.train_params, 'log', 'Тренировочные параметры') 
        send_message("Сбор слоёв", 'log') 
        self.collect_layers(datanode)
        send_message("Инициализация сети", 'log') 
        try:
            self.net = Module(sequential=nn.Sequential(*self.pipeline), optimizer=self.train_params['Optimizer'],
                              lr=self.train_params['Learning Rate'], loss_func=self.train_params['Loss'])
            self.net.apply_init(self.dataset, self.train_params['Initialization'])
            send_message(self.net, 'log', 'Инициализированная сеть') 
            self.net.layer_summary((1,1,*self.dataset.shape)) if Pipeline.debug else None
        except (BaseException, RuntimeError, TypeError) as err: 
            send_message(err, 'error', "Возникла ошибка во время инициализации сети")
            raise err
        
        self.task: clearml.Task = clearml.Task.init(
                    project_name=self.train_params["Project name"],
                    task_name=self.train_params["Task name"],
                    continue_last_task=True)
        self.task.connect(self.train_params)
        link = self.task.get_output_log_web_page()
        send_message(link, 'log', 'Запуск clearml', 
                     lambda: webbrowser.open(link)) 

    
    @staticmethod
    def flow(sender=None, app_data=None, datanode: Optional[dataset.DataNode]=None):
        assert datanode
        Pipeline(datanode).train()
            
        
    @staticmethod
    def keep_train(sender, app_data, datanode: dataset.DataNode):
        if hasattr(datanode.train_params, "pipeline"):
            try: datanode.train_params.pipeline.train()
            except BaseException as err:
                send_message(err, 'error', "Возникла ошибка во время дообучения")
                raise err
        else:
            Pipeline.flow(datanode=datanode)
            
    @staticmethod
    def terminate(sender, app_data, datanode: dataset.DataNode):
        try:
            if hasattr(datanode.train_params, "pipeline"):
                self = datanode.train_params.pipeline
                if hasattr(self, "thread"):
                    terminate_thread(self.thread)
        except (BaseException, ValueError) as warn:
            send_message(warn, 'warning')
            raise warn
        
        
    @staticmethod
    def save_weight(sender, app_data, train_params__file):
        train_params, filepath_uuid = train_params__file
        self = train_params.pipeline
        if self:
            filepath = dpg.get_value(filepath_uuid)
            torch.save(self.net.state_dict(), filepath)

    @staticmethod
    def load_weight(sender, app_data, train_params__file: tuple[train_params_node.TrainParamsNode, int]):
        train_params, filepath_uuid = train_params__file
        self: Pipeline = train_params.pipeline
        if not self:
            datanode = train_params.datanode
            self = Pipeline(datanode=datanode)
        filepath = dpg.get_value(filepath_uuid)
        try: self.net.load_state_dict(torch.load(filepath))
        except BaseException as err:
            send_message(err, 'error', "Файл параметров не найден")
            raise err
        
    def collect_layers(self, node: Node):
        self.dataset = node.init_with_params('data')
        self.pipeline = []
        while next_node := node.next():
            # pdb.set_trace()
            if isinstance(next_node._data, list):
                for module_node in next_node._data:
                    self.pipeline.append(module_node.init_with_params())
            else:
                self.pipeline.append(next_node.init_with_params())
            node = next_node
        
    def train(self):
        try: self.trainer = Trainer(max_epochs=self.train_params['Max Epoches'], accelerator='gpu')
        except (BaseException, RuntimeError) as err: 
            send_message(err, 'error', "Ошибка инициализации тренировочного класса")
            raise err
        try:
            self.thread = threading.Thread(target=self.trainer.fit, args=(),name='train',
                                           kwargs=dict(model=self.net, datamodule=self.dataset))
            self.thread.start() 
        except (BaseException, RuntimeError) as err: 
            terminate_thread(self.thread)
            send_message(err, 'error', "Возникла ошибка во время обучения модели")
            raise err
        

