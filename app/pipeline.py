
from __future__ import annotations
import pdb

import dearpygui.dearpygui as dpg
import clearml # type: ignore

from typing import Optional, Any, Union
from torch import nn
import torch
from lightning import Trainer

from core.node import Node
from app.lightning_module import Module, MultiBranch
from nodes import dataset
from nodes import train_params_node
from core.utils import send_message, terminate_thread
import threading


class Pipeline:

    debug = True

    def __init__(self, datanode: dataset.DataNode):
        send_message("Инициализация пайплайна", 'log') 
        datanode.train_params.set_pipline(self)
        self.train_params = datanode.train_params.get_params()
        send_message(self.train_params, 'log', 'Тренировочные параметры') 
        send_message("Сбор слоёв", 'log') 
        self.dataset = datanode.init_with_params('data')
        self.pipeline = Pipeline.collect_layers(datanode)
        send_message("Инициализация сети", 'log') 
        try:
            self.net = Module(sequential=nn.Sequential(*self.pipeline), optimizer=self.train_params['Optimizer'],
                              lr=self.train_params['Скорость обучения'], loss_func=self.train_params['Функция потерь'])
            self.net.apply_init(self.dataset, self.train_params['Initialization'])
            send_message(self.net, 'code', 'Инициализированная сеть') 
            self.net.layer_summary((self.dataset.batch_size,1,*self.dataset.shape)) if Pipeline.debug else None
        except (BaseException, RuntimeError, TypeError) as err: 
            send_message(err, 'error', "Возникла ошибка во время инициализации сети")
            raise err
        
        self.task: clearml.Task = clearml.Task.init(
                    project_name=self.train_params["Название проекта"],
                    task_name=self.train_params["Название задачи"],
                    continue_last_task=True)
        self.task.connect(self.train_params)
        send_message(f'[ClearML]({self.task.get_output_log_web_page()})', 
                     'log', 'Запуск ClearML') 

    
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
        
    @staticmethod
    def collect_multi_branch(next_node):
        multi_branch = []
        for branch in next_node:
            collected = Pipeline.collect_layers(branch, True)
            multi_branch.append(collected[:-1])
            node = collected[-1]
        return Node('Multi-branch', multi_branch)\
                        .init_with_params('multi_branch', node.get_params()), \
                node

    @staticmethod
    def convert_to_data(module_node):
        if isinstance(module_node, list):
            new_data_list = []
            for branch in module_node:
                new_data_list.append(Pipeline.convert_to_data(branch))
            return new_data_list
        else: return module_node.init_with_params()


    @staticmethod
    def collect_layers(node: Node, reckon_in = False):
        pipeline = [node.init_with_params()] if reckon_in else []
        while next_node := node.next():
            if isinstance(next_node, list): 
                multi_batch, node = Pipeline.collect_multi_branch(next_node)
                pipeline.append(multi_batch)
                continue
            elif isinstance(next_node._data, list):
                module_index = 0
                modules = next_node._data
                
                while module_index < len(modules):
                    module_node = modules[module_index]
                   
                    if isinstance(module_node, list):
                        if len(module_node) == 1:
                            module_node = module_node[0] 
                            for sub_node in module_node:
                                pipeline.append(sub_node.init_with_params())
                        else:
                            module_node = Pipeline.convert_to_data(module_node)
                            multi_batch = Node('Multi-branch', module_node)\
                                        .init_with_params('multi_branch', 
                                                        modules[module_index+1].get_params())

                            pipeline.append(multi_batch)
                            module_index += 1
                    else:  
                        pipeline.append(module_node.init_with_params())
                    module_index += 1
            elif next_node._data.__name__ == 'cat':
                pipeline.append(next_node)
                return pipeline
            else:
                pipeline.append(next_node.init_with_params())
            
            node = next_node
        return pipeline
        
    def train(self):
        try: self.trainer = Trainer(max_epochs=self.train_params['Эпохи'], accelerator='gpu')
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
        

