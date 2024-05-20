from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from core.param_node import ParamNode
from app import lightning_data
from config.settings import _completion_theme
from typing import Any, Union
import pdb


class Node:

    debug=True

    def __init__(self, label: str, data=None, **node_params):

        self.__uuid: Union[int, str] = dpg.generate_uuid()
        self._label: str = label
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = []
        self._data = data
        self._node_params = node_params

    def _finish(self):
        dpg.bind_item_theme(self.__uuid, _completion_theme)

    def _add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)

    def _add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)

    def _add_params(self, params: list[dict[str, Any]]):
        if params:
            self._params += [ParamNode(**param) for param in params] 

    def _delinks(self):
        for input in self._input_attributes:
            if out := input._linked_out_attr:
                out.remove_child(input)
        for out in self._output_attributes:
            (input.reset_linked_attr() for input in out._children if input)

    def _del(self):
        self._delinks()
        dpg.delete_item(self.__uuid)
        del self
        

    def _submit(self, parent):
        with dpg.node(**self._node_params, parent=parent, label=self._label, tag=self.__uuid, user_data=self):
            for attribute in self._input_attributes:
                attribute._submit(self.__uuid)

            for attribute in self._params:
                attribute._submit(self.__uuid)
            
            for attribute in self._output_attributes:
                attribute._submit(self.__uuid)
        dpg.bind_item_handler_registry(self.__uuid, "hover_handler")
        self._finish()
        
    def next(self, out_idx: int=0, node_idx: int=0) -> Union[Node, None]:
        input_attrs: list[InputNodeAttribute] = self._output_attributes[out_idx]._children
        return input_attrs[node_idx].get_node() if input_attrs else None
        

    def init_with_params(self, mode='simple') -> Any:
        try: 
            params = self.get_params()
            print("\nСлой: ", self._data) if Node.debug else None
            print("Параметры слоя: ", params) if Node.debug else None
            match mode:
                case 'simple': return self._data(**params)
                case 'data': return lightning_data.DataModule(self._data, **params)
        except: raise RuntimeError("Ошибка инициализации слоя")
        
        
    def get_params(self) -> dict:
        params = {}
        for param in self._params:
            returned = param.get_value()
            params.update(returned) if returned else None
        return params
    