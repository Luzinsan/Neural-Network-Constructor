import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from core.param_node import ParamNode
from config.settings import _completion_theme
from typing import List, Any, Callable, Union, Tuple
import pdb


class Node:

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

    def delinks(self):
        for input in self._input_attributes:
            out = input._linked_out_attr
            out.remove_child(input)
        for out in self._output_attributes:
            (input.reset_linked_attr() for input in out._children)


    def __del__(self):
        self.delinks()
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
        self._finish()