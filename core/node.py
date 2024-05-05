import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from core.param_node import ParamNode
from config.settings import _completion_theme
from typing import List, Any, Callable, Union, Tuple


class Node:

    def __init__(self, label: str, data=None, **node_params):

        self._label: str = label
        self._uuid: Union[int, str] = dpg.generate_uuid()
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = []
        self._data = data
        self._node_params = node_params

    def _finish(self):
        dpg.bind_item_theme(self._uuid, _completion_theme)

    def _add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)

    def _add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)

    def _add_params(self, params: tuple[dict]):
        if params:
            self._params += [ParamNode(**param) for param in params] 

    def _custom(self):
        pass
        
    def execute(self):
        for attribute in self._output_attributes:
            attribute.execute(self._data)
        


    def _submit(self, parent):
        with dpg.node(**self._node_params, parent=parent, label=self._label, tag=self._uuid, user_data=self):

            for attribute in self._input_attributes:
                attribute._submit(self._uuid)

            for attribute in self._params:
                attribute._submit(self._uuid)
            
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static,
                                    user_data=self):
                self._custom()

            for attribute in self._output_attributes:
                attribute._submit(self._uuid)
        self._finish()