import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from core.param_node import ParamNode
from config.settings import _completion_theme


class Node:

    def __init__(self, label: str, data=None, **node_params):

        self.label = label
        self.uuid = dpg.generate_uuid()
        self.static_uuid = dpg.generate_uuid()
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = []
        self._data = data
        self.node_params = node_params

    def finish(self):
        dpg.bind_item_theme(self.uuid, _completion_theme)

    def add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)

    def add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)

    def add_params(self, params: tuple[dict]):
        if params:
            self._params += [ParamNode(**param) for param in params] 

    def custom(self):
        pass
        
    def execute(self):
        for attribute in self._output_attributes:
            attribute.execute(self._data)
        self.finish()


    def submit(self, parent):

        with dpg.node(**self.node_params, parent=parent, label=self.label, tag=self.uuid, user_data=self):

            for attribute in self._input_attributes:
                attribute.submit(self.uuid)

            for attribute in self._params:
                attribute.submit(self.uuid)
            
            with dpg.node_attribute(parent=self.uuid, attribute_type=dpg.mvNode_Attr_Static,
                                    user_data=self, tag=self.static_uuid):
                self.custom()

            for attribute in self._output_attributes:
                attribute.submit(self.uuid)