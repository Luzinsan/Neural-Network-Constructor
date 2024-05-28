from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from core.param_node import ParamNode
from app import lightning_data
from app import lightning_module
from config.settings import _completion_theme, Configs, BaseGUI, hover_handler
from typing import Any, Union, Optional
from copy import deepcopy
from config import dicts 
import pdb


class Node(BaseGUI):

    debug=True

    def __init__(self, label: str, data=None, uuid:Optional[int]=None,
                 params:list[ParamNode]=[], 
                 **node_params):
        super().__init__(uuid)
        self._label: str = label
        self._data = data
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = deepcopy(params)
        self._node_params = node_params
        
    @staticmethod    
    def init_factory(editor, 
                     label: str, 
                     input_attributes: list[dict], 
                     output_attributes: list[dict], 
                     params: list[dict], 
                     node_params: dict) -> Node:
        
        params_instances: list[ParamNode] = \
            [ParamNode.factory(
                attr['label'], 
                attr['type'], 
                attr['params']) 
             for attr 
             in params]
        node = Node(label, 
                    dicts.modules[label].func, 
                    params=params_instances, 
                    **node_params)
        map_input_uuids = {attr['uuid']: node._add_input_attribute(
                                        InputNodeAttribute.factory(
                                            attr['label'], 
                                            attr['uuid'], 
                                            attr['linked_out_attr'])
                                        ).uuid
                            for attr 
                            in input_attributes}
        map_output_uuids = {attr['uuid']: node._add_output_attribute(
                                        OutputNodeAttribute.factory(
                                            attr['label'], 
                                            attr['uuid'], 
                                            attr['children'])
                                        ).uuid
                            for attr 
                            in output_attributes}
        editor.add_node(node)
        node._submit(editor.uuid)
        return node, map_input_uuids, map_output_uuids

    def _finish(self):
        dpg.bind_item_theme(self.uuid, _completion_theme)

    def _add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)
        return attribute

    def _add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)
        return attribute

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
        dpg.delete_item(self.uuid)
        del self
        

    def _submit(self, parent):
        self._node_editor_uuid = parent
        with dpg.node(tag=self.uuid,
                      parent=parent, 
                      label=self._label, 
                      user_data=self, 
                      **self._node_params):
            for attribute in self._input_attributes:
                attribute._submit(self.uuid)

            for attribute in self._params:
                attribute._submit(self.uuid)
            
            for attribute in self._output_attributes:
                attribute._submit(self.uuid)
           
        dpg.bind_item_handler_registry(self.uuid, hover_handler)
        self._finish()
        
        
    def get_dict(self) -> dict:
        return dict(
            uuid=self.uuid,
            label=self._label, 
            input_attributes=[attr.get_dict() 
                              for attr 
                              in self._input_attributes],
            output_attributes=[attr.get_dict() 
                               for attr 
                               in self._output_attributes],
            params=[param.get_dict() 
                    for param 
                    in self._params],
            node_params=self._node_params,
        )
    
    def next(self, out_idx: int=0, node_idx: int=0) -> Union[Node, None]:
        input_attrs: list[InputNodeAttribute] = \
            self._output_attributes[out_idx]._children
        nodes = [input_attr.get_node() for input_attr in input_attrs] if input_attrs else None
        if nodes and len(nodes) == 1:
            nodes = nodes[0]
        return nodes
        

    def init_with_params(self, mode='simple', params: Optional[dict]=None) -> Any:
        try: 
            params = params if params else self.get_params()
            print("\nСлой: ", self._data) if Node.debug else None
            print("Параметры слоя: ", params) if Node.debug else None
            match mode:
                case 'simple': return self._data(**params)
                case 'data': return lightning_data.DataModule(self._data, **params)
                case 'multi_branch': return lightning_module.MultiBranch(self._data, **params)
        except: raise RuntimeError("Ошибка инициализации слоя")
        
        
    def get_params(self) -> dict:
        params = {}
        for param in self._params:
            returned = param.get_value()
            params.update(returned) if returned else None
        return params
    