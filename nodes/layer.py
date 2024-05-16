from core.node import Node
from core.node_editor import NodeEditor
from core.input_node_attr import InputNodeAttribute
from core.output_node_attr import OutputNodeAttribute
from core.link_node import LinkNode
from core.dragndrop import DragSource
import dearpygui.dearpygui as dpg
from typing import Optional
from copy import deepcopy
import pdb


class LayerNode(Node):

    @staticmethod
    def factory(name, data, params:tuple[dict]|None=None, default_params: dict[str,str]=None, **node_params):
        node = LayerNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:tuple[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)
        self._add_input_attribute(InputNodeAttribute("data", self))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        self._add_params(params)


class ModuleNode:

    @staticmethod
    def factory(name, sequential: tuple[tuple[DragSource, dict]], params, default_params: dict[str,str]=None, **node_params):
        node = ModuleNode(name, sequential, params, default_params, **node_params)
        return node

    def __init__(self, name, sequential: tuple[tuple[DragSource, dict]], params, default_params: dict[str,str]=None, **node_params):
        self.sequential = sequential
        self.node_editor: NodeEditor = node_params['node_editor']
        self.pos: dict[str, int] = node_params.get('pos', {"x":0, "y":0})
        self.nodes: list[Node] = []

    @staticmethod
    def replace_default_params(sequential: tuple[tuple[object, dict]], new_defaults: dict):
        idx = 0
        updated_sequential = deepcopy(sequential)
        for i, layer_defaults in enumerate(updated_sequential):
            if len(layer_defaults) > 1: # если указаны новые параметры по умолчанию для слоя
                # params = item[1]
                for key in layer_defaults[1].keys():
                    if params := new_defaults.get(key):
                        layer_defaults[1][key] = params[idx]
                idx += 1
                          
        return updated_sequential
   
    def submit_module(self):
        for idx, node in enumerate(self.sequential):
            source: DragSource = node[0]
            if len(node)>1:
                defaults = node[1].copy()
                if source._generator.__qualname__ == 'ModuleNode.factory':
                    source._data = ModuleNode.replace_default_params(source._data, defaults)
                    
                    nodes, self.pos = ModuleNode(source._label, source._data, None, None, 
                                        node_editor=self.node_editor, pos={'x':0, "y":self.pos['y'] + 200}).submit_module()
                    # self.pos = {key: axis1 + axis2 for key, axis1, axis2 in zip(self.pos.keys(), self.pos.values(), pos.values())}
                    self.nodes += nodes
                    continue
                if source._params:
                    for param in source._params:
                        if (label := param['label']) in defaults.keys():
                            param['default_value'] = defaults[label]
            self.pos['x'] += 200
            params = {"pos":(self.pos['x'], self.pos['y'])}
            source = source._label, source._generator, source._data, source._params, source._default_params, params
            self.nodes.append(self.node_editor.on_drop(None, source, None))

        return self.nodes, self.pos
    

    def _submit(self, parent):
        self.nodes, _ = self.submit_module()
        for inx in range(len(self.nodes) - 1):
            first = self.nodes[inx]._output_attributes[0]
            sec = self.nodes[inx + 1]._input_attributes[0]
            LinkNode._link_callback(self.node_editor._uuid, (first._uuid, sec._uuid))

