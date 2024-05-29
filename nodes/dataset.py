from __future__ import annotations
from typing import Optional, Any

from core.node import Node
from core.node_editor import NodeEditor
from core.output_node_attr import OutputNodeAttribute
from core.link_node import LinkNode
from nodes.train_params_node import TrainParamsNode
import dearpygui.dearpygui as dpg



class DataNode(Node):

    @staticmethod
    def factory(label:str, data, params:list[dict[str, Any]], 
                default_params: Optional[dict[str, str]]=None,**node_params):
        node = DataNode(label, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:list[dict[str, Any]], 
                 default_params: Optional[dict[str, str]]=None, **node_params):
        node_params['pos'] = NodeEditor.mouse_pos
        super().__init__(label, data, **node_params)
        self._add_output_attribute(OutputNodeAttribute("data"))
        self._add_output_attribute(OutputNodeAttribute("train graph"))
        self._add_output_attribute(OutputNodeAttribute("train params"))
        if params: self._add_params(params)
        self._default_params = default_params
        
    def _del(self):
        super()._del()
        self.train_params._del()

    def _submit(self, parent:int):
        super()._submit(parent)
        
        self.train_params: TrainParamsNode = TrainParamsNode('Train Params', 
                                                             data_node=self, 
                                                             default_params=self._default_params, 
                                                             pos=(self._node_params['pos'][0], 
                                                                  self._node_params['pos'][1] + 250))
        editor: NodeEditor = dpg.get_item_user_data(dpg.get_item_parent(parent))
        editor.add_node(self.train_params)
        self.train_params._submit(parent)
        self.train_params.set_datanode(self)
        LinkNode._link_callback(parent, 
                                (self._output_attributes[2].uuid, 
                                 self.train_params._input_attributes[0].uuid))

