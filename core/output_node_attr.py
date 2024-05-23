from __future__ import annotations
import dearpygui.dearpygui as dpg
from core import input_node_attr 
from typing import Optional, Union
from config.settings import Configs, BaseGUI
import pdb

class OutputNodeAttribute(BaseGUI):

    def __init__(self, label: str = "output", uuid: Optional[int] = None, children: Optional[list[input_node_attr.InputNodeAttribute]] = None):
        super().__init__(uuid)
        self._label = label
        self._children: list[input_node_attr.InputNodeAttribute] = children if children else []
        

    @staticmethod
    def factory(label:str, uuid:int, children:list) -> OutputNodeAttribute:
        return OutputNodeAttribute(label, uuid, children)
    
    def convert_input_attrs(self, map_input_uuids=None):
        for idx, attr in enumerate(self._children):
            if isinstance(attr, int):
                if map_input_uuids and (attr in map_input_uuids):
                    attr = map_input_uuids[attr]
                if attr_instance := dpg.get_item_user_data(attr):
                    self._children[idx] = attr_instance

    def get_dict(self) -> dict:
        return dict(
            uuid=self.uuid,
            label=self._label,
            children=[input.uuid 
                      for input 
                      in self._children]
        )   

    def add_child(self, child: input_node_attr.InputNodeAttribute):
        child.set_linked_attr(self)
        self._children.append(child)
        

    def remove_child(self, child: input_node_attr.InputNodeAttribute):
        self._children.remove(child)
        child.reset_linked_attr()


    def _submit(self, parent):
        try:
            with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                    user_data=self, tag=self.uuid):
                dpg.add_text(self._label)
        except: raise RuntimeError("Ошибка при попытке прикрепления выходного атрибута")