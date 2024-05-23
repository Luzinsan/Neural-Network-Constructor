from __future__ import annotations
import dearpygui.dearpygui as dpg
from core import output_node_attr
from core import node
from typing import Optional, Union
from config.settings import BaseGUI

class InputNodeAttribute(BaseGUI):

    def __init__(self, label: str, uuid:Optional[int]=None, linked_out_attr: Optional[output_node_attr.OutputNodeAttribute] = None):
        super().__init__(uuid)
        self._linked_out_attr: Optional[output_node_attr.OutputNodeAttribute] = linked_out_attr
        self._label = label

    @staticmethod
    def factory(label:str, uuid:int, linked_out_attr: Optional[output_node_attr.OutputNodeAttribute]) -> InputNodeAttribute:
        return InputNodeAttribute(label, uuid, linked_out_attr)
    
    def convert_output_attr(self, map_output_uuids=None):
        if self._linked_out_attr \
            and isinstance(self._linked_out_attr, int):
            if map_output_uuids and (self._linked_out_attr in map_output_uuids):
                self._linked_out_attr = map_output_uuids[self._linked_out_attr]
            try: 
                if linked_attr := dpg.get_item_user_data(self._linked_out_attr):
                    self._linked_out_attr = linked_attr 
            except BaseException as err: print(err)

    def get_dict(self) -> dict:
        return dict(
            uuid=self.uuid,
            label=self._label,
            linked_out_attr=self._linked_out_attr.uuid if self._linked_out_attr else None 
        )

    def get_node(self) -> node.Node:
        return dpg.get_item_user_data(dpg.get_item_parent(self.uuid))

    def set_linked_attr(self, out_attr: output_node_attr.OutputNodeAttribute):
        self._linked_out_attr = out_attr

    def reset_linked_attr(self):
        self._linked_out_attr = None

    def _submit(self, parent):
        try:
            with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, tag=self.uuid):
                dpg.add_text(self._label)
        except: raise RuntimeError("Ошибка при попытке прикрепления входного атрибута")