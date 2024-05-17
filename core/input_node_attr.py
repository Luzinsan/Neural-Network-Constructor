from __future__ import annotations
import dearpygui.dearpygui as dpg
from core import output_node_attr
from core import node
from typing import Optional

class InputNodeAttribute:

    def __init__(self, label: str, data: node.Node):

        self._uuid = dpg.generate_uuid()
        self._linked_out_attr: Optional[output_node_attr.OutputNodeAttribute] = None
        self._label = label


    def get_node(self) -> node.Node:
        return dpg.get_item_user_data(dpg.get_item_parent(self._uuid))

    def set_linked_attr(self, out_attr: output_node_attr.OutputNodeAttribute):
        self._linked_out_attr = out_attr

    def reset_linked_attr(self):
        self._linked_out_attr = None

    def _submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, id=self._uuid):
            dpg.add_text(self._label)