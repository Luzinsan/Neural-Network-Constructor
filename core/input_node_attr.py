import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute

class InputNodeAttribute:

    def __init__(self, label: str, data: "core.Node"):

        self._uuid = dpg.generate_uuid()
        self._linked_out_attr: OutputNodeAttribute|None = None
        self._label = label


    def get_node(self) -> "core.Node":
        return dpg.get_item_user_data(dpg.get_item_parent(self._uuid))

    def set_linked_attr(self, out_attr: OutputNodeAttribute):
        self._linked_out_attr = out_attr

    def reset_linked_attr(self):
        self._linked_out_attr: OutputNodeAttribute|None = None

    def _submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, id=self._uuid):
            dpg.add_text(self._label)