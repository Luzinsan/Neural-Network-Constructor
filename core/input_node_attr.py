import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute

class InputNodeAttribute:

    def __init__(self, label: str = "input", data=None):

        self._label = label
        self.uuid = dpg.generate_uuid()
        self._parent = None  # output attribute
        self._data = data

    def get_data(self):
        return self._data

    def set_parent(self, parent: OutputNodeAttribute):
        self._parent = parent

    def reset_parent(self, parent: OutputNodeAttribute):
        self._parent = None

    def submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, id=self.uuid):
            dpg.add_text(self._label)