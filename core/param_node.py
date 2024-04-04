import dearpygui.dearpygui as dpg
from core.utils import select_path


class ParamNode:

    def __init__(self, label: str, type: str, **params):

        self.uuid = dpg.generate_uuid()
        self._label: str = label
        self._type: str = type
        self._params = params


    def submit(self, parent):

        with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Static):
            match self._type:
                case 'int':
                    dpg.add_input_int(**self._params, label=self._label, tag=self.uuid)
                case 'float':
                    dpg.add_input_float(**self._params, label=self._label, tag=self.uuid)
                case 'text'|'text/tuple':
                    dpg.add_input_text(**self._params, label=self._label, tag=self.uuid)
                case 'combo':
                    dpg.add_combo(**self._params, label=self._label, tag=self.uuid)
                case 'button':
                    dpg.add_button(**self._params, label=self._label, tag=self.uuid)
                case 'file':
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(width=150, no_spaces=True, tag=self.uuid)
                        dpg.add_button(label="Path", user_data=self.uuid, callback=select_path)
                    dpg.add_button(**self._params, label=self._label, user_data=(dpg.get_item_user_data(parent), self.uuid))