import dearpygui.dearpygui as dpg
from core.utils import select_path
from torchvision.transforms import v2 


class ParamNode:

    def __init__(self, label: str, type: str, **params):
        self._uuid = dpg.generate_uuid()
        self._container_uuid = dpg.generate_uuid()
        self._label: str = label
        self._type: str = type
        self._params = params

    def info(self):
        return (self._uuid, self._container_uuid, self._label, self._type, self._params)
    
    def _submit(self, node_uuid):

        with dpg.node_attribute(parent=node_uuid, user_data=self, attribute_type=dpg.mvNode_Attr_Static, 
                                tag=self._container_uuid):
            self.__submit_in_container(self._container_uuid)
          
            
    def __submit_in_container(self, parent, inner_type=None):
        match_type = inner_type if inner_type else self._type
        match match_type:
            case 'int':
                dpg.add_input_int(**self._params, label=self._label, tag=self._uuid, parent=parent)
            case 'float':
                dpg.add_input_float(**self._params, label=self._label, tag=self._uuid, parent=parent)
            case 'text'|'text/tuple':
                dpg.add_input_text(**self._params, label=self._label, tag=self._uuid, parent=parent)
            case 'combo':
                dpg.add_combo(**self._params, label=self._label, tag=self._uuid, parent=parent)
            case 'collaps':
                with dpg.child_window(width=250, height=80, parent=parent):
                    self.checkboxes_uuids = []
                    with dpg.collapsing_header(label=self._label, default_open=True, tag=self._uuid):
                        for item in self._params['items']:
                            self.checkboxes_uuids.append(ParamNode(**item)\
                                .__submit_in_container(self._uuid, 'bool'))     
            case 'blank':
                dpg.add_text(**self._params, default_value=self._label, tag=self._uuid, parent=parent)
            case 'bool':
                with dpg.group(horizontal=True) as group:     
                    param = ParamNode(label=self._label, type=self._type, **self._params)
                    param.__submit_in_container(group)
                    return dpg.add_checkbox(default_value=False, 
                                            before=param._uuid,
                                            user_data=param,
                                            tag=self._uuid)
            case 'button':
                dpg.add_button(**self._params, label=self._label, tag=self._uuid, parent=parent)
            case 'file':
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_input_text(width=150, no_spaces=True, tag=self._uuid)
                    dpg.add_button(label="Path", user_data=self._uuid, callback=select_path)
                dpg.add_button(**self._params, label=self._label, user_data=(dpg.get_item_user_data(parent), self._uuid))
            case 'path':
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_input_text(**self._params, width=150, no_spaces=True, tag=self._uuid)
                    dpg.add_button(label="Path", user_data=self._uuid, callback=select_path)
    
                    
    def get_value(self, with_user_data=False):
        match self._type:
            case 'combo':
                choices = dpg.get_item_user_data(self._uuid)
                value = dpg.get_value(self._uuid)
                return {self._label: choices[value]}
                        
            case 'collaps':  
                values = [dpg.get_item_user_data(checkbox_uuid).get_value(True)
                          for checkbox_uuid 
                          in self.checkboxes_uuids 
                          if dpg.get_value(checkbox_uuid) is True]
                if self._label == 'Transforms':
                    if len(values): return {'transforms': v2.Compose([list(value.values())[0] for value in values])}
                    else: return {'transforms': None}
                return {self._label: values}
            case 'int' | 'float' | 'text' | 'text/tuple':
                value = dpg.get_value(self._uuid) 
                if self._type=='text/tuple':
                    value = list(map(int, value.split(", ")))
                return {self._label: dpg.get_item_user_data(self._uuid)(value) \
                                    if with_user_data \
                                    else value}
                       
            case 'blank':
                return {self._label: dpg.get_item_user_data(self._uuid)()} 
            case 'bool':
                return {self._label: dpg.get_item_user_data(self._uuid).get_value()\
                        if dpg.get_value(self._uuid) is True \
                        else None}
            case 'file'|'path'|'button':
                return None    
            case _:
                raise RuntimeError(f"Ошибка во время парсинга параметра: {self._type} - {self._label}")