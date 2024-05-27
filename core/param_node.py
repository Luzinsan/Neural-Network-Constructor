from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.utils import select_path
from torchvision.transforms import v2 # type: ignore
from config.settings import Configs, BaseGUI
import external.DearPyGui_Markdown as dpg_markdown
from typing import Optional, Union


class ParamNode(BaseGUI):

    def __init__(self, label: str, type: str, uuid:Optional[int]=None, **params):
        super().__init__(uuid)
        self._container_uuid = BaseGUI.generate_uuid()
        self._label: str = label
        self._type: str = type
        self._params = params
        if type not in ['blank', 'bool']:
            self._params.update(
                {'width': Configs.width(),
                 })
       

    @staticmethod
    def factory(label: str, type: str, params: dict) -> ParamNode:
        return ParamNode(label, type, **params)

    def get_dict(self) -> dict:
        return dict(
            label=self._label,
            type=self._type,
            params=self._params
        )
    
    def _submit(self, node_uuid):
        try:
            with dpg.node_attribute(tag=self._container_uuid,
                                    parent=node_uuid, 
                                    user_data=self, 
                                    attribute_type=dpg.mvNode_Attr_Static):
                self._submit_in_container(self._container_uuid)
        except: raise RuntimeError("Ошибка при попытке прикрепления статического атрибута")
          
            
    def _submit_in_container(self, parent, inner_type=None):
        match_type = inner_type if inner_type else self._type
        tooltip = self._params.pop('tooltip', None)
        match match_type:
            case 'int':
                dpg.add_input_int(**self._params, label=self._label, min_value=1, min_clamped=True, step=2, tag=self.uuid, parent=parent)
            case 'float':
                dpg.add_input_float(**self._params, label=self._label, step=0.00001, min_value=0.000001, min_clamped=True, tag=self.uuid, parent=parent)
            case 'text'|'text/tuple':
                dpg.add_input_text(**self._params, label=self._label, tag=self.uuid, parent=parent)
            case 'combo':
                dpg.add_combo(**self._params, label=self._label, tag=self.uuid, parent=parent)
            case 'collaps':
                with dpg.child_window(width=250, height=80, parent=parent):
                    self.checkboxes_uuids = []
                    with dpg.collapsing_header(label=self._label, default_open=False, tag=self.uuid):
                        for item in self._params['items']:
                            self.checkboxes_uuids.append(ParamNode(**item)\
                                ._submit_in_container(self.uuid, 'bool'))     
            case 'blank':
                dpg.add_text(**self._params, default_value=self._label, tag=self.uuid, parent=parent)
            case 'bool':
                if self._type == 'bool': self._type='blank'
                with dpg.group(horizontal=True) as group:     
                    param = ParamNode(label=self._label, type=self._type, **self._params)
                    param._submit_in_container(group)
                    return dpg.add_checkbox(default_value=False, 
                                            before=param.uuid,
                                            user_data=param,
                                            tag=self.uuid)
            case 'button':
                dpg.add_button(**self._params, label=self._label, tag=self.uuid, parent=parent)
            case 'file':
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_input_text(width=150, no_spaces=True, tag=self.uuid)
                    dpg.add_button(label="Path", user_data=self.uuid, callback=select_path)
                dpg.add_button(**self._params, label=self._label, user_data=(dpg.get_item_user_data(parent), self.uuid))
            case 'path':
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_input_text(**self._params, width=150, no_spaces=True, tag=self.uuid)
                    dpg.add_button(label="Path", user_data=self.uuid, callback=select_path)
        if tooltip:
            with dpg.tooltip(dpg.last_item()):
                dpg_markdown.add_text(tooltip, wrap=300)
    
    @staticmethod
    def submit_config(name, params, defaults, parent):
        source_params = None
        if params:
            for param in params:
                if (label := param['label']) in defaults.keys():
                    param['default_value'] = defaults[label]
            source_params = [ParamNode(**param) for param in params]
            with dpg.collapsing_header(parent=parent, label=name, default_open=False) as submodule:
                for attribute in source_params:
                    attribute._submit_in_container(submodule)
        else:
            dpg.add_input_text(parent=parent, default_value=name, readonly=True, enabled=False)   
        return source_params
                    
    def get_value(self, with_user_data=False):
        match self._type:
            case 'combo':
                choices = dpg.get_item_user_data(self.uuid)
                value = dpg.get_value(self.uuid)
                value = value if value else self._params.get('default_value')
                return {self._label: choices[value]}
                        
            case 'collaps':  
                values = [dpg.get_item_user_data(checkbox_uuid).get_value(True)
                          for checkbox_uuid 
                          in self.checkboxes_uuids 
                          if dpg.get_value(checkbox_uuid) is True]
                if self._label == 'Transform':
                    if len(values): return {'Transform': v2.Compose([list(value.values())[0] for value in values])}
                    else: return {'Transform': None}
                return {self._label: values}
            case 'int' | 'float' | 'text' | 'text/tuple':
                value = dpg.get_value(self.uuid) 
                value = value if value else self._params.get('default_value')
                if self._type=='text/tuple' and isinstance(value, str):
                    value = list(map(int, value[1:-1].split(", ")))
                return {self._label: dpg.get_item_user_data(self.uuid)(value) \
                                    if with_user_data \
                                    else value}
            case 'blank':
                return {self._label: dpg.get_item_user_data(self.uuid)()} 
            case 'bool': # TODO: fix with with_user_data
                return {self._label: dpg.get_item_user_data(self.uuid).get_value()\
                        if dpg.get_value(self.uuid) is True \
                        else None}
            case 'file'|'path'|'button':
                return None    
            case _:
                raise RuntimeError(f"Ошибка во время парсинга параметра: {self._type} - {self._label}")