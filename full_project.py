import dearpygui.dearpygui as dpg
import external.DearPyGui_Markdown as dpg_markdown
from typing import Optional, DefaultDict, NamedTuple, Any, Union
from collections import namedtuple, defaultdict
from copy import deepcopy

import datetime, ctypes, re, os, threading, sys
import json

import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 # type: ignore
import torchvision.datasets as ds # type: ignore
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule, Trainer
import clearml # type: ignore



reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
float32 = torch.float32



#region setup
font_size = 14
default_path = './config/fonts/Comfortaa-Regular.ttf'
bold_path = './config/fonts/InterTight-Bold.ttf'
italic_path = './config/fonts/InterTight-Italic.ttf'
italic_bold_path = './config/fonts/InterTight-BoldItalic.ttf'

def add_font(file, size: int | float, parent=0, **kwargs) -> int:
    if not isinstance(size, (int, float)):
        raise ValueError(f'font size must be an integer or float. Not {type(size)}')

    with dpg.font(file, size, parent=parent, **kwargs) as font:
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    return font

def load() -> int:
    dpg_markdown.set_font_registry(dpg.add_font_registry())
    dpg_markdown.set_add_font_function(add_font)

    return dpg_markdown.set_font(
        font_size=font_size,
        default=default_path,
        bold=bold_path,
        italic=italic_path,
        italic_bold=italic_bold_path
    )

dpg.create_context()
dpg.create_viewport()
dpg.bind_font(load())
dpg.setup_dearpygui()
########################################################################################################################
def on_exit(sender, app_data, user_data):
    print("closed")
    dpg.stop_dearpygui()
    dpg.destroy_context()
dpg.set_exit_callback(on_exit)
########################################################################################################################
# Settings
########################################################################################################################
class Configs:
    
    __width: int = 150
    __logger: bool = False
    __max_uuid: int = 0
    
    @staticmethod
    def width() -> int:
        return Configs.__width
    @staticmethod
    def logger() -> bool:
        return Configs.__logger
    @staticmethod
    def set_logger(s, check_value, u):
        Configs.__logger = check_value
    @staticmethod
    def uuid():
        return Configs.__max_uuid
    @staticmethod
    def set_uuid(new_uuid=None):
        if new_uuid: 
            Configs.__max_uuid = new_uuid
        else:
            new_uuid = dpg.generate_uuid()
            if Configs.__max_uuid < new_uuid: 
                Configs.__max_uuid = new_uuid
            else: Configs.__max_uuid += 1
        return Configs.__max_uuid
    @staticmethod
    def reset_uuid():
        Configs.__max_uuid = 0
         
class BaseGUI:
    
    __global_max_uuid = 0
    __reserved_uuids: list[int] = []

    def __init__(self, uuid:Optional[int]=0):
        self._uuid = BaseGUI.generate_uuid(uuid)
    
    @property
    def uuid(self):
        return self._uuid
    
    @staticmethod
    def generate_uuid(uuid: Optional[int]=None) -> int:
        if uuid: 
            if uuid in BaseGUI.__reserved_uuids:
                uuid = BaseGUI._refresh_uuid(uuid)
        else:
            BaseGUI.__global_max_uuid = uuid = BaseGUI._refresh_uuid(dpg.generate_uuid())
            
        BaseGUI.__reserved_uuids.append(uuid)
        print("max: ", BaseGUI.__global_max_uuid, "\n",
              "reserved: ", BaseGUI.__reserved_uuids, "\n",
              "uuid: ", uuid) if False else None
        return uuid
    
    @staticmethod
    def _refresh_uuid(uuid: int) -> int:
        while uuid in BaseGUI.__reserved_uuids:
            uuid = dpg.generate_uuid() 
        return uuid
        
########################################################################################################################
# Themes
########################################################################################################################
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (227, 243, 255), category=dpg.mvThemeCat_Nodes) # Цвет сетки для узлов
        dpg.add_theme_color(dpg.mvNodeCol_GridLine, (255, 255, 255), category=dpg.mvThemeCat_Nodes) # Цвет линий в сетке
        dpg.add_theme_color(dpg.mvNodeCol_Link, (64, 64, 219), category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (35, 35, 194), category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (27, 27, 150), category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_Pin, (0, 0, 255), category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (0, 0, 0), category=dpg.mvThemeCat_Nodes)

        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (227, 255, 255), category=dpg.mvThemeCat_Core) # Цвет фона в контейнерах
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255), category=dpg.mvThemeCat_Core) # Цвет общего фона
        
        dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, [128, 200, 255], category=dpg.mvThemeCat_Core) # Цвет фона менюшек
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (227, 243, 255), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Header, (189, 189, 242), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (189, 189, 242), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Button, [152, 103, 204]) # Цвет кнопок в логгере
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Border, (179, 222, 255), category=dpg.mvThemeCat_Core) # Цвет оконтовок
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (227, 255, 255), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,  (142, 130, 217), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (255, 255, 255), category=dpg.mvThemeCat_Core)
dpg.bind_theme(global_theme)
########################################################################################################################
with dpg.theme() as _source_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, [152, 103, 204]) # Цвет кнопок в контейнерах
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [182, 156, 230]) # Цвет кнопок при наведении
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [227, 204, 237]) # Цвет кнопок при нажатии
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255), category=dpg.mvThemeCat_Core)
########################################################################################################################        
with dpg.theme() as _completion_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar, [142, 130, 217], category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, [136, 0, 206], category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, [145, 106, 212], category=dpg.mvThemeCat_Nodes)

        dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, [204, 204, 255], category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, [204, 204, 255], category=dpg.mvThemeCat_Nodes)
        dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, [175, 218, 252], category=dpg.mvThemeCat_Nodes)

        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (173, 194, 255), category=dpg.mvThemeCat_Core) 
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (227, 243, 255), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Button, (227, 243, 255), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 0, 230), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (227, 243, 255), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Header, (189, 189, 242), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
########################################################################################################################    
def curent_item(uuid):
    try:
        message = f"Текущий элемент: {dpg.get_item_label(uuid)}"
        dpg.configure_item('hover_logger', default_value=message)
    except BaseException: pass
          
def popup_tooltip(parent):
    if (source := dpg.get_item_user_data(parent)):
        # проверка на core.dragndrop.DragSource 
        # (для isinstance нужно импортировать класс, что приводит к циклическому импорту)
        if not hasattr(source, '_image'): return 
        if image := source._image:
            width, height, channels, data = dpg.load_image(image)
            with dpg.texture_registry():
                image = dpg.add_static_texture(width=width, height=height, default_value=data)    
        if source._popup or source._image or source._details:
            wrap=800
            with dpg.popup(parent):
                if popup := source._popup:
                    dpg_markdown.add_text(popup, wrap=wrap)
                if details := source._details:
                    with dpg.collapsing_header(label="Подробнее"):
                        dpg_markdown.add_text(details, wrap=wrap)
                if image: 
                    dpg.add_image(image, width=wrap, height=wrap * height / width)
        if tooltip:= source._tooltip:
            with dpg.tooltip(parent):
                dpg_markdown.add_text(tooltip, wrap=200)      
  
try:
    with dpg.item_handler_registry(tag=BaseGUI.generate_uuid()) as hover_handler:
        dpg.add_item_hover_handler(callback=lambda s,a,u: curent_item(a))   
        dpg.add_item_clicked_handler(callback=lambda s, a, u: popup_tooltip(a[1]))     
except SystemError as err: print("Удаление узла", err)
#endregion

#region utils
accepted_initializations = [nn.Linear, nn.Conv2d]

def init_normal(module: nn.Module):
    try:
        if type(module) in accepted_initializations:
            nn.init.normal_(module.weight, mean=0, std=0.01)
            nn.init.zeros_(module.bias)
    except: raise RuntimeError("Ошибка во время предварительной инициализации слоя нормальным распределением")


def init_xavier(module: nn.Module):
    try: 
        if type(module) in accepted_initializations:
            nn.init.xavier_uniform_(module.weight)
    except: raise RuntimeError("Ошибка во время предварительной инициализации слоя распределением Ксавье")


def terminate_thread(thread: threading.Thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """
    if not thread.is_alive():
        raise ValueError("Обучение не запущено")

    exc = ctypes.py_object(SystemExit)
    if id := thread.ident:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(id), exc)
        if res == 0:
            raise ValueError("nonexistent thread id")
        elif res > 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(id, None)
            send_message("Обучение остановлено", 'warning')
            raise KeyboardInterrupt("Обучение остановлено")


def send_message(message, type_message: str = 'error', brief:Optional[str]=None, callback=None): 
    dpg.set_item_label('menu_message_logger', f"Сообщения: {brief if brief else message}")
    
    message = f"{brief}.\nПодробнее: {message}" if brief else message
     
    map_style_message = {
        'error': '<font color="(255,0,0)">{}</font>', 
        'warning':'<font color="(255,255,0)">{}</font>',
        'log':'<font color="(0,255,0)">{}</font>',
        'code':'```\n{}\n```',
    }
    style = map_style_message.get(type_message)
    if style: 
        if type_message=='error' and re.search('Traceback', message):
            message = "\n".join(re.findall(r'^.*Error:.*$', message, re.MULTILINE))
        message=style.format(message)
    
    group = dpg_markdown.add_text(markdown_text=message, parent='message_logger', 
                                  wrap=dpg.get_item_width('message_logger'))
    
    dpg.set_y_scroll('message_logger', 
                     dpg.get_y_scroll_max('message_logger') + dpg.get_item_height(group))
    print(message) if Configs.logger() else None

def select_path(sender, app_data, user_data):
    dpg.set_item_user_data('file_dialog', user_data)
    dpg.show_item('file_dialog')


def set_path(sender, app_data):
    tag_path = dpg.get_item_user_data('file_dialog')
    dpg.configure_item(tag_path, default_value=app_data['file_path_name'])


with dpg.file_dialog(directory_selector=False, show=False, callback=set_path, tag="file_dialog",
                     width=700, height=400, modal=True):
    dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255), custom_text="[Calc]")
    dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")
    dpg.add_file_extension(".params", color=(0, 255, 0, 255), custom_text="[Params]")


with dpg.file_dialog(
        directory_selector=False,
        show=False,
        modal=True,
        width=700, height=400, 
        default_filename=datetime.datetime.now().strftime('%Y_%m_%d'),
        tag='json_file',
        ):
    dpg.add_file_extension('.json')
    dpg.add_file_extension('', color=(150, 255, 150, 255))

#endregion

class OutputNodeAttribute(BaseGUI): ...
class InputNodeAttribute(BaseGUI): ...
class Node(BaseGUI): ...
class DataNode(Node): ...
class ParamNode(BaseGUI): ...
class DragSource: ...
class TrainParamsNode(Node): ...
class Pipeline: ...
#region nodes


class InputNodeAttribute(BaseGUI):

    def __init__(self, label: str, uuid:Optional[int]=None, linked_out_attr: Optional[OutputNodeAttribute] = None):
        super().__init__(uuid)
        self._linked_out_attr: Optional[OutputNodeAttribute] = linked_out_attr
        self._label = label

    @staticmethod
    def factory(label:str, uuid:int, linked_out_attr: Optional[OutputNodeAttribute]) -> InputNodeAttribute:
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

    def get_node(self) -> Node:
        return dpg.get_item_user_data(dpg.get_item_parent(self.uuid))

    def set_linked_attr(self, out_attr: OutputNodeAttribute):
        self._linked_out_attr = out_attr

    def reset_linked_attr(self):
        self._linked_out_attr = None

    def _submit(self, parent):
        try:
            with dpg.node_attribute(parent=parent, user_data=self, attribute_type=dpg.mvNode_Attr_Input, tag=self.uuid):
                dpg.add_text(self._label)
        except: raise RuntimeError("Ошибка при попытке прикрепления входного атрибута")


class OutputNodeAttribute(BaseGUI):

    def __init__(self, label: str = "output", uuid: Optional[int] = None, children: Optional[list[InputNodeAttribute]] = None):
        super().__init__(uuid)
        self._label = label
        self._children: list[InputNodeAttribute] = children if children else []
        

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

    def add_child(self, child: InputNodeAttribute):
        child.set_linked_attr(self)
        self._children.append(child)
        

    def remove_child(self, child: InputNodeAttribute):
        self._children.remove(child)
        child.reset_linked_attr()


    def _submit(self, parent):
        try:
            with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                    user_data=self, tag=self.uuid):
                dpg.add_text(self._label)
        except: raise RuntimeError("Ошибка при попытке прикрепления выходного атрибута")


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
                dpg.add_input_float(**self._params, format='%.5f', label=self._label, step=0.00001, min_value=0.000001, min_clamped=True, tag=self.uuid, parent=parent)
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
                self._params['default_value'] = self._label
                dpg.add_text(**self._params, tag=self.uuid, parent=parent)
            case 'bool':
               
                with dpg.group(horizontal=True) as group:  
                    param = ParamNode(label=self._label, type='blank'
                                                            if self._type == 'bool' 
                                                            else self._type, 
                                     **self._params)
                    param._submit_in_container(group)
                    default_value = self._params.get("default_value", False)
                    return dpg.add_checkbox(default_value=default_value if isinstance(default_value, bool) else False, 
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
            if defaults:
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
            case 'bool':
                if dpg.does_item_exist(self.uuid):
                    obj = dpg.get_item_user_data(self.uuid)
                else: return {self._label: self._params['default_value']} 
                if (not obj) or (obj._type == 'blank' and not dpg.get_item_user_data(obj.uuid)): 
                    return {self._label: dpg.get_value(self.uuid)}
                return {self._label: obj.get_value()\
                                    if dpg.get_value(self.uuid) is True \
                                    else None}
            case 'file'|'path'|'button':
                return None    
            case _:
                raise RuntimeError(f"Ошибка во время парсинга параметра: {self._type} - {self._label}")


class LinkNode(BaseGUI):

    def __init__(self, input_uuid:int, output_uuid:int):
        super().__init__()
        self._input_attr = input_uuid
        self._output_attr = output_uuid

    def get_attrs(self):
        return self._input_attr, self._output_attr

    @staticmethod
    def _link_callback(node_editor_uuid: int, in_out_uuids: tuple[int, int]):
        output_attr_uuid, input_attr_uuid = in_out_uuids

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        link_node = LinkNode(input_attr_uuid, output_attr_uuid)
        dpg.add_node_link(*link_node.get_attrs(), parent=node_editor_uuid, user_data=link_node, tag=link_node.uuid)
        output_attr.add_child(input_attr)


    @staticmethod
    def _delink_callback(node_editor_uuid: int, link_uuid: int):
        link: LinkNode = dpg.get_item_user_data(link_uuid)
        input_attr_uuid, output_attr_uuid = link.get_attrs()

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        output_attr.remove_child(input_attr)
        dpg.delete_item(link_uuid)
        del link
        
    
    @staticmethod
    def _link_nodes(left_node, right_node, editor_uuid):
        first = left_node._output_attributes[0]
        sec = right_node._input_attributes[0]
        LinkNode._link_callback(editor_uuid, (first.uuid, sec.uuid))
    

    @staticmethod
    def link_nodes(nodes, editor_uuid):
        for inx in range(len(nodes) - 1):
            LinkNode._link_nodes(nodes[inx], nodes[inx + 1], editor_uuid)
    
    # BUG: input_attr почему-то иногда isinstance(input_attr, int)
    # BUG: multibranch модели линкуются только по первой ветке
    @staticmethod
    def link_by_children(node: Node, editor_uuid):
        output_attr = node._output_attributes[0]
        for input_attr in output_attr._children:
            LinkNode._link_callback(editor_uuid, (output_attr.uuid, input_attr.uuid))
            LinkNode.link_by_children(input_attr.get_node(), editor_uuid)


class NodeEditor(BaseGUI):

    mouse_pos = [0, 0]
    
    def __init__(self) -> None:
        super().__init__()
        self.__nodes: list[Node] = [] 
    
    @staticmethod
    def factory(app_data: tuple, node_params: dict=None, module=False):
        label, generator, data, params, default_params, node_params = app_data
        if module: default_params = {'part_of_module':True}
        node: Node = generator(label, data, params, default_params, **node_params) \
                if node_params \
                else generator(label, data, params, default_params)
        return node      

    def add_node(self, node: Node):
        self.__nodes.append(node)

    def on_drop(self, sender: int, node, node_params: dict, module=False):
        generator = node[1].__qualname__
        node = NodeEditor.factory(node, node_params, module)
        if not module: 
            if not (generator=='ModuleNode.factory'):
                 self.add_node(node)
            node._submit(self.uuid)
        return node

    def clear(self):
        dpg.delete_item(self.uuid, children_only=True)
        dpg.add_node(parent=self.uuid, label='ref node', show=True, draggable=False)
        self.__nodes.clear()
        Configs.reset_uuid()
        
   
    def callback_file(self, target):
        dpg.configure_item('json_file', show=True, callback=target)
       
    def save(self, sender, data, user_data):
        setting_dict = {}
        for node in self.__nodes:
            node = node.get_dict()
            setting_dict[f"{node['label']}_{node['uuid']}"] = node
        if len(setting_dict):
            with open(data['file_path_name'], 'w') as fp:
                json.dump(setting_dict, fp, indent=4)
                
                
    def open(self, sender, data):
        setting_dict = None
        with open(data['file_path_name']) as fp:
            setting_dict = json.load(fp)
        node_instances = []
        map_input_uuids = {}
        map_output_uuids = {}
        
        for name, node in setting_dict.items():
            node_instance, map_input_uuid, map_output_uuid = \
                    Node.init_factory(
                        self, 
                        node['label'], 
                        node['input_attributes'], 
                        node['output_attributes'], 
                        node['params'], 
                        node['node_params'])
            node_instances.append(node_instance)
            map_input_uuids.update(map_input_uuid)
            map_output_uuids.update(map_output_uuid)
       
        for node_instance in node_instances:
            [attr.convert_output_attr(map_output_uuids)
                for attr 
                in node_instance._input_attributes]
            [attr.convert_input_attrs(map_input_uuids)
                for attr 
                in node_instance._output_attributes]
        LinkNode.link_by_children(node_instances[0], self.uuid)
        
        
    def delete_selected_nodes(self) -> None:
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        for node_id in selected_nodes:
            node: Node = dpg.get_item_user_data(node_id)
            if not node._label == 'Train Params':
                self.delete_node(node)
    
    def delete_node(self, node: Node):
        self.__nodes.remove(node)
        node._del()

    @staticmethod
    def delete_in_editor(editor_uuid:int, node:Node):
        editor: NodeEditor = dpg.get_item_user_data(dpg.get_item_parent(editor_uuid))
        editor.delete_node(node)
        return editor
    
    @staticmethod
    def set_mouse_pos(node_editor):
        pos = dpg.get_mouse_pos(local=False)
        ref_node = dpg.get_item_children(node_editor, slot=1)[0]
        ref_screen_pos = dpg.get_item_rect_min(ref_node)
        ref_grid_pos = dpg.get_item_pos(ref_node)

        NODE_PADDING = (0, 0)

        pos[0] = pos[0] - (ref_screen_pos[0] - NODE_PADDING[0]) + ref_grid_pos[0]
        pos[1] = pos[1] - (ref_screen_pos[1] - NODE_PADDING[1]) + ref_grid_pos[1]

        NodeEditor.mouse_pos = pos

    def submit(self, parent):
        
        with dpg.child_window(width=-160, height=-25, 
                              parent=parent, 
                              user_data=self, 
                              drop_callback=lambda s, a, u: 
                                  dpg.get_item_user_data(s).on_drop(s, a, u)):
            with dpg.node_editor(callback=LinkNode._link_callback,
                                 delink_callback=LinkNode._delink_callback,
                                 tag=self.uuid, 
                                 width=-1, height=-1,
                                 minimap=True, minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                dpg.add_node(label='ref node', tag='ref_node', show=True, draggable=False)

                for node in self.__nodes:
                    node._submit(self.uuid)
                
                with dpg.handler_registry():
                    dpg.add_key_press_handler(dpg.mvKey_Delete,
                                            callback=self.delete_selected_nodes)
                    dpg.add_mouse_drag_handler(callback=lambda:NodeEditor.set_mouse_pos(self.uuid))

    
class DragSourceContainer(BaseGUI):

    def __init__(self, label: str, width: int = 150, height: int = -1):
        super().__init__()
        self._label = label
        self._width = width
        self._height = height
        self._children: list[DragSource] = []

    def add_drag_source(self, sources: tuple[DragSource]):
        for source in sources:
            self._children.append(source)

    def submit(self, parent):

        with dpg.child_window(tag=self.uuid, 
                              parent=parent, 
                              width=self._width, height=self._height, 
                              menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label, 
                             enabled=False)
            filter_uuid = dpg.generate_uuid()
            
            dpg.add_input_text(hint="Поиск", callback=lambda s, a: dpg.set_value(filter_uuid, a), width=-1)
            with dpg.filter_set(tag=filter_uuid):
                for child in self._children:
                    child._submit(filter_uuid)


class Node(BaseGUI):

    debug=True

    def __init__(self, label: str, data=None, uuid:Optional[int]=None,
                 params:list[ParamNode]=[], 
                 **node_params):
        super().__init__(uuid)
        self._label: str = label
        self._data = data
        self._input_attributes: list[InputNodeAttribute] = []
        self._output_attributes: list[OutputNodeAttribute] = []
        self._params: list[ParamNode] = deepcopy(params)
        self._node_params = node_params
        
    @staticmethod    
    def init_factory(editor, 
                     label: str, 
                     input_attributes: list[dict], 
                     output_attributes: list[dict], 
                     params: list[dict], 
                     node_params: dict) -> Node:
        
        params_instances: list[ParamNode] = \
            [ParamNode.factory(
                attr['label'], 
                attr['type'], 
                attr['params']) 
             for attr 
             in params]
        node = Node(label, 
                    dicts.modules[label].func, 
                    params=params_instances, 
                    **node_params)
        map_input_uuids = {attr['uuid']: node._add_input_attribute(
                                        InputNodeAttribute.factory(
                                            attr['label'], 
                                            attr['uuid'], 
                                            attr['linked_out_attr'])
                                        ).uuid
                            for attr 
                            in input_attributes}
        map_output_uuids = {attr['uuid']: node._add_output_attribute(
                                        OutputNodeAttribute.factory(
                                            attr['label'], 
                                            attr['uuid'], 
                                            attr['children'])
                                        ).uuid
                            for attr 
                            in output_attributes}
        editor.add_node(node)
        node._submit(editor.uuid)
        return node, map_input_uuids, map_output_uuids

    def _finish(self):
        dpg.bind_item_theme(self.uuid, _completion_theme)

    def _add_input_attribute(self, attribute: InputNodeAttribute):
        self._input_attributes.append(attribute)
        return attribute

    def _add_output_attribute(self, attribute: OutputNodeAttribute):
        self._output_attributes.append(attribute)
        return attribute

    def _add_params(self, params: list[dict[str, Any]]):
        if params:
            self._params += [ParamNode(**param) for param in params] 

    def _delinks(self):
        for input in self._input_attributes:
            if out := input._linked_out_attr:
                out.remove_child(input)
        for out in self._output_attributes:
            (input.reset_linked_attr() for input in out._children if input)

    def _del(self):
        self._delinks()
        dpg.delete_item(self.uuid)
        del self
        

    def _submit(self, parent):
        self._node_editor_uuid = parent
        with dpg.node(tag=self.uuid,
                      parent=parent, 
                      label=self._label, 
                      user_data=self, 
                      **self._node_params):
            for attribute in self._input_attributes:
                attribute._submit(self.uuid)

            for attribute in self._params:
                attribute._submit(self.uuid)
            
            for attribute in self._output_attributes:
                attribute._submit(self.uuid)
           
        dpg.bind_item_handler_registry(self.uuid, hover_handler)
        self._finish()
        
        
    def get_dict(self) -> dict:
        return dict(
            uuid=self.uuid,
            label=self._label, 
            input_attributes=[attr.get_dict() 
                              for attr 
                              in self._input_attributes],
            output_attributes=[attr.get_dict() 
                               for attr 
                               in self._output_attributes],
            params=[param.get_dict() 
                    for param 
                    in self._params],
            node_params=self._node_params,
        )
    
    def next(self, out_idx: int=0, node_idx: int=0) -> Union[Node, None]:
        input_attrs: list[InputNodeAttribute] = \
            self._output_attributes[out_idx]._children
        nodes = [input_attr.get_node() for input_attr in input_attrs] if input_attrs else None
        if nodes and len(nodes) == 1:
            nodes = nodes[0]
        return nodes
        

    def init_with_params(self, mode='simple', params: Optional[dict]=None) -> Any:
        try: 
            params = params if params else self.get_params()
            print("\nСлой: ", self._data) if Node.debug else None
            print("Параметры слоя: ", params) if Node.debug else None
            match mode:
                case 'simple': return self._data(**params)
                case 'data': return DataModule(self._data, **params)
                case 'multi_branch': return MultiBranch(self._data, **params)
        except: raise RuntimeError("Ошибка инициализации слоя")
        
        
    def get_params(self) -> dict:
        params = {}
        for param in self._params:
            returned = param.get_value()
            params.update(returned) if returned else None
        return params


class TrainParamsNode(Node):
    
    WIDTH=150
    __params: dict[str, dict[str, Any]] = dict(
        Loss= 
            {"MSE": nn.MSELoss, 
            "Cross Entropy": F.cross_entropy, 
            "L1": nn.L1Loss},
        Optimizer=
            {"SGD":torch.optim.SGD, 
            "Adam":torch.optim.Adam, 
            "RMSprop":torch.optim.RMSprop,
            "Adagrad":torch.optim.Adagrad
            },
        Initialization=
            {'Default':None, 
            'Normal': init_normal, 
            'Xavier': init_xavier},
    )


    @staticmethod
    def factory(name, data_node: DataNode, default_params: Optional[dict[str, str]]=None, **node_params):
        node = TrainParamsNode(name, data_node, default_params, **node_params)
        return node

    def __init__(self, label: str, data_node: DataNode, default_params: Optional[dict[str, str]]=None, **node_params):
        super().__init__(label, data_node, **node_params)

        self._add_input_attribute(InputNodeAttribute("train dataset"))
        if not default_params: 
            default_params = {key:list(choices.keys())[0] for key, choices in TrainParamsNode.__params.items()}
        
        def get_default(value):
            return default_params.get(value, list(TrainParamsNode.__params[value].keys())[0])
        
        train_params: list[dict[str, object]] = [
            {"label":"Название проекта", "type":'text', "default_value":default_params.get('Название задачи', data_node._label), "width":TrainParamsNode.WIDTH},
            {"label":"Название задачи", "type":'text', "default_value":default_params.get('Название проекта','DLC'), "width":TrainParamsNode.WIDTH},
            {
                "label":"Функция потерь", "type":'combo', "default_value":get_default('Loss'), 'tooltip':"""
### Функция потерь
___
+ MSE - критерий, который измеряет среднеквадратичную ошибку (квадрат нормы L2) между каждым элементом во входном x и целевом y
+ Cross Entropy - критерий вычисляет потерю перекрестной энтропии между входными логитами и таргетом. Используется для задачи классификации
+ L1 - критерий измеряет среднюю абсолютную ошибку (MAE) между каждым элементом во входном x и целевом y
                """,
                "items":tuple(TrainParamsNode.__params['Loss'].keys()), "user_data":TrainParamsNode.__params['Loss'], "width":TrainParamsNode.WIDTH
            },
            {
                "label":"Optimizer", "type":'combo', "default_value":get_default('Optimizer'), 'tooltip':"""
### Оптимизатор
+ SGD - стохастический градиентный спуск - итерационный метод оптимизации целевой функции с подходящими свойствами гладкости (например, дифференцируемость или субдифференцируемость)
+ Adam - сочетает в себе идеи RMSProp и оптимизатора импульса
+ RMSprop - среднеквадратичное распространение корня - это экспоненциально затухающее среднее значение. RMSprop вносит свой вклад в экспоненциально затухающее среднее значение прошлых «квадратичных градиентов»
+ Adagrad (алгоритм адаптивного градиента) - регулирует скорость обучения для каждого параметра на основе его предыдущих градиентов
                """,
                "items":tuple(TrainParamsNode.__params['Optimizer'].keys()), "user_data":TrainParamsNode.__params['Optimizer'], "width":TrainParamsNode.WIDTH
            },
            {
                "label":"Initialization", "type":'combo', "default_value":get_default('Initialization'), 'tooltip':"""
### Инициализация параметров
+ Normal - из нормального распределения (среднее=0, отклонение=0.01)
+ Xavier - из Ксавье распределения - подойдет для симметричных относительно нуля функций активации (например, Tanh), оставляет дисперсию весов одинаковой
                """,
                "items":tuple(TrainParamsNode.__params['Initialization'].keys()), "user_data":TrainParamsNode.__params['Initialization'], "width":TrainParamsNode.WIDTH
            },
            {"label":"Скорость обучения", "type":'float', "default_value":default_params.get('Скорость обучения', 0.05), "width":TrainParamsNode.WIDTH},
            {"label":"Эпохи", "type":'int', "default_value":default_params.get('Эпохи', 2), "width":TrainParamsNode.WIDTH},
            {"label":"Сохранить веса", "type":"file", "callback":Pipeline.save_weight},
            {"label":"Загрузить веса", "type":"file", "callback":Pipeline.load_weight},
            {"label":"Дообучить", "type":"button", "callback":Pipeline.keep_train, "user_data":data_node},
            {"label":"Прервать", "type":"button", "callback":Pipeline.terminate, "user_data":data_node},
            {"label":"Запустить обучение", "type":"button", "callback":Pipeline.flow, "user_data":data_node},
        ]
        self._add_params(train_params)
    
    
    
    def set_pipline(self, pipeline: Pipeline):
        self.pipeline = pipeline
        
    def set_datanode(self, datanode: DataNode):
        self.datanode = datanode
           
  
class DragSource():

    def __init__(self, label: str, data=None, 
                 **node_params):
        
        self._label = label
        func = modules[label].func
        
        self._data =  func if func else data
        self._generator = modules[label].generator
        self._params = modules[label].params
        self._default_params = modules[label].default_params
        self._popup: str = modules[label].popup
        self._tooltip: str = modules[label].tooltip
        self._details: str = modules[label].details
        self._image: str = modules[label].image
        self._node_params = node_params
        

    def _submit(self, parent: DragSourceContainer):
        button = dpg.add_button(label=self._label, 
                        parent=parent, 
                        width=-1,
                        user_data=self,
                        filter_key=self._label)
       
        dpg.bind_item_handler_registry(button, hover_handler)
        dpg.bind_item_theme(button, _source_theme)
        with dpg.drag_payload(parent=button, 
                              drag_data=(self._label, 
                                         self._generator, 
                                         self._data, 
                                         self._params, 
                                         self._default_params, 
                                         self._node_params)):
            dpg.add_text(f"Name: {self._label}")

      
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
        self._add_output_attribute(OutputNodeAttribute("train params"))
        if params:
            try:
                df = data(
                        root=f"{os.getcwd()}/datasets/", 
                        download=True)
                
                params[0]['items'][0]['default_value'] = df[0][0].size
                          
            except: pass
            self._add_params(params)
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
                                (self._output_attributes[1].uuid, 
                                 self.train_params._input_attributes[0].uuid))


class LayerNode(Node):

    @staticmethod
    def factory(name, data, params:Optional[tuple[dict]]=None, default_params: Optional[dict[str, str]]=None, **node_params):
        node = LayerNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, 
                 label: str, 
                 data, 
                 params:Optional[tuple[dict]]=None, 
                 default_params: Optional[dict[str,str]]=None, 
                 **node_params):
        if not default_params or not default_params.get('part_of_module', None):
            node_params['pos'] = NodeEditor.mouse_pos
        super().__init__(label, data, **node_params)
        self._add_input_attribute(InputNodeAttribute("data"))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        self._add_params(params)


class ModuleNode(Node):

    @staticmethod
    def factory(label, 
                sequential: tuple[tuple[DragSource, dict]], 
                params: Optional[list[dict[str, Any]]]=None, 
                default_params: Optional[dict[str,str]]=None, 
                **node_params):
        node = ModuleNode(label, sequential, params, default_params, **node_params)
        return node

    def __init__(self, 
                 label, 
                 sequential: tuple[tuple[DragSource, dict]], 
                 params: Optional[list[dict[str, Any]]]=None, 
                 default_params: Optional[dict[str,str]]=None, 
                 **node_params):
        if node_params:
            node_params['pos'] = node_params.get('pos', [NodeEditor.mouse_pos[0], NodeEditor.mouse_pos[1]])
            if (x := node_params['pos'].get('x', None)) and (y := node_params['pos'].get('y', None)):
                node_params['pos'] = [x,y]
        else: node_params = {'pos':[NodeEditor.mouse_pos[0], NodeEditor.mouse_pos[1]]}
        super().__init__(label, None, **node_params)
        self._add_input_attribute(InputNodeAttribute("data"))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        expand = [{"label":"Развернуть", "type":"button", "callback":self.expand},
                  {"label":"Редактировать", "type":"button", "callback":self.edit}]
        if not params: params = expand
        else: params += expand
        
        self._add_params(params)
        self.sequential = sequential
        self.pos: dict[str, int] = {"x":node_params['pos'][0], "y":node_params['pos'][1]}


    @staticmethod
    def _replace(module, new_defaults, idx):
        if len(module) > 1: # если указаны новые параметры по умолчанию для слоя
            for key in module[1].keys():
                if params := new_defaults.get(key):
                    module[1][key] = params[idx]
        

    @staticmethod
    def replace_default_params(sequential: tuple[tuple[object, dict]], 
                               new_defaults: dict):
        updated_sequential = deepcopy(sequential)
        for i, layer_defaults in enumerate(updated_sequential):
            if isinstance(layer_defaults[0], tuple):
                for idx_module,  module in enumerate(layer_defaults):
                    ModuleNode._replace(module, new_defaults[i], idx_module)
            else:
                ModuleNode._replace(layer_defaults, new_defaults, i)
            
        return updated_sequential


    def _submit_in_editor(self, editor, parent=None):
       
        if not hasattr(self, '_node_editor_uuid') and (parent==None):
            raise RuntimeError("id node_editor не определен")
        if not hasattr(self, '_node_editor_uuid'): self._node_editor_uuid = parent
        
        prev_node = None
        lasts_in_branch = []
        for node in self._data:
            if isinstance(node, list):
                flag = False
                for branch in node:
                    if isinstance(branch, list):
                        for module in branch:
                            editor.add_node(module)
                            module._submit(self._node_editor_uuid)
                        lasts_in_branch.append(branch[-1])
                        if prev_node: branch.insert(0, prev_node)
                        LinkNode.link_nodes(branch, self._node_editor_uuid) 
                    else:
                        flag = True
                        editor.add_node(branch)
                        branch._submit(self._node_editor_uuid)
                if flag:
                    lasts_in_branch.append(node[-1])
                    if prev_node: node.insert(0, prev_node)
                    LinkNode.link_nodes(node, self._node_editor_uuid)
            else:
                editor.add_node(node)
                node._submit(self._node_editor_uuid)
                if len(lasts_in_branch):
                    for last in lasts_in_branch:
                        LinkNode._link_nodes(last, node, self._node_editor_uuid)
                    lasts_in_branch = []
                elif prev_node: LinkNode._link_nodes(prev_node, node, self._node_editor_uuid)
                prev_node = node
       
     
    def expand(self) -> None:
        editor = NodeEditor.delete_in_editor(self._node_editor_uuid, self)
        self._submit_in_editor(editor)
            
    def edit(self) -> None:
        NodeEditor.delete_in_editor(self._node_editor_uuid, self)
        dpg.configure_item(self.modal, show=True)
   
    def submit_module(self, sequential, modal):
        all_sources = []
        all_params = []
        for idx, node in enumerate(sequential):
            source: DragSource = deepcopy(node[0])
            defaults = None
            if len(node)>1:
                defaults = deepcopy(node[1])
                if source._generator.__qualname__ == 'ModuleNode.factory':
                    source._data = ModuleNode.replace_default_params(
                                            source._data, 
                                            defaults)
                    with dpg.collapsing_header(parent=modal, 
                                               label=source._label, 
                                               default_open=False) as submodule:
                        sources, self.pos, source_params = \
                            ModuleNode(source._label, 
                                       source._data,
                                       pos={'x':NodeEditor.mouse_pos[0], "y":self.pos['y'] + 150})\
                                           .submit_sequential(submodule)
                    all_sources.append(sources)
                    all_params.append(source_params)
                    continue
            all_params.append(
                ParamNode.submit_config(source._label, 
                                        source._params, 
                                        defaults, 
                                        modal))
            params = {"pos":(self.pos['x'], self.pos['y'])}
            self.pos['x'] += 200
            all_sources.append(
                deepcopy((source._label, 
                          source._generator, 
                          source._data, 
                          source._params, 
                          source._default_params, 
                          params))
                )
        return all_sources, self.pos, all_params

    def submit_sequential(self, modal: int) -> tuple[list, dict, list]:
        if not isinstance(self.sequential[0][0], tuple):
            return self.submit_module(self.sequential, modal)
        
        all_sources = []
        all_params = []

        is_branch = len(self.sequential) > 1

        for idx, branch in enumerate(self.sequential, 1):
            if is_branch:
                modal = dpg.add_collapsing_header(label=f'Ветка #{idx}', default_open=False)
                self.pos['x'] = NodeEditor.mouse_pos[0]
                self.pos['y'] += 200
                
            sources, _, source_params = self.submit_module(branch, modal)
            all_sources.append(sources)
            all_params.append(source_params)
        return all_sources, self.pos, all_params
    
    @staticmethod
    def _collect(sources, params):
        if params:
            for param, origin_param in zip(params, sources[3]):
                returned = param.get_value()
                if returned and ((value := returned.get(origin_param['label'], None)) is not None):
                    origin_param['default_value'] = value   

    def recursive_collect(self, sources, params, editor):
        if isinstance(sources, list) and isinstance(params, list):
            multi_source = []
            for source, param in zip(sources, params):
                multi_source.append(self.recursive_collect(source, param, editor))
            return multi_source
        ModuleNode._collect(sources, params)
        return editor.on_drop(None, sources, None, module=True)

    def _collect_submit(self, all_sources, all_params, parent, modal, collapse_checkbox):
        editor: NodeEditor = dpg.get_item_user_data(dpg.get_item_parent(parent))
        self._data = self.recursive_collect(all_sources, all_params, editor)
        
        if dpg.get_value(collapse_checkbox):
            super()._submit(parent)
            editor.add_node(self)
        else:
            self._submit_in_editor(editor, parent)
        dpg.configure_item(modal, show=False)
        

    def _submit(self, parent):
        with dpg.window(modal=True, pos=(500,0), label=f'Конфигурирование модуля {self._label}', min_size=(300, 500)) as modal:
            self.modal = modal
            with dpg.child_window(height=-50) as child_window:
                all_sources, _, all_params = self.submit_sequential(child_window)
            dpg.add_separator()
            collapse_checkbox = dpg.add_checkbox(label="Свернуть модель", default_value=True)
            dpg.add_button(label="Продолжить", callback=lambda:self._collect_submit(all_sources, all_params, parent, modal, collapse_checkbox))
        

class MultiBranch(nn.Module):

    def __init__(self, multi_branch, **kwargs):
        super(MultiBranch, self).__init__()
        self.multi_branch = multi_branch
        for key, value in kwargs.items():
            setattr(self, key, value)


    @staticmethod
    def sub_forward(branch, x):
        if isinstance(branch, list):
            res = MultiBranch.sub_forward(branch[0], x)
            for module in branch[1:]:
                res = MultiBranch.sub_forward(module, res)
            return res
        else:
            return branch(x)

    def forward(self, x):
        branchs = []
        for branch in self.multi_branch:
            branchs.append(MultiBranch.sub_forward(branch, x))
        return torch.cat(branchs, dim=self.dim)


class DataModule(LightningDataModule):
    def __init__(self, dataset_class, **kwargs):
        super().__init__()
        args: tuple = dataset_class.__init__.__code__.co_varnames
        has_train_arg = 'train' in args 
        has_split_arg = 'split' in args

        params = dict(root=f"{os.getcwd()}/datasets/", 
                      download=True)
        
        name_transform = next((key for key in ['transform', 'transforms'] if key in args), None)
        if name_transform:
            transform = kwargs.get('Transform', None)
            if transform: params.update({name_transform: transform})
        
        if has_train_arg: params.update({'train': True})
        elif has_split_arg: params.update({'split':'train'})
        self.train = dataset_class(**params)
        
        if has_train_arg: params.update({'train': False})
        elif has_split_arg: params.update({'split':'val'})
        self.val = dataset_class(**params)
        
        if transform and (resize := re.search('(?<=Resize\(size=)(\[.*\])', transform.extra_repr())):
            self.shape = eval(resize[0])
        else: self.shape = self.train[0][0].size
    
        for attr, value in kwargs.items():
            setattr(self, attr, value)
            
        assert hasattr(self, 'batch_size')
        
    
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=7)


class LModule(LightningModule):
    
    debug=True
    accept_init = [nn.Linear, nn.Conv2d]
    
    def __init__(self, sequential, optimizer, lr, loss_func):
        super(LModule, self).__init__()
        self.net = sequential
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func
        print('\n\t\t\tНеинициализированная сеть: ', self.net) if LModule.debug else None
        

    def apply_init(self, data_source, init):
        inputs = [next(iter(data_source.train_dataloader()))[0]]
    
        if init is not None:
            self.net.apply(init)
            self.forward(*inputs)
            self.net.apply(init)

    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr)
        return optim
    
    def forward(self, X):
        return self.net(X)
        

    def accuracy(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), float32)
        return reduce_mean(compare) if averaged else compare


    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return self.loss_func(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    
    
    def metric(self, batch, mode='train', averaged=True):
        Out, Y = self(*batch[:-1]), batch[-1]
        loss = self.loss(Out, Y)
        # Logging
        self.log_dict({f"{mode}_loss":loss, 
                       f"{mode}_acc":self.accuracy(Out, Y)}, 
                       prog_bar=True, 
                       on_epoch=True)
        return loss

    def training_step(self, batch):
        return self.metric(batch, 'train')

    def validation_step(self, batch):
        return self.metric(batch, 'val')
    
    
    def layer_summary(self, X_shape):
        send_message(f"{'Размер входного тензора':25}:\t{X_shape}", 'log', "") 
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            send_message(f"{layer.__class__.__name__:<45}:\t{X_shape}", 'log', "")


class CustomDataset:

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform = None,
        **params
    ) -> None:
        print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEY")

    @staticmethod
    def copy_dataset(sender, app_data, train_params__file: tuple[2]):
        train_params, filepath_uuid = train_params__file
        print(train_params__file)
        
        data_node = dpg.get_item_user_data(
                        dpg.get_item_parent(train_params._input_attributes[0]._linked_out_attr.uuid))  
            
        filepath = dpg.get_value(filepath_uuid)
        print(data_node, filepath)


class Pipeline:

    debug = True

    def __init__(self, datanode: DataNode):
        send_message("Инициализация пайплайна", 'log') 
        datanode.train_params.set_pipline(self)
        self.train_params = datanode.train_params.get_params()
        send_message(self.train_params, 'log', 'Тренировочные параметры') 
        send_message("Сбор слоёв", 'log') 
        self.dataset = datanode.init_with_params('data')
        self.pipeline = Pipeline.collect_layers(datanode)
        send_message("Инициализация сети", 'log') 
        try:
            self.net = LModule(sequential=nn.Sequential(*self.pipeline), optimizer=self.train_params['Optimizer'],
                              lr=self.train_params['Скорость обучения'], loss_func=self.train_params['Функция потерь'])
            self.net.apply_init(self.dataset, self.train_params['Initialization'])
            send_message(self.net, 'code', 'Инициализированная сеть') 
            self.net.layer_summary((self.dataset.batch_size,1,*self.dataset.shape)) if Pipeline.debug else None
        except (BaseException, RuntimeError, TypeError) as err: 
            send_message(err, 'error', "Возникла ошибка во время инициализации сети")
            raise err
        self.task: clearml.Task = clearml.Task.init(
                    project_name=self.train_params["Название проекта"],
                    task_name=self.train_params["Название задачи"],
                    continue_last_task=True)
        self.task.connect(self.train_params)
        send_message(f'[ClearML]({self.task.get_output_log_web_page()})', 
                     'log', 'Запуск ClearML') 

    
    @staticmethod
    def flow(sender=None, app_data=None, datanode: Optional[DataNode]=None):
        assert datanode
        Pipeline(datanode).train()
            
        
    @staticmethod
    def keep_train(sender, app_data, datanode: DataNode):
        if hasattr(datanode.train_params, "pipeline"):
            try: 
                self = datanode.train_params.pipeline
                self.train_params = datanode.train_params.get_params()
                self.net.lr = self.train_params['Скорость обучения']
                self.train()
            except BaseException as err:
                send_message(err, 'error', "Возникла ошибка во время дообучения")
                raise err
        else:
            Pipeline.flow(datanode=datanode)
            
    @staticmethod
    def terminate(sender, app_data, datanode: DataNode):
        try:
            if hasattr(datanode.train_params, "pipeline"):
                self = datanode.train_params.pipeline
                if hasattr(self, "thread"):
                    terminate_thread(self.thread)
                    self.task.close()
        except (BaseException, ValueError) as warn:
            send_message(warn, 'warning')
            raise warn
        
        
    @staticmethod
    def save_weight(sender, app_data, train_params__file):
        train_params, filepath_uuid = train_params__file
        self = train_params.pipeline
        if self:
            filepath = dpg.get_value(filepath_uuid)
            torch.save(self.net.state_dict(), filepath)

    @staticmethod
    def load_weight(sender, app_data, train_params__file: tuple[TrainParamsNode, int]):
        train_params, filepath_uuid = train_params__file
        self: Pipeline = train_params.pipeline
        if not self:
            datanode = train_params.datanode
            self = Pipeline(datanode=datanode)
        filepath = dpg.get_value(filepath_uuid)
        try: self.net.load_state_dict(torch.load(filepath))
        except BaseException as err:
            send_message(err, 'error', "Файл параметров не найден")
            raise err
        
    @staticmethod
    def collect_multi_branch(next_node):
        multi_branch = []
        for branch in next_node:
            collected = Pipeline.collect_layers(branch, True)
            multi_branch.append(collected[:-1])
            node = collected[-1]
        return Node('Multi-branch', multi_branch)\
                        .init_with_params('multi_branch', node.get_params()), \
                node

    @staticmethod
    def convert_to_data(module_node):
        if isinstance(module_node, list):
            new_data_list = []
            for branch in module_node:
                new_data_list.append(Pipeline.convert_to_data(branch))
            return new_data_list
        else: return module_node.init_with_params()


    @staticmethod
    def collect_layers(node: Node, reckon_in = False):
        pipeline = [node.init_with_params()] if reckon_in else []
        while next_node := node.next():
            if isinstance(next_node, list): 
                multi_batch, node = Pipeline.collect_multi_branch(next_node)
                pipeline.append(multi_batch)
                continue
            elif isinstance(next_node._data, list):
                module_index = 0
                modules = next_node._data
                
                while module_index < len(modules):
                    module_node = modules[module_index]
                   
                    if isinstance(module_node, list):
                        if len(module_node) == 1:
                            module_node = module_node[0] 
                            for sub_node in module_node:
                                pipeline.append(sub_node.init_with_params())
                        else:
                            module_node = Pipeline.convert_to_data(module_node)
                            multi_batch = Node('Multi-branch', module_node)\
                                        .init_with_params('multi_branch', 
                                                        modules[module_index+1].get_params())

                            pipeline.append(multi_batch)
                            module_index += 1
                    else:  
                        pipeline.append(module_node.init_with_params())
                    module_index += 1
            elif next_node._data.__name__ == 'cat':
                pipeline.append(next_node)
                return pipeline
            else:
                pipeline.append(next_node.init_with_params())
            
            node = next_node
        return pipeline
        
    def train(self):
        try: self.trainer = Trainer(max_epochs=self.train_params['Эпохи'], accelerator='auto', # 'gpu' if torch.cuda.is_available() else 'cpu' 
                                    log_every_n_steps=len(self.dataset.train) / self.dataset.batch_size) 
        except (BaseException, RuntimeError) as err: 
            send_message(err, 'error', "Ошибка инициализации тренировочного класса")
            raise err
        try:
            self.thread = threading.Thread(target=self.trainer.fit, args=(),name='train',
                                           kwargs=dict(model=self.net, datamodule=self.dataset))
            self.thread.start() 
        except (BaseException, RuntimeError) as err: 
            terminate_thread(self.thread)
            send_message(err, 'error', "Возникла ошибка во время обучения модели")
            raise err
#endregion


#region dicts
Module: NamedTuple = namedtuple('Module', ['generator', 'func', 'params', 'default_params','popup','details','image', 'tooltip'], 
                                defaults=(None, None, None, None, None, None, None, None))
modules: DefaultDict[str, NamedTuple] = defaultdict(lambda: Module("Not present"))

_dtypes = {'float32': v2.ToDtype(torch.float32, scale=True), 
                  'int32': v2.ToDtype(torch.int32), 
                  'int64': v2.ToDtype(torch.int64)} 
transforms_img = [
    {"label":"Resize", "type":'text/tuple', "default_value":"[224, 224]", "user_data": v2.Resize},
    {"label":"ToImage", "type":'blank', "user_data": v2.ToImage},
    {"label":"ToDtype", "type":'combo', "default_value":"float32",
        "items":tuple(_dtypes.keys()), "user_data": _dtypes},
    {"label":"AutoAugment", "type":'blank', "user_data": v2.AutoAugment },
    {"label":"RandomIoUCrop", "type":'blank', "user_data": v2.RandomIoUCrop},
    {"label":"ElasticTransform", "type":'blank', "user_data": v2.ElasticTransform},
    {"label":"Grayscale", "type":'blank', "user_data": v2.Grayscale},
    # {"label":"RandomCrop", "type":'blank', "user_data": v2.RandomCrop},
    {"label":"RandomVerticalFlip", "type":'blank', "user_data": v2.RandomVerticalFlip},
    {"label":"RandomHorizontalFlip", "type":'blank', "user_data": v2.RandomHorizontalFlip}
    ] 
transforms_setting_img = {"label":"Transform", "type":'collaps', "items":transforms_img}


params = {
    "img_transforms": transforms_setting_img,
    "batch_size" :   {"label":"batch_size", "type":'int', "default_value":64, "tooltip": "__Батчи/Пакеты/сеты/партии__ - это набор объектов тренировочного датасета, который пропускается итеративно через сеть во время обучения\n___\n1 < batch_size < full_size"},
    "val_size":     {"label":"val_size", "type":'float', 
                     "max_value": 0.9999999, "max_clamped":True, "default_value":0.2},
    "button_load":  {"label":"Load Dataset", "type":"path", "default_value":"/home/luzinsan/Environments/petrol/data/"},
    "default_train":{'Loss':'L1 Loss','Optimizer':'SGD'},
    "out_features": {"label":"out_features", "type":'int', "default_value":1, "tooltip":"Количество признаков на выходе линейной трансформации.\nКоличество признаков на входе определяется автоматически"},
    "out_channels": {"label":"out_channels", "type":'int', "default_value":6, 
                     "tooltip":"Количество выходных каналов/признаковых карт, которые являются репрезентациями для последующих слоёв (рецептивное поле)"
                     },
    "num_features": {"label":"num_features", "type":'int', "default_value":6},
    "output_size":{"label":"output_size", "type":'text/tuple', "default_value":'[1, 2]', 'tooltip':"Целевой выходной размер изображения формы HxW. Может быть списком [H, W] или одним H (для квадратного изображения HxH).\nH и W могут быть либо int , либо None. None означает, что размер будет таким же, как и у входных данных."},
    "kernel_size":{"label":"kernel_size", "type":'int', "default_value":5, 
                   'tooltip':"Размер тензорного ядра"
                   },
    "stride":{"label":"stride", "type":'int', "default_value":1, 
              'tooltip': "Шаг прохождения тензорного ядра во время свёртки (взаимной корреляции)"
              },
    "padding":{"label":"padding", "type":'int', "default_value":0, 
               'tooltip':"Размер заполнения краёв входной матрицы"
               },
    "eps":          {"label":"eps", "type":'float', "default_value":1e-5},
    "momentum":{"label":"momentum", "type":'float', "default_value":0.1},
    "affine":{"label":"affine", "type":'bool', "default_value": True},
    "p":{"label":"p", "type":'float', "default_value":0.5, 'tooltip':"Вероятность обнуления элемента"},
    "dim":{"label":"dim", "type":'int', "default_value":1, 'tooltip':"Рассматриваемое измерение"},
}

default_params = {
    'Loss':'Cross Entropy',
    'Optimizer':'SGD',
}
fromkeys = lambda d, keys: {x:d.get(x) for x in keys}
modules.update({
####################################################################~~~ DATASETS ~~~####################################################################
    "FashionMNIST":       Module(DataNode.factory, ds.FashionMNIST, 
                                 (params['img_transforms'], params['batch_size']), 
                                 fromkeys(default_params, ['Loss', 'Optimizer']), image='./static/images/fashion-mnist.png', popup="""
### Fashion-MNIST
___
+ это набор изображений, предоставленный [Zalando](https://arxiv.org/pdf/1708.07747)
+ состоит из обучающего набора из 60 000 примеров и тестового набора из 10 000 примеров. 
+ Каждый пример представляет собой изображение в оттенках серого размером 28x28
+ Представлено 10 классов
_На изображении каждый класс представлен в трёх строках_
                            """),
    "Caltech101":       Module(DataNode.factory, ds.Caltech101, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Caltech256":       Module(DataNode.factory, ds.Caltech256, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CarlaStereo":       Module(DataNode.factory, ds.CarlaStereo, 
                                (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CelebA":       Module(DataNode.factory, ds.CelebA, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CIFAR10":       Module(DataNode.factory, ds.CIFAR10, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Cityscapes":       Module(DataNode.factory, ds.Cityscapes, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CLEVRClassification":       Module(DataNode.factory, ds.CLEVRClassification, 
                                        (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "EMNIST":       Module(DataNode.factory, ds.EMNIST, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CocoCaptions":       Module(DataNode.factory, ds.CocoCaptions, 
                                 (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "EuroSAT":       Module(DataNode.factory, ds.EuroSAT, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Flowers102":       Module(DataNode.factory, ds.Flowers102, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Food101":       Module(DataNode.factory, ds.Food101, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "ImageNet":       Module(DataNode.factory, ds.ImageNet, 
                             (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "SUN397":       Module(DataNode.factory, ds.SUN397, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Dataset from File":       Module(DataNode.factory, CustomDataset, 
                                      (params['val_size'], params['button_load']), fromkeys(default_params, ['Loss', 'Optimizer'])),

######################################################################~~~ LINEARS ~~~#######################################################################
    "LazyLinear":       Module(LayerNode.factory, nn.LazyLinear, (params['out_features'],), image="./static/images/linear_layer.png", popup=
                               """
Линейный слой 
___
_Другие названия: полносвязный или плотный (Dense) слой_
+ это линейное преобразование над входящими данными (его обучаемые параметры - это матрица _W_ и вектор _b_). Такой слой преобразует _d_-размерные векторы в _k_-размерные
                               """),
    "LazyConv1d":       Module(LayerNode.factory, nn.LazyConv1d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding']), tooltip="Применяет одномерную свертку к входному сигналу, состоящему из нескольких входных плоскостей"),
    "LazyConv2d":       Module(LayerNode.factory, nn.LazyConv2d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding']), tooltip="Применяет 2D-свертку к входному сигналу, состоящему из нескольких входных плоскостей"),
    "LazyConv3d":       Module(LayerNode.factory, nn.LazyConv3d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding']), tooltip="Применяет 3D-свертку к входному сигналу, состоящему из нескольких входных плоскостей"),
    "LazyBatchNorm1d":  Module(LayerNode.factory, nn.LazyBatchNorm1d, (params['eps'], params['momentum'], params['affine']), tooltip='_Рекомендуемый размер пакета (в гиперпараметрах обучения) = 50-100_'),
    "LazyBatchNorm2d":  Module(LayerNode.factory, nn.LazyBatchNorm2d, (params['eps'], params['momentum'], params['affine'])),
    "LazyBatchNorm3d":  Module(LayerNode.factory, nn.LazyBatchNorm3d, (params['eps'], params['momentum'], params['affine'])),
    "Flatten":          Module(LayerNode.factory, nn.Flatten),
    "Concatenate":      Module(LayerNode.factory, torch.cat, (params['dim'], )),
    "AvgPool2d":        Module(LayerNode.factory, nn.AvgPool2d, (params['kernel_size'], params['stride'], params['padding'])),
    "MaxPool2d":        Module(LayerNode.factory, nn.MaxPool2d, (params['kernel_size'], params['stride'], params['padding'])),
    "AdaptiveAvgPool2d":Module(LayerNode.factory, nn.AdaptiveAvgPool2d, (params['output_size'], ), tooltip="Применяет двумерное адаптивное усреднение к входному сигналу, состоящему из нескольких входных плоскостей"),
    "Dropout":          Module(LayerNode.factory, nn.Dropout, (params['p'], ), tooltip="__Метод регуляризации и предотвращения совместной адаптации нейронов__\nВо время обучения случайным образом обнуляет некоторые элементы входного тензора.\nЭлементы выбираются независимо во время каждого прямого прохода (feed-forward) из распределения Бернулли. "),

#####################################################################~~~ ACTIVATIONS ~~~#####################################################################
    "ReLU":             Module(LayerNode.factory, nn.ReLU, image='./static/images/ReLU.png', details="__Функция активации__ (activation function) - нелинейное преобразование, поэлементно применяющееся к пришедшим на вход данным. Благодаря функциям активации нейронные сети способны порождать более информативные признаковые описания, преобразуя данные нелинейным образом.", popup=
                               """
### Rectified Linear Unit
___
+ Наиболее популярная функция активации из-за простоты реализации и хорошей производительности
+ Сохраняет только положительные значения, обнуляя все отрицательные
+ Кусочно-линейная функция
+ Решает проблему затухающего градиента
                               """),
    "Sigmoid":          Module(LayerNode.factory, nn.Sigmoid, image='./static/images/sigmoid.png', details="__Функция активации__ (activation function) - нелинейное преобразование, поэлементно применяющееся к пришедшим на вход данным. Благодаря функциям активации нейронные сети способны порождать более информативные признаковые описания, преобразуя данные нелинейным образом.", popup=
                               """
### Сигмоидная функция активации
___
+ Сжимает входные данные, преобразуя их в значения на интервале (0, 1) 
+ По этой причине сигмоиду часто называют сжимающей функцией: она сжимает любой вход в диапазоне (-inf, inf) до некоторого значения в диапазоне (0, 1)
+ Градиент функции обращается в нуль при больших положительных и отрицательных значениях аргументов, что является проблемой затухающего градиента
+ Полезна в рекуррентных сетях
                               """),
    "Tanh":             Module(LayerNode.factory, nn.Tanh, image='./static/images/tanh.png', details="__Функция активации__ (activation function) - нелинейное преобразование, поэлементно применяющееся к пришедшим на вход данным. Благодаря функциям активации нейронные сети способны порождать более информативные признаковые описания, преобразуя данные нелинейным образом.", popup=
                               """
### Гиперболический тангенс
___
+ Сжимает входные данные, преобразуя их в значения на интервале (-1, 1)
+ Является симметричной относительно начала координат
+ Производная функции в нуле принимает значени 1
+ Градиент функции обращается в нуль при больших положительных и отрицательных значениях аргументов, что является проблемой затухающего градиента
                               """),
    
####################################################################~~~ ARCHITECTURES ~~~####################################################################
    "LeNet5":           Module(ModuleNode.factory, image='./static/images/lenet.png', details="Базовыми единицами в каждом сверточном блоке являются сверточный слой, сигмоидальная функция активации и последующая операция объединения усреднений. Обратите внимание, что, хотя ReLU и max-pooling работают лучше, они еще не были обнаружены. Каждый сверточный слой использует ядро 5x5 и сигмоидальную функцию активации. Эти слои сопоставляют пространственно упорядоченные входные данные с несколькими двумерными картами объектов, обычно увеличивая количество каналов. Первый сверточный слой имеет 6 выходных каналов, а второй - 16. Каждая 2x2 операция объединения в пул (stride/шаг=2) уменьшает размерность в 4 раза, понижая пространственную дискретизацию. Сверточный блок выдает объект, размерностью (размер пакета, номер канала, высота, ширина).", popup=
                               """
### Свёрточная нейросеть
___
Одна из первых свёрточных сетей, заложившая основы глубокого обучения
+ Открыл: Ян Лекун [LeCun et al., 1989](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf), [LeCun et al., 1998b](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
+ В оригинальной версии используется average pooling и сигмоидные функции активации
+ Данный модуль модифицирован: max-pooling и ReLU функции активации
+ Применяется для классификации изображений (по-умолчанию настроен на 10 классов)
### Архитектура 
   Состоит из 2-х частей: 
+ Сверточные слои (kernel_size=5) с пуллинг слоями (kernel_size=2, stride=2)
+ 3 полносвязных слоя (out_features: 120 | 84 | 10 _(кол-во классов)_)
+ Без модификаций принимает изображения размером [28, 28]
                               """), 
    "AlexNet":          Module(ModuleNode.factory, image='./static/images/alexnet.png', details="Вплоть до 2012 года самой важной частью конвейера было репрезентативность, которая рассчитывалась в основном механически. К тому же, разработка нового набора признаков (feature engineering), улучшение результатов и описание метода - все это занимало видное место в статьях: SIFT [Lowe, 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), SURF [Bay et al., 2006](https://people.ee.ethz.ch/~surf/eccv06.pdf), HOG (гистограммы ориентированного градиента) [Dalal and Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), наборы визуальных слов [Sivic and Zisserman, 2003](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/sivic03.pdf) и аналогичные экстракторы признаков. На самых нижних уровнях сети модель изучает элементарные признаки, которые напоминали некоторые традиционные фильтры. Более высокие слои сети могут опираться на эти представления для представления более крупных структур, таких как глаза, носы, травинки и так далее. Еще более высокие слои могут представлять целые объекты, такие как люди, самолеты, собаки или фрисби. В итоге, конечное скрытое состояние изучает компактное представление изображения, которое суммирует его содержимое таким образом, что данные, принадлежащие к различным категориям, могут быть легко разделены.", popup=
                               """
### Глубокая сверточная нейросеть
___
Углубленная и расширенная версия LeNet, разработанная для конкурса/набора_данных ImageNet 
+ Открыл: Алекс Крижевский [Krizhevsky et al., 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf#page=5&zoom=310,88,777)
+ Используются функции активации ReLU в качестве нелинейностей.
+ Используется методика отбрасывания (Dropout) для выборочного игнорирования отдельных нейронов в ходе обучения, что позволяет избежать переобучения модели.
+ Перекрытие max pooling, что позволяет избежать эффектов усреднения (в случае average pooling)
+ Почти 10 раз больше карт признаков, чем у LeNet
### Архитектура 
   Состоит из 2-х частей: 
* Сверточные слои (kernel_size: 11 | 5 | 3 * 3) с max-pooling слоями (kernel_size=3, stride=2)
* 3 полносвязных слоя (out_features: 4096 | 84 | 10 _(кол-во классов)_) _(Осторожно: требуется почти 1 ГБ параметров модели)_
+ Без модификаций принимает изображения размером [224, 224]
                        """),
    "VGG-11":              Module(ModuleNode.factory, image='./static/images/vgg.png', details="VGG противоречит заложенным в LeNet принципам, согласно которым большие свёртки использовались для извлечения одинаковых свойств изображения. Вместо применяемых в AlexNet фильтров 9х9 и 11х11 стали применять гораздо более мелкие фильтры, опасно близкие к свёрткам 1х1, которых старались избежать авторы LeNet, по крайней мере в первых слоях сети. Но большим преимуществом VGG стала находка, что несколько свёрток 3х3, объединённых в последовательность, могут эмулировать более крупные рецептивные поля, например, 5х5 или 7х7. \nСети VGG для представления сложных признаков используют многочисленные свёрточные слои 3x3. \n__Примечательно__: в [VGG-E](https://arxiv.org/pdf/1409.1556#page=3&zoom=160,-97,717) в блоках 3, 4 и 5 для извлечения более сложных свойств и их комбинирования применяются последовательности 256×256 и 512×512 фильтров 3×3. Это равносильно большим свёрточным классификаторам 512х512 с тремя слоями! Это даёт нам огромное количество параметров и прекрасные способности к обучению. Но учить такие сети было сложно, приходилось разбивать их на более мелкие, добавляя слои один за другим. Причина заключалась в отсутствии эффективных способов регуляризации моделей или каких-то методов ограничения большого пространства поиска, которому способствует множество параметров.", popup=
                                  """
### Блочная свёрточная нейросеть
___
Противоречит принципам LeNet и AlexNet, однако заложила основы для архитектур Inception и ResNet
+ Открыла: группа исследователей из VGG, Оксфордский университет [Simonyan and Zisserman, 2014](https://arxiv.org/pdf/1409.1556#page=3&zoom=160,-97,717)
+ Состоит из повторяющихся структур - VGG блоков
### Архитектура:
Данная версия рассчитана на 10 классов и намного меньше VGG-11 по кол-ву параметров.
+ __5__ VGG блоков:
  - __(1 | 1 | 2 | 2 | 3)__ свёрточных слоя на каждый блок соответственно (out_channels: 16 | 32 | 64 | 128 | 128, kernel_size=3, padding=1) с ReLU функциями активаций
  - Max-pooling слой (kernel_size=2, stride=2)
+ 3 полносвязных слоя (out_features: 120 | 84 | 10 _(кол-во классов)_) со слоями отбрасывания (Dropout, p=0.5)
+ Без модификаций принимает изображения размером [224, 224]
                               """),
    "Conv-MLP":              Module(ModuleNode.factory, popup=
                                    """
### Многослойный перцептрон со свёрточными слоями
___
Является составляющим архитектуры NiN
### Архитектура:
+ 3 свёрточных слоя (out_channels: 96 | 96 | 96, kernel_size: 11 | 1 | 1, stride: 4 | 1 | 1, padding=0)
+ ReLU функции активации после каждого свёрточного слоя
                               """),
    "NiN":          Module(ModuleNode.factory, image='./static/images/NiN.png', details="NiN были предложены на основе очень простого понимания: \n1. использовать 1x1 свертки для добавления локальных нелинейностей по активациям каналов и\n2. использовать глобальный средний пул для интеграции по всем местоположениям в последнем слое представления. _Причём глобальное среднее объединение не было бы эффективным, если бы не дополнительные нелинейности_", popup=
                               """                    
### Архитектура NiN
___
Призвана решить проблемы VGG по части большого кол-ва параметров
+ Открыл: Мин Линь [Lin et al., 2013](https://arxiv.org/pdf/1312.4400)
+ Содержит модули Conv-MLP
+ MLP позволяют сильно повысить эффективность отдельных свёрточных слоёв посредством их комбинирования в более сложные группы.
+ Совершенно не использует полносвязные слои, что кратно уменьшает кол-во параметров, однако потенциально увеличивает время обучения
                                """),
     "Inception":          Module(ModuleNode.factory, image='./static/images/inception.png', popup='_Блок сети GoogLeNet_'),
                                 
     "GoogLeNet":          Module(ModuleNode.factory, image='./static/images/googlenet.png',
                                details="""
### Задания
GoogLeNet оказался настолько успешным, что прошел ряд итераций, постепенно улучшая скорость и точность. Попробуйте реализовать и запустить некоторые из них. Они включают в себя следующее:
+ Добавьте слой пакетной нормализации [Иоффе и Сегеди, 2015](https://arxiv.org/pdf/1502.03167)
+ Внесите изменения в блок Inception (ширина, выбор и порядок сверток), как описано у [Szegedy et al. (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
+ Используйте сглаживание меток для регуляризации модели, как описано в [Szegedy et al. (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
+ Внесите дальнейшие корректировки в блок Inception, добавив остаточное (residual) соединение [Szegedy et al., 2017](http://www.cs.cmu.edu/~jeanoh/16-785/papers/szegedy-aaai2017-inception-v4.pdf)
                                """,
                                  popup="""
### Многофилиальная сеть [GoogLeNet](https://arxiv.org/pdf/1409.4842)
___
Сочетает в себе сильные стороны NiN, вдохновлен LeNet и AlexNet
+ Довольно время- и ресурсозатратна
+ Здесь представлен упрощенный вариант. Оригинал включает ряд приёмов стабилизации обучения. В них больше нет необходимости из-за улучшенных алгоритмов обучения.
+ Базовый сверточный блок - Inception. Настраеваемые гиперпараметры - кол-во выходных каналов (output channels)
+ GoogLeNet использует 9 Inception блоков, сгруппированных на 3 группы (по 2, 5, 2 блока) 
+ Одна из первых сверточных сетей, в которой различаются основная часть (приём данных), тело (обработка данных) и голова (прогнозирование)
  - Основу задают 2-3 свертки, которые работают с изображением и извлекают низкоуровневые признаки
  - Телом является набор сверточных блоков
  - В голове сопоставляются полученные признаки с целевым признаком по задаче (классификации, сегментации, детекции или отслеживания)
                                """),
    "BN LeNet":         Module(ModuleNode)

})


#endregion



    
class App:

    def __init__(self):

        self.plugin_menu_id = dpg.generate_uuid()
        self.left_panel = dpg.generate_uuid()
        self.center_panel = dpg.generate_uuid()
        self.right_panel = dpg.generate_uuid()
        self.node_editor = NodeEditor()
    
        
        #region datasets
        datasets = {
            data: DragSource(data)
            for data
            in ["FashionMNIST",
                "Caltech101","Caltech256",
                "CIFAR10",
                "Flowers102",
                "SUN397",
                # доработать
                "CarlaStereo",
                "CelebA",
                "Cityscapes",
                "CLEVRClassification",
                "CocoCaptions",
                "EuroSAT",
                "Food101",
                "ImageNet",
                "Dataset from File",]
        }
        self.dataset_container = DragSourceContainer("Датасеты", 150, -300)
        self.dataset_container.add_drag_source(datasets.values())
        #endregion
        
        
        
        #region layers
        layers = {
            layer: DragSource(layer)
            for layer
            in ["LazyLinear",
                "LazyConv1d","LazyConv2d","LazyConv3d",
                "Flatten","Concatenate",
                "LazyBatchNorm1d","LazyBatchNorm2d","LazyBatchNorm3d",
                "AvgPool2d","MaxPool2d","AdaptiveAvgPool2d",
                "Dropout"]
         }
        self.layer_container = DragSourceContainer("Слои|ф.активации", 150, -300)
        self.layer_container.add_drag_source(layers.values())
        #endregion
        
        #region activations
        activations = {
            activation: DragSource(activation)
            for activation
            in ["ReLU","Sigmoid","Tanh",]
         }
        self.activation_container = DragSourceContainer("Функции активаций", 150, 0)
        self.activation_container.add_drag_source(activations.values())
        #endregion

        #region architectures
        archs = {
            'LeNet5': DragSource("LeNet5",
                                (
                                    (layers['LazyConv2d'], {'out_channels':6,"kernel_size":5,"stride":1,"padding":3}),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":5,"stride":1,"padding":1}),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':120}), (activations['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':84}), (activations['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':10}),
                                ),
                                 ),
            'AlexNet': DragSource("AlexNet",
                                (
                                    (layers['LazyConv2d'], {'out_channels':96,"kernel_size":11,"stride":4,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':256,"kernel_size":5,"stride":1,"padding":2, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':384,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':256,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":3,"stride":2}),(layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':4096,  }), (activations['ReLU'], ),(layers['Dropout'], {'p':0.5}),
                                    (layers['LazyLinear'], {'out_features':84,  }), (activations['ReLU'], ),(layers['Dropout'], {'p':0.5}),
                                    (layers['LazyLinear'], {'out_features':10,  }),
                                ),
                                 ),
            'VGG-11': DragSource("VGG-11",
                                (
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':32,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':64,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':64,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':128,"kernel_size":3,"stride":1,"padding":1, }),(activations['ReLU'], ),
                                    (layers['MaxPool2d'], {"kernel_size":2,"stride":2}),
                                    (layers['Flatten'], ),
                                    (layers['LazyLinear'], {'out_features':120,  }), (activations['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':84,  }), (activations['ReLU'], ),
                                    (layers['LazyLinear'], {'out_features':10,  }),
                                ),
                                 ),
            'Conv-MLP': DragSource("Conv-MLP",
                                (
                                    (
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':11, 'stride':4,'padding':0, }),(activations['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0,  }),(activations['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1, 'stride':1,'padding':0,  }),(activations['ReLU'], )
                                    ),
                                )),
            
        }
        archs['NiN'] = DragSource("NiN",
                                (
                                    (archs['Conv-MLP'], [{'out_channels':[96, None, 96, None, 96, None], 'kernel_size':[11, None, 1, None, 1], 'stride':[4, None,1,None, 1,None],'padding':[0,None,0,None,0,None]}]),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['Conv-MLP'], [{'out_channels':[256, None, 256, None, 256, None], 'kernel_size':[5,None,1,None,1,None], 'stride':[1,None,1,None,1, None],'padding':[2,None,0,None,0,None]}]),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (archs['Conv-MLP'], [{'out_channels':[384, None, 384, None, 384, None], 'kernel_size':[3,None,1,None,1,None], 'stride':[1,None,1,None,1, None],'padding':[1,None,0,None,0,None]}]),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2}),
                                    (layers['Dropout'], {'p':0.5}),
                                    (archs['Conv-MLP'], [{'out_channels':[10, None, 10, None, 10, None], 'kernel_size':[3,None,1,None,1,None], 'stride':[1,None,1,None,1,None],'padding':[1,None,0,None,0,None]}]),
                                    (layers['AdaptiveAvgPool2d'], {'output_size':'[1, 1]'}),(layers['Flatten'],)
                                ))
        archs['Inception'] = DragSource("Inception",
                                (
                                    (   (layers['LazyConv2d'], {'out_channels':64, 'kernel_size':1}), (activations['ReLU'], ) ),
                                    (
                                        (layers['LazyConv2d'], {'out_channels':96, 'kernel_size':1}), (activations['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':128, 'kernel_size':3, 'padding':1}), (activations['ReLU'], )
                                    ),    
                                    (
                                        (layers['LazyConv2d'], {'out_channels':16, 'kernel_size':1}),(activations['ReLU'], ),
                                        (layers['LazyConv2d'], {'out_channels':32, 'kernel_size':5, 'padding':2}), (activations['ReLU'], )
                                    ),
                                    (
                                        (layers['MaxPool2d'], {'kernel_size':3, 'stride':1, "padding":1}),
                                        (layers['LazyConv2d'], {'out_channels':32, 'kernel_size':1}), (activations['ReLU'], )
                                    )
                                ))
        archs['GoogLeNet'] = DragSource("GoogLeNet",
                                (
                                    (layers['LazyConv2d'], {'out_channels':64, 'kernel_size':7, 'stride':2, 'padding':3}), 
                                    (activations['ReLU'], ),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2, 'padding':1}),
                                    (layers['LazyConv2d'], {'out_channels':64, 'kernel_size':1}), 
                                    (activations['ReLU'], ),
                                    (layers['LazyConv2d'], {'out_channels':192, 'kernel_size':3, 'padding':1}), 
                                    (activations['ReLU'], ),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2, 'padding':1}),
                                    (archs['Inception'], [{'out_channels': [64, None]}, 
                                                          {'out_channels': [96, None, 128, None]}, 
                                                          {'out_channels': [16, None, 32, None]}, 
                                                          {'out_channels': [None, 32, None]}]),
                                    (layers['Concatenate'], ),
                                    (archs['Inception'], [{'out_channels': [128, None]}, 
                                                          {'out_channels': [128, None, 192, None]}, 
                                                          {'out_channels': [32, None, 96, None]}, 
                                                          {'out_channels': [None, 64, None]}]),
                                    (layers['Concatenate'], ),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2, 'padding':1}),
                                    (archs['Inception'], [{'out_channels': [192, None]}, 
                                                          {'out_channels': [96, None, 208, None]}, 
                                                          {'out_channels': [16, None, 48, None]}, 
                                                          {'out_channels': [None, 64, None]}]),
                                    (layers['Concatenate'], ),
                                    (archs['Inception'], [{'out_channels': [160, None]}, 
                                                          {'out_channels': [112, None, 224, None]}, 
                                                          {'out_channels': [24, None, 64, None]}, 
                                                          {'out_channels': [None, 64, None]}]),
                                    (layers['Concatenate'], ),
                                    (archs['Inception'], [{'out_channels': [128, None]}, 
                                                          {'out_channels': [128, None, 256, None]}, 
                                                          {'out_channels': [24, None, 64, None]}, 
                                                          {'out_channels': [None, 64, None]}]),
                                    (layers['Concatenate'], ),
                                    (archs['Inception'], [{'out_channels': [112, None]}, 
                                                          {'out_channels': [144, None, 288, None]}, 
                                                          {'out_channels': [32, None, 64, None]}, 
                                                          {'out_channels': [None, 64, None]}]),
                                    (layers['Concatenate'], ),
                                    (archs['Inception'], [{'out_channels': [256, None]}, 
                                                          {'out_channels': [160, None, 320, None]}, 
                                                          {'out_channels': [32, None, 128, None]}, 
                                                          {'out_channels': [None, 128, None]}]),
                                    (layers['Concatenate'], ),
                                    (layers['MaxPool2d'], {'kernel_size':3, 'stride':2, 'padding':1}),    
                                    (archs['Inception'], [{'out_channels': [256, None]}, 
                                                          {'out_channels': [160, None, 320, None]}, 
                                                          {'out_channels': [32, None, 128, None]}, 
                                                          {'out_channels': [None, 128, None]}]),  
                                    (layers['Concatenate'], ),
                                    (archs['Inception'], [{'out_channels': [384, None]}, 
                                                          {'out_channels': [192, None, 384, None]}, 
                                                          {'out_channels': [48, None, 128, None]}, 
                                                          {'out_channels': [None, 128, None]}]),    
                                    (layers['Concatenate'], ),
                                    (layers['AdaptiveAvgPool2d'], {'output_size':'[1, 1]'}),  
                                    (layers['Flatten'], ), 
                                    (layers['LazyLinear'], {'out_features':10}), 
                                   
                                ))
        archs['BN LeNet'] = DragSource('BN LeNet',
                                (
                                    (layers['LazyConv2d'], {'out_channels':6,"kernel_size":5}),(layers['LazyBatchNorm2d'], ),
                                    (activations['Sigmoid'],), (layers['AvgPool2d'], {'kernel_size': 2, 'stride':2}),
                                    (layers['LazyConv2d'], {'out_channels':16,"kernel_size":5}),(layers['LazyBatchNorm2d'], ),
                                    (activations['Sigmoid'],), (layers['AvgPool2d'], {'kernel_size': 2, 'stride':2}),
                                    (layers['Flatten'], ), (layers['LazyLinear'], {'out_features': 120}),(layers['LazyBatchNorm1d'], ), 
                                    (activations['Sigmoid'], ), (layers['LazyLinear'], {'out_features': 84}),(layers['LazyBatchNorm1d'], ),
                                    (activations['Sigmoid'], ), (layers['LazyLinear'], {'out_features': 10})
                                ))
        
        self.archs_container = DragSourceContainer("Модули", 150, 0)
        self.archs_container.add_drag_source(archs.values())
        #endregion

        

        
    def update(self):

        with dpg.mutex():
            dpg.delete_item(self.left_panel, children_only=True)
            self.dataset_container.submit(self.left_panel)
            self.archs_container.submit(self.left_panel)

            dpg.delete_item(self.right_panel, children_only=True)
            self.layer_container.submit(self.right_panel)
            self.activation_container.submit(self.right_panel)

                

    def start(self):
        dpg.set_viewport_title("Deep Learning Constructor")
        dpg.show_viewport()
        
            
        with dpg.window() as main_window:

            with dpg.menu_bar():
                with dpg.menu(label="Файл"):
                    dpg.add_menu_item(label="Открыть", callback=lambda:self.node_editor.callback_file(self.node_editor.open))
                    dpg.add_menu_item(label="Сохранить", callback=lambda:self.node_editor.callback_file(self.node_editor.save))
                    dpg.add_menu_item(label="Сбросить", callback=self.node_editor.clear)

                with dpg.menu(label="Настройки"):
                    dpg.add_menu_item(label="Логирование", check=True, callback=lambda s,check_value,u:Configs.set_logger)
                    with dpg.menu(label="Инструменты"):
                        dpg.add_menu_item(label="Show Metrics", callback=lambda:dpg.show_tool(dpg.mvTool_Metrics))
                        dpg.add_menu_item(label="Show Documentation", callback=lambda:dpg.show_tool(dpg.mvTool_Doc))
                        dpg.add_menu_item(label="Show Debug", callback=lambda:dpg.show_tool(dpg.mvTool_Debug))
                        dpg.add_menu_item(label="Show Style Editor", callback=lambda:dpg.show_tool(dpg.mvTool_Style))
                        dpg.add_menu_item(label="Show Font Manager", callback=lambda:dpg.show_tool(dpg.mvTool_Font))
                        dpg.add_menu_item(label="Show Item Registry", callback=lambda:dpg.show_tool(dpg.mvTool_ItemRegistry))
                        dpg.add_menu_item(label="Show About", callback=lambda:dpg.show_tool(dpg.mvTool_About))
                
                with dpg.menu(tag='menu_message_logger', label='---Сообщения---'):
                    dpg.add_child_window(tag='message_logger', height=200, width=1000)
            
            with dpg.group(tag='panel', horizontal=True):
                # left panel
                with dpg.group(tag=self.left_panel):
                    self.dataset_container.submit(self.left_panel)
                    self.archs_container.submit(self.left_panel) 

                # center panel
                with dpg.group(tag=self.center_panel):
                    self.node_editor.submit(self.center_panel)
                    dpg.add_text(tag='hover_logger', default_value="Текущий элемент: ", 
                                 parent=self.center_panel)

                # right panel
                with dpg.group(tag=self.right_panel):
                    self.layer_container.submit(self.right_panel)
                    self.activation_container.submit(self.right_panel)


            def on_key_la(sender, app_data):
                if dpg.is_key_released(dpg.mvKey_S):
                    self.node_editor.callback_file(self.node_editor.save)
                if dpg.is_key_released(dpg.mvKey_O):
                    self.node_editor.callback_file(self.node_editor.open)

            with dpg.handler_registry():
                dpg.add_key_press_handler(dpg.mvKey_Control, callback=on_key_la)
                    
                              
        dpg.set_primary_window(main_window, True)
        dpg.start_dearpygui()
        

app = App()
app.start()
