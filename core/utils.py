from __future__ import annotations
import dearpygui.dearpygui as dpg
from torch import nn
from typing import Optional
import re
from config.settings import Configs

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




def send_message(message, type_message: str = 'error', brief:Optional[str]=None, callback=None): 
    
    
    dpg.set_item_label('menu_message_logger', f"Сообщения: {brief if brief else message}")
    logs: list = dpg.get_item_children('message_logger', 1)
    message = f"{brief}. Подробнее: {message}" if brief else message
    kwargs = dict(parent='message_logger', 
                  before=logs[0] if len(logs) else 0, 
                #   tracked=True
                  )
    if callback: 
        kwargs.update(dict(callback=callback, 
                           label=message))
        return dpg.add_button(**kwargs) 
        
    map_color_message = {
        'error':(255,0,0), 
        'warning':(255,255,0),
        'log':(0,255,0),
    }
    color = map_color_message.get(type_message)
    if color: 
        kwargs.update(dict(color=color))
        if type_message=='error' and re.search('Traceback', message):
            print(message)
            message = "\n".join(re.findall(r'^.*Error:.*$', message, re.MULTILINE))
            print("after: ", message, "ENDDDD")
    
    kwargs.update(dict(default_value=message))
    dpg.add_text(**kwargs)
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