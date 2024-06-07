from __future__ import annotations
import dearpygui.dearpygui as dpg
import external.DearPyGui_Markdown as dpg_markdown
from torch import nn
import datetime
from typing import Optional
import threading
import ctypes
import re
import pdb
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
        'error': '<font color="(191,47,1)">{}</font>', 
        'warning':'<font color="(191,165,1)">{}</font>',
        'log':'<font color="(22,144,0)">{}</font>',
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
    dpg.add_file_extension(".xlsx", color=(3, 138, 48, 255), custom_text="[Calc]")
    dpg.add_file_extension(".csv", color=(3, 138, 48, 255), custom_text="[CSV]")
    dpg.add_file_extension(".params", color=(3, 138, 48, 255), custom_text="[Params]")


with dpg.file_dialog(
        directory_selector=False,
        show=False,
        modal=True,
        width=700, height=400, 
        default_filename=datetime.datetime.now().strftime('%Y_%m_%d'),
        tag='json_file',
        ):
    dpg.add_file_extension('.json', color=(126, 138, 3, 255))
