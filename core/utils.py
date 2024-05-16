import dearpygui.dearpygui as dpg
from torch import nn


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