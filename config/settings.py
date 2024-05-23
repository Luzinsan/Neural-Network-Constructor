import dearpygui.dearpygui as dpg
from dearpygui import demo
from typing import Optional
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
        # demo.show_demo()
        # show_documentation()
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
            
try:
    with dpg.item_handler_registry(tag=BaseGUI.generate_uuid()) as hover_handler:
        dpg.add_item_hover_handler(callback=lambda s,a,u: curent_item(a))        
except SystemError as err: print("Удаление узла")