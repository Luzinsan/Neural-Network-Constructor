import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light 
import dearpygui.demo as demo

from environs import Env
env = Env()
env.read_env()


########################################################################################################################
# Setup
########################################################################################################################
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()


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
        dpg.add_theme_color(dpg.mvThemeCol_Border, (179, 222, 255), category=dpg.mvThemeCat_Core) # Цвет оконтовок
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (227, 255, 255), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,  (142, 130, 217), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (255, 255, 255), category=dpg.mvThemeCat_Core)
        
        # demo.show_demo()
        light_theme = create_theme_imgui_light()
        # dpg.bind_theme(light_theme)

    with dpg.font_registry() as default_font:
        with dpg.font('config/Comfortaa-Regular.ttf', 14, default_font=True, tag="Default font"):
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    
    dpg.bind_font("Default font")

with dpg.theme() as _source_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, [152, 103, 204]) # Цвет кнопок в контейнерах
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [182, 156, 230]) # Цвет кнопок при наведении
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [227, 204, 237]) # Цвет кнопок при нажатии
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

 

dpg.bind_theme(global_theme)


def on_exit(sender, app_data, user_data):
    print("closed")


dpg.set_exit_callback(on_exit)