import dearpygui.dearpygui as dpg
import external.DearPyGui_Markdown as dpg_markdown # Import the library



########################################################################################################################
# Setup
########################################################################################################################
dpg.create_context()
dpg.create_viewport()

from . import font
dpg.bind_font(font.load())

dpg.setup_dearpygui()
import config.dicts
########################################################################################################################
def on_exit(sender, app_data, user_data):
    print("closed")
    dpg.stop_dearpygui()
    dpg.destroy_context()
dpg.set_exit_callback(on_exit)