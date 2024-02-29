from . import dpg
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
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (77, 7, 143), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
    with dpg.theme_component(dpg.mvInputInt):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (30, 77, 70), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
    with dpg.theme_component(dpg.mvText):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (15, 61, 131), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
dpg.bind_theme(global_theme)


with dpg.font_registry():
    with dpg.font('config/UbuntuMono-R.ttf', 20, default_font=True, tag="Default font"):
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
dpg.bind_font("Default font")


def on_exit(sender, app_data, user_data):
    print("closed")


dpg.set_exit_callback(on_exit)