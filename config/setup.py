import dearpygui.dearpygui as dpg

########################################################################################################################
# Setup
########################################################################################################################
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()
########################################################################################################################
def on_exit(sender, app_data, user_data):
    print("closed")
    dpg.stop_dearpygui()
    dpg.destroy_context()
dpg.set_exit_callback(on_exit)