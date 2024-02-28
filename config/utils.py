from . import dpg
from .settings import env


class GUI:

    def __init__(self):
        pass

    @staticmethod
    def select_path(sender, app_data, user_data):
        dpg.set_item_user_data('file_dialog', user_data)
        dpg.show_item('file_dialog')

    @staticmethod
    def set_path(sender, app_data):
        tag_path = dpg.get_item_user_data('file_dialog')
        dpg.configure_item(tag_path, default_value=app_data['file_path_name'])


    def show_box_file_data():
        with dpg.group(tag='File', parent='kind_data', horizontal=True):
             dpg.add_input_text(tag='input_file',
                                default_value='test.txt')
             dpg.add_button(label='Select Path Manually', 
                            callback=GUI.select_path, user_data='input_file')

    @staticmethod
    def switch_kind(sender, kind):
        dpg.delete_item('kind_data',children_only=True)
        dpg.show_item('kind_data')
        match kind:
            case 'File':
                GUI.show_box_file_data()
            case _:
                print("Default")
    
    @staticmethod
    def get_data(sender, app_data, 
                #  pipline: Pipline
                ):
        kind = dpg.get_value('train_data')
        match kind:
            case 'Syntetic':
                pass
                # setattr(pipline, 'data', 
                        # d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2))
            case 'File':
                file_path = dpg.get_value('input_file')
                # with open(file_path, 'r') as file:
                    # setattr(pipline, 'data', file.read())
                

with dpg.file_dialog(directory_selector=False, show=False, callback=GUI.set_path, tag="file_dialog",
                     width=700, height=400, modal=True):
    dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")
    dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255), custom_text="[Excel]")