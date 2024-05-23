import dearpygui.dearpygui as dpg
from core.node import Node
from core.link_node import LinkNode
from core.utils import select_path
from config.settings import Configs, BaseGUI
import pdb
import json

class NodeEditor(BaseGUI):

    
    def __init__(self) -> None:
        super().__init__()
        self.__nodes: list[Node] = [] 
    
    @staticmethod
    def factory(app_data: tuple, node_params: dict=None):
        label, generator, data, params, default_params, node_params = app_data

        node: Node = generator(label, data, params, default_params, **node_params) \
                if node_params \
                else generator(label, data, params, default_params)
        return node      

    def add_node(self, node: Node):
        self.__nodes.append(node)

    def on_drop(self, sender: int, node, node_params: dict, module=False):
        generator = node[1].__qualname__
        node = NodeEditor.factory(node, node_params)
        if not module: 
            if not (generator=='ModuleNode.factory'):
                 self.add_node(node)
            node._submit(self.uuid)
        return node

    def clear(self):
        dpg.delete_item(self.uuid, children_only=True)
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
        LinkNode.link_nodes(node_instances, self.uuid)
        
        
    def delete_selected_nodes(self) -> None:
        selected_nodes = dpg.get_selected_nodes(self.uuid)
        for node_id in selected_nodes:
            node: Node = dpg.get_item_user_data(node_id)
            self.delete_node(node)
    
    def delete_node(self, node: Node):
        self.__nodes.remove(node)
        node._del()

    @staticmethod
    def delete_in_editor(editor_uuid:int, node:Node):
        editor: NodeEditor = dpg.get_item_user_data(dpg.get_item_parent(editor_uuid))
        editor.delete_node(node)
        return editor

    def submit(self, parent):
        
        with dpg.child_window(width=-160, height=-25, 
                              parent=parent, 
                              user_data=self, 
                              drop_callback=lambda s, a, u: 
                                  dpg.get_item_user_data(s).on_drop(s, a, u)):
            with dpg.node_editor(callback=LinkNode._link_callback,
                                 delink_callback=LinkNode._delink_callback,
                                 tag=self.uuid, 
                                 width=-1, height=-1):
                for node in self.__nodes:
                    node._submit(self.uuid)
                with dpg.handler_registry():
                    dpg.add_key_press_handler(dpg.mvKey_Delete,
                                            callback=self.delete_selected_nodes)