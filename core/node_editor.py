import dearpygui.dearpygui as dpg
from core.node import Node
from core.link_node import LinkNode

class NodeEditor:

    
    def __init__(self) -> None:

        self._uuid = dpg.generate_uuid()
        self.__nodes: list[Node] = []
        

    def add_node(self, node: Node):
        self.__nodes.append(node)

    def on_drop(self, sender: int, app_data: tuple, node_params: dict):
        label, generator, data, params, default_params, node_params = app_data
        
        node: Node = generator(label, data, params, default_params, **node_params) \
                if node_params \
                else generator(label, data, params, default_params)
       
        node._submit(self._uuid)
        self.add_node(node)
        return node

    def clear(self):
        dpg.delete_item(self._uuid, children_only=True)
        self.__nodes.clear()
        
    def delete_node(self) -> None:
        selected_nodes = dpg.get_selected_nodes(self._uuid)
        for node_id in selected_nodes:
            node: Node = dpg.get_item_user_data(node_id)
            self.__nodes.remove(node)
            node.__del__()


    def submit(self, parent):
        
        with dpg.child_window(width=-160, height=-25, parent=parent, user_data=self, 
                              drop_callback=lambda s, a, u: dpg.get_item_user_data(s).on_drop(s, a, u)):
            with dpg.node_editor(callback=LinkNode._link_callback,
                                 delink_callback=LinkNode._delink_callback,
                                 tag=self._uuid, width=-1, height=-1):
                for node in self.__nodes:
                    node._submit(self._uuid)
                with dpg.handler_registry():
                    dpg.add_key_press_handler(dpg.mvKey_Delete,
                                            callback=self.delete_node)