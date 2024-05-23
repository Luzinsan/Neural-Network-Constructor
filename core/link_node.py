import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from config.settings import BaseGUI


class LinkNode(BaseGUI):

    def __init__(self, input_uuid:int, output_uuid:int):
        super().__init__()
        self._input_attr = input_uuid
        self._output_attr = output_uuid

    def get_attrs(self):
        return self._input_attr, self._output_attr

    @staticmethod
    def _link_callback(node_editor_uuid: int, in_out_uuids: tuple[int, int]):
        output_attr_uuid, input_attr_uuid = in_out_uuids

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        link_node = LinkNode(input_attr_uuid, output_attr_uuid)
        dpg.add_node_link(*link_node.get_attrs(), parent=node_editor_uuid, user_data=link_node, tag=link_node.uuid)
        output_attr.add_child(input_attr)


    @staticmethod
    def _delink_callback(node_editor_uuid: int, link_uuid: int):
        link: LinkNode = dpg.get_item_user_data(link_uuid)
        input_attr_uuid, output_attr_uuid = link.get_attrs()

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        output_attr.remove_child(input_attr)
        dpg.delete_item(link_uuid)
        del link
        
    @staticmethod
    def link_nodes(nodes, editor_uuid):
        for inx in range(len(nodes) - 1):
            first = nodes[inx]._output_attributes[0]
            sec = nodes[inx + 1]._input_attributes[0]
            LinkNode._link_callback(editor_uuid, (first.uuid, sec.uuid))