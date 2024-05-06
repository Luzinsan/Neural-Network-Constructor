import dearpygui.dearpygui as dpg
from core.output_node_attr import OutputNodeAttribute
from core.input_node_attr import InputNodeAttribute
from typing import Union


class LinkNode:

    def __init__(self, input_uuid:InputNodeAttribute, output_uuid:OutputNodeAttribute):

        self.uuid = dpg.generate_uuid()
        self._input_attr = input_uuid
        self._output_attr = output_uuid

    def get_attrs(self):
        return self._input_attr, self._output_attr

    @staticmethod
    def _link_callback(node_editor_uuid, in_out_uuids):
        output_attr_uuid, input_attr_uuid = in_out_uuids

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        link_node = LinkNode(input_attr_uuid, output_attr_uuid)
        dpg.add_node_link(*link_node.get_attrs(), parent=node_editor_uuid, user_data=link_node, tag=link_node.uuid)
        output_attr.add_child(input_attr)


    @staticmethod
    def _delink_callback(node_editor_uuid, link_uuid):
        link: LinkNode = dpg.get_item_user_data(link_uuid)
        input_attr_uuid, output_attr_uuid = link.get_attrs()

        input_attr: InputNodeAttribute = dpg.get_item_user_data(input_attr_uuid)
        output_attr: OutputNodeAttribute = dpg.get_item_user_data(output_attr_uuid)

        output_attr.remove_child(node_editor_uuid, input_attr)
        dpg.delete_item(link.uuid)
        del link