import dearpygui.dearpygui as dpg

class OutputNodeAttribute:

    def __init__(self, label: str = "output"):

        self._uuid = dpg.generate_uuid()
        self._label = label
        self._children: list["core.InputNodeAttribute"] = []  # input attributes

    def add_child(self, child: "core.InputNodeAttribute"):
        child.set_linked_attr(self)
        self._children.append(child)

    def remove_child(self, child: "core.InputNodeAttribute"):
        self._children.remove(child)
        child.reset_linked_attr()


    def _submit(self, parent):

        with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                user_data=self, tag=self._uuid):
            dpg.add_text(self._label)