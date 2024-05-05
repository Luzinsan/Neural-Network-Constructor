import dearpygui.dearpygui as dpg

class OutputNodeAttribute:

    def __init__(self, label: str = "output"):

        self._uuid = dpg.generate_uuid()
        self._label = label
        self._children: list["core.InputNodeAttribute"] = []  # output attributes
        self._data = None

    def add_child(self, parent, child: "core.InputNodeAttribute"):
        child.set_linked_attr(self)
        self._children.append(child)

    def remove_child(self, parent, child: "core.InputNodeAttribute"):
        self._children.remove(child)
        child.reset_linked_attr()

    def execute(self, data):
        self._data = data
        for child in self._children:
            child._data = self._data

    def _submit(self, parent):

        with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                user_data=self, tag=self._uuid):
            dpg.add_text(self._label)