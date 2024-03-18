import dearpygui.dearpygui as dpg

class OutputNodeAttribute:

    def __init__(self, label: str = "output"):

        self._label = label
        self.uuid = dpg.generate_uuid()
        self._children = []  # output attributes
        self._data = None

    def add_child(self, parent, child):
        child.set_parent(self)
        self._children.append(child)

    def remove_child(self, parent, child):
        self._children.remove(child)
        child.reset_parent(self)

    def execute(self, data):
        self._data = data
        for child in self._children:
            child._data = self._data

    def submit(self, parent):

        with dpg.node_attribute(parent=parent, attribute_type=dpg.mvNode_Attr_Output,
                                user_data=self, tag=self.uuid):
            dpg.add_text(self._label)