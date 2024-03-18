import dearpygui.dearpygui as dpg
from config.settings import _source_theme

class DragSource:

    def __init__(self, label: str, node_generator, data=None, params: tuple[dict]=None, default_params: dict[str, str]= None, node_params=None):

        self.label = label
        self._generator = node_generator
        self._data = data
        self._params = params
        self._default_params = default_params
        self._node_params = node_params
        

    def submit(self, parent):
        dpg.add_button(label=self.label, parent=parent, width=-1)
        dpg.bind_item_theme(dpg.last_item(), _source_theme)
        with dpg.drag_payload(parent=dpg.last_item(), drag_data=(self, self._generator, self._data, self._params, self._default_params, self._node_params)):
            dpg.add_text(f"Name: {self.label}")




class DragSourceContainer:

    def __init__(self, label: str, width: int = 150, height: int = -1):

        self._label = label
        self._width = width
        self._height = height
        self._uuid = dpg.generate_uuid()
        self._children: list[DragSource] = []  # drag sources

    def add_drag_source(self, sources: tuple[DragSource]):
        for source in sources:
            self._children.append(source)

    def submit(self, parent):

        with dpg.child_window(parent=parent, width=self._width, height=self._height, tag=self._uuid, menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label, enabled=False)

            for child in self._children:
                child.submit(child_parent)