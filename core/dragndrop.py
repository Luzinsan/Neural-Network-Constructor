from __future__ import annotations
import dearpygui.dearpygui as dpg
from config.settings import _source_theme
from typing import Optional


class DragSourceContainer:

    def __init__(self, label: str, width: int = 150, height: int = -1):

        self._label = label
        self._width = width
        self._height = height
        self._uuid = dpg.generate_uuid()
        self._children: list[DragSource] = []

    def add_drag_source(self, sources: tuple[DragSource]):
        for source in sources:
            self._children.append(source)

    def submit(self, parent):

        with dpg.child_window(parent=parent, width=self._width, height=self._height, tag=self._uuid, menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label, enabled=False)

            for child in self._children:
                child._submit(child_parent)



class DragSource:

    def __init__(self, label: str, node_generator, data, 
                 params: Optional[tuple[dict]]=None, default_params: Optional[dict[str, str]]=None, **node_params):

        self._label = label
        self._generator = node_generator
        self._data = data
        self._params = params
        self._default_params = default_params
        self._node_params = node_params
        

    def _submit(self, parent: DragSourceContainer):
        dpg.add_button(label=self._label, parent=parent, width=-1)
        dpg.bind_item_handler_registry(dpg.last_item(), "hover_handler")
        dpg.bind_item_theme(dpg.last_item(), _source_theme)
        with dpg.drag_payload(parent=dpg.last_item(), drag_data=(self._label, self._generator, self._data, self._params, self._default_params, self._node_params)):
            dpg.add_text(f"Name: {self._label}")


