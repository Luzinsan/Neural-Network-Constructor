from __future__ import annotations
import dearpygui.dearpygui as dpg
from config.settings import _source_theme, BaseGUI, hover_handler
from typing import Optional
from config import dicts



class DragSourceContainer(BaseGUI):

    def __init__(self, label: str, width: int = 150, height: int = -1):
        super().__init__()
        self._label = label
        self._width = width
        self._height = height
        self._children: list[DragSource] = []

    def add_drag_source(self, sources: tuple[DragSource]):
        for source in sources:
            self._children.append(source)

    def submit(self, parent):

        with dpg.child_window(tag=self.uuid, 
                              parent=parent, 
                              width=self._width, height=self._height, 
                              menubar=True) as child_parent:
            with dpg.menu_bar():
                dpg.add_menu(label=self._label, 
                             enabled=False)

            for child in self._children:
                child._submit(child_parent)



class DragSource():

    def __init__(self, label: str, data=None, 
                 **node_params):
        
        self._label = label
        func = dicts.modules[label].func
        
        self._data =  func if func else data
        self._generator = dicts.modules[label].generator
        self._params = dicts.modules[label].params
        self._default_params = dicts.modules[label].default_params
        self._popup: str = dicts.modules[label].popup
        self._tooltip: str = dicts.modules[label].tooltip
        self._details: str = dicts.modules[label].details
        self._image: str = dicts.modules[label].image
        self._node_params = node_params
        

    def _submit(self, parent: DragSourceContainer):
        button = dpg.add_button(label=self._label, 
                        parent=parent, 
                        width=-1,
                        user_data=self)
       
        dpg.bind_item_handler_registry(button, hover_handler)
        dpg.bind_item_theme(button, _source_theme)
        with dpg.drag_payload(parent=button, 
                              drag_data=(self._label, 
                                         self._generator, 
                                         self._data, 
                                         self._params, 
                                         self._default_params, 
                                         self._node_params)):
            dpg.add_text(f"Name: {self._label}")


