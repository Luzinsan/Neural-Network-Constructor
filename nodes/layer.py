from core.node import Node
from core.node_editor import NodeEditor
from core.input_node_attr import InputNodeAttribute
from core.output_node_attr import OutputNodeAttribute
from core.link_node import LinkNode
from core.dragndrop import DragSource
from core.param_node import ParamNode
import dearpygui.dearpygui as dpg
from typing import Optional, Any
from copy import deepcopy
import pdb


class LayerNode(Node):

    @staticmethod
    def factory(name, data, params:tuple[dict]|None=None, default_params: dict[str,str]=None, **node_params):
        node = LayerNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:tuple[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)
        self._add_input_attribute(InputNodeAttribute("data", self))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        self._add_params(params)


class ModuleNode(Node):

    @staticmethod
    def factory(label, sequential: tuple[tuple[DragSource, dict]], params: list[dict[str, Any]]=None, default_params: dict[str,str]=None, **node_params):
        node = ModuleNode(label, sequential, params, default_params, **node_params)
        return node

    def __init__(self, label, sequential: tuple[tuple[DragSource, dict]], params: list[dict[str, Any]]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, None, **node_params)
        self._add_input_attribute(InputNodeAttribute("data", self))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        expand = [{"label":"Развернуть", "type":"button", "callback":self.expand},
                  {"label":"Редактировать", "type":"button", "callback":self.edit}]
        if not params: params = expand
        else: params += expand
        
        self._add_params(params)
        self.sequential = sequential
       
        self.pos: dict[str, int] = node_params.get('pos', {"x":0, "y":0})

    @staticmethod
    def replace_default_params(sequential: tuple[tuple[object, dict]], new_defaults: dict):
        idx = 0
        updated_sequential = deepcopy(sequential)
        for i, layer_defaults in enumerate(updated_sequential):
            if len(layer_defaults) > 1: # если указаны новые параметры по умолчанию для слоя
                # params = item[1]
                for key in layer_defaults[1].keys():
                    if params := new_defaults.get(key):
                        layer_defaults[1][key] = params[idx]
                idx += 1
        return updated_sequential
   
    def _submit_in_editor(self, editor, parent=None):
       
        if not hasattr(self, '_node_editor_uuid') and (parent==None):
            raise RuntimeError("id node_editor не определен")
        if not hasattr(self, '_node_editor_uuid'): self._node_editor_uuid = parent
        
        for node in self._data:
            editor.add_node(node)
            node._submit(self._node_editor_uuid)
            
        for inx in range(len(self._data) - 1):
            first = self._data[inx]._output_attributes[0]
            sec = self._data[inx + 1]._input_attributes[0]
            LinkNode._link_callback(self._node_editor_uuid, (first._uuid, sec._uuid))
     
    def expand(self) -> None:
        editor = NodeEditor.delete_in_editor(self._node_editor_uuid, self)
        self._submit_in_editor(editor)
            
    def edit(self) -> None:
        NodeEditor.delete_in_editor(self._node_editor_uuid, self)
        dpg.configure_item(self.modal, show=True)
   
    def submit_module(self, modal: int) -> tuple[list, dict, list]:
        all_sources = []
        all_params = []
        for idx, node in enumerate(self.sequential):
            source: DragSource = node[0]
            if len(node)>1:
                defaults = node[1].copy()
                if source._generator.__qualname__ == 'ModuleNode.factory':
                    source._data = ModuleNode.replace_default_params(source._data, defaults)
                    with dpg.collapsing_header(parent=modal, label=source._label, default_open=False) as submodule:
                        sources, self.pos, source_params = ModuleNode(source._label, source._data,
                                                                      pos={'x':0, "y":self.pos['y'] + 200}).submit_module(submodule)
                    all_sources += sources
                    all_params += source_params
                    continue
            all_params.append(ParamNode.submit_config(source._label, source._params, defaults, modal))
            self.pos['x'] += 200
            params = {"pos":(self.pos['x'], self.pos['y'])}

            all_sources.append(
                deepcopy((source._label, source._generator, source._data, source._params, source._default_params, params))
                )

        return all_sources, self.pos, all_params
    
    def _collect_submit(self, all_sources, all_params, parent, modal, collapse_checkbox):
        for idx, source in enumerate(all_sources):
            if all_params[idx]:
                for param, origin_param in zip(all_params[idx], source[3]):
                    returned = param.get_value()
                    if returned and ((value := returned.get(origin_param['label'], None)) is not None):
                        origin_param['default_value'] = value      
        editor: NodeEditor = dpg.get_item_user_data(dpg.get_item_parent(parent))
        
        self._data = [editor.on_drop(None, source, None, module=True) for source in all_sources]
        if dpg.get_value(collapse_checkbox):
            super()._submit(parent)
            editor.add_node(self)
        else:
            self._submit_in_editor(editor, parent)
        dpg.configure_item(modal, show=False)
        

    def _submit(self, parent):
        with dpg.window(modal=True, pos=(500,0), label=f'Конфигурирование модуля {self._label}', min_size=(300, 500)) as modal:
            self.modal = modal
            with dpg.child_window(height=-50) as child_window:
                all_sources, _, all_params = self.submit_module(child_window)
            dpg.add_separator()
            collapse_checkbox = dpg.add_checkbox(label="Свернуть модель", default_value=True)
            dpg.add_button(label="Продолжить", callback=lambda:self._collect_submit(all_sources, all_params, parent, modal, collapse_checkbox))
        

