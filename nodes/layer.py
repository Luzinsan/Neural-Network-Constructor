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
    def factory(name, data, params:Optional[tuple[dict]]=None, default_params: Optional[dict[str, str]]=None, **node_params):
        node = LayerNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, 
                 label: str, 
                 data, 
                 params:Optional[tuple[dict]]=None, 
                 default_params: Optional[dict[str,str]]=None, 
                 **node_params):
        super().__init__(label, data, **node_params)
        self._add_input_attribute(InputNodeAttribute("data"))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        self._add_params(params)


class ModuleNode(Node):

    @staticmethod
    def factory(label, 
                sequential: tuple[tuple[DragSource, dict]], 
                params: Optional[list[dict[str, Any]]]=None, 
                default_params: Optional[dict[str,str]]=None, 
                **node_params):
        node = ModuleNode(label, sequential, params, default_params, **node_params)
        return node

    def __init__(self, 
                 label, 
                 sequential: tuple[tuple[DragSource, dict]], 
                 params: Optional[list[dict[str, Any]]]=None, 
                 default_params: Optional[dict[str,str]]=None, 
                 **node_params):
        super().__init__(label, None, **node_params)
        self._add_input_attribute(InputNodeAttribute("data"))
        self._add_output_attribute(OutputNodeAttribute("weighted data"))
        expand = [{"label":"Развернуть", "type":"button", "callback":self.expand},
                  {"label":"Редактировать", "type":"button", "callback":self.edit}]
        if not params: params = expand
        else: params += expand
        
        self._add_params(params)
        self.sequential = sequential
       
        self.pos: dict[str, int] = node_params.get('pos', {"x":0, "y":0})

    @staticmethod
    def _replace(module, new_defaults, idx):
        if len(module) > 1: # если указаны новые параметры по умолчанию для слоя
            for key in module[1].keys():
                if params := new_defaults.get(key):
                    module[1][key] = params[idx]
        

    @staticmethod
    def replace_default_params(sequential: tuple[tuple[object, dict]], 
                               new_defaults: dict):
        updated_sequential = deepcopy(sequential)
        for i, layer_defaults in enumerate(updated_sequential):
            if isinstance(layer_defaults[0], tuple):
                for idx_module,  module in enumerate(layer_defaults):
                    ModuleNode._replace(module, new_defaults[i], idx_module)
            else:
                ModuleNode._replace(layer_defaults, new_defaults, i)
            
        return updated_sequential


    def _submit_in_editor(self, editor, parent=None):
       
        if not hasattr(self, '_node_editor_uuid') and (parent==None):
            raise RuntimeError("id node_editor не определен")
        if not hasattr(self, '_node_editor_uuid'): self._node_editor_uuid = parent
        
        prev_node = None
        lasts_in_branch = []
        for node in self._data:
            if isinstance(node, list):
                flag = False
                for branch in node:
                    if isinstance(branch, list):
                        for module in branch:
                            editor.add_node(module)
                            module._submit(self._node_editor_uuid)
                        lasts_in_branch.append(branch[-1])
                        if prev_node: branch.insert(0, prev_node)
                        LinkNode.link_nodes(branch, self._node_editor_uuid) 
                    else:
                        flag = True
                        editor.add_node(branch)
                        branch._submit(self._node_editor_uuid)
                if flag:
                    lasts_in_branch.append(node[-1])
                    if prev_node: node.insert(0, prev_node)
                    LinkNode.link_nodes(node, self._node_editor_uuid)
            else:
                editor.add_node(node)
                node._submit(self._node_editor_uuid)
                if len(lasts_in_branch):
                    for last in lasts_in_branch:
                        LinkNode._link_nodes(last, node, self._node_editor_uuid)
                    lasts_in_branch = []
                elif prev_node: LinkNode._link_nodes(prev_node, node, self._node_editor_uuid)
                prev_node = node
       
     
    def expand(self) -> None:
        editor = NodeEditor.delete_in_editor(self._node_editor_uuid, self)
        self._submit_in_editor(editor)
            
    def edit(self) -> None:
        NodeEditor.delete_in_editor(self._node_editor_uuid, self)
        dpg.configure_item(self.modal, show=True)
   
    def submit_module(self, sequential, modal):
        all_sources = []
        all_params = []
        for idx, node in enumerate(sequential):
            source: DragSource = deepcopy(node[0])
            defaults = None
            if len(node)>1:
                defaults = deepcopy(node[1])
                if source._generator.__qualname__ == 'ModuleNode.factory':
                    source._data = ModuleNode.replace_default_params(
                                            source._data, 
                                            defaults)
                    with dpg.collapsing_header(parent=modal, 
                                               label=source._label, 
                                               default_open=False) as submodule:
                        sources, self.pos, source_params = \
                            ModuleNode(source._label, 
                                       source._data,
                                       pos={'x':0, "y":self.pos['y'] + 150})\
                                           .submit_sequential(submodule)
                    all_sources.append(sources)
                    all_params.append(source_params)
                    continue
            all_params.append(
                ParamNode.submit_config(source._label, 
                                        source._params, 
                                        defaults, 
                                        modal))
            self.pos['x'] += 300
            params = {"pos":(self.pos['x'], self.pos['y'])}

            all_sources.append(
                deepcopy((source._label, 
                          source._generator, 
                          source._data, 
                          source._params, 
                          source._default_params, 
                          params))
                )
        return all_sources, self.pos, all_params

    def submit_sequential(self, modal: int) -> tuple[list, dict, list]:
        if not isinstance(self.sequential[0][0], tuple):
            return self.submit_module(self.sequential, modal)
        
        all_sources = []
        all_params = []

        is_branch = len(self.sequential) > 1

        for idx, branch in enumerate(self.sequential, 1):
            if is_branch:
                modal = dpg.add_collapsing_header(label=f'Ветка #{idx}', default_open=False)
                self.pos['x'] = 0
                self.pos['y'] += 200
                
            sources, _, source_params = self.submit_module(branch, modal)
            all_sources.append(sources)
            all_params.append(source_params)
            self.pos['x'] += self.pos['x'] / 2
        return all_sources, self.pos, all_params
    
    @staticmethod
    def _collect(sources, params):
        if params:
            for param, origin_param in zip(params, sources[3]):
                returned = param.get_value()
                if returned and ((value := returned.get(origin_param['label'], None)) is not None):
                    origin_param['default_value'] = value   

    def recursive_collect(self, sources, params, editor):
        if isinstance(sources, list) and isinstance(params, list):
            multi_source = []
            for source, param in zip(sources, params):
                multi_source.append(self.recursive_collect(source, param, editor))
            return multi_source
        ModuleNode._collect(sources, params)
        return editor.on_drop(None, sources, None, module=True)

    def _collect_submit(self, all_sources, all_params, parent, modal, collapse_checkbox):
        editor: NodeEditor = dpg.get_item_user_data(dpg.get_item_parent(parent))
        self._data = self.recursive_collect(all_sources, all_params, editor)
        
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
                all_sources, _, all_params = self.submit_sequential(child_window)
            dpg.add_separator()
            collapse_checkbox = dpg.add_checkbox(label="Свернуть модель", default_value=True)
            dpg.add_button(label="Продолжить", callback=lambda:self._collect_submit(all_sources, all_params, parent, modal, collapse_checkbox))
        

