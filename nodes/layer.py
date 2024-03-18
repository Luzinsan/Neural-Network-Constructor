from core.node import Node
from core.node_editor import NodeEditor
from core.input_node_attr import InputNodeAttribute
from core.output_node_attr import OutputNodeAttribute
from core.link_node import LinkNode
from core.dragndrop import DragSource


class LayerNode(Node):

    @staticmethod
    def factory(name, data, params:tuple[dict]=None, default_params: dict[str,str]=None, **node_params):
        node = LayerNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params:tuple[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)
        self.add_input_attribute(InputNodeAttribute("data", self))
        self.add_output_attribute(OutputNodeAttribute("weighted data"))
        self.add_params(params)


class ModuleNode:

    @staticmethod
    def factory(name, sequential: tuple[tuple[DragSource, dict]], node_editor, default_params: dict[str,str]=None, **node_params):
        node = ModuleNode(name, sequential, node_editor, default_params, **node_params)
        return node

    def __init__(self, name, sequential: tuple[tuple[DragSource, dict]], node_editor, default_params: dict[str,str]=None, **node_params):
        self.sequential = sequential
        self.node_editor: NodeEditor = node_editor


    def submit(self, parent):
        nodes = []
        for idx, node in enumerate(self.sequential):
            source: DragSource = node[0]
            if len(node)>1:
                defaults = node[1]
                for param in source._params:
                    if (label := param['label']) in defaults.keys():
                        param['default_value'] = defaults[label]
            params = {"pos":(idx*210+ 190, 0)}
            
            source = source, source._generator, source._data, source._params, source._default_params, params
            nodes.append(self.node_editor.on_drop(None, source, None))

        
        for inx in range(len(nodes) - 1):
            first = nodes[inx]._output_attributes[0]
            sec = nodes[inx + 1]._input_attributes[0]
            LinkNode._link_callback(self.node_editor.uuid, (first.uuid, sec.uuid))

