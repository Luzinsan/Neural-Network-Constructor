from core.input_node_attr import InputNodeAttribute
from core.output_node_attr import OutputNodeAttribute
from core.node import Node



class UtilityNode(Node):

    @staticmethod
    def factory(name, data, params: tuple[dict]=None, default_params: dict[str,str]=None,**node_params):
        node = UtilityNode(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data, params: list[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)

        self.add_input_attribute(InputNodeAttribute("data", self))
        self.add_output_attribute(OutputNodeAttribute("processed data"))
        self.add_params(params)


