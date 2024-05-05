from core.node import Node
from core.input_node_attr import InputNodeAttribute


class TrainParamsNode(Node):

    @staticmethod
    def factory(name, data=None, train_params: tuple[dict]=None, **node_params):
        node = TrainParamsNode(name, data, train_params, **node_params)
        return node

    def __init__(self, label: str, data=None, train_params: tuple[dict]=None, **node_params):
        super().__init__(label, data, **node_params)

        self._add_input_attribute(InputNodeAttribute("train dataset", self))
        self._add_params(train_params)


    def set_pipline(self, pipeline):
        self.pipeline = pipeline

