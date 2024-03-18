import dearpygui.dearpygui as dpg
from core.node import Node
from core.input_node_attr import InputNodeAttribute


class ViewNode_2D(Node):

    @staticmethod
    def factory(name, data=None, params:tuple[dict]=None, default_params: dict[str,str]=None, **node_params):
        node = ViewNode_2D(name, data, params, default_params, **node_params)
        return node

    def __init__(self, label: str, data=None, params:tuple[dict]=None, default_params: dict[str,str]=None, **node_params):
        super().__init__(label, data, **node_params)

        self.add_input_attribute(InputNodeAttribute("full dataset", self))
        self.add_params(params)

        self.x_axis = dpg.generate_uuid()
        self.y_axis = dpg.generate_uuid()
        self.plot = dpg.generate_uuid()
        

    def custom(self):

        with dpg.plot(height=400, width=400, no_title=True, tag=self.plot):
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag=self.x_axis)
            # dpg.set_axis_limits(dpg.last_item(), 0, 10)
            dpg.add_plot_axis(dpg.mvYAxis, label="estimates", tag=self.y_axis)
            # dpg.set_axis_limits(dpg.last_item(), -0.1, 1)
            dpg.add_plot_legend()


    def execute(self, plt_lines=None, labels=None):

        x_axis_id = self.x_axis
        y_axis_id = self.y_axis
        dpg.delete_item(y_axis_id, children_only=True)
        for idx, line in enumerate(plt_lines):
            x_orig_data, y_orig_data = line.get_xdata(), line.get_ydata()
            dpg.add_line_series(x_orig_data, y_orig_data, parent=y_axis_id, label=labels[idx])
        dpg.fit_axis_data(x_axis_id)
        dpg.fit_axis_data(y_axis_id)
        self.finish()


