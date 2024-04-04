import dearpygui.dearpygui as dpg
from core.node import Node
from core.input_node_attr import InputNodeAttribute
import collections
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt


def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

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
            dpg.add_plot_axis(dpg.mvYAxis, label="estimates", tag=self.y_axis)
            dpg.add_plot_legend()

    def end(self, max_steps):
        dpg.set_axis_limits(self.x_axis, 0, max_steps)
        # dpg.set_axis_limits(self.y_axis, -0.1, 1)

    
    def draw(self, x, y, label, every_n=1, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5)):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        
        use_svg_display()
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), ls, colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        return (plt_lines, labels)

    def execute(self, metrics: dict, x, every_n=1):
        for label, y in metrics.items():
            (plt_lines, labels) = self.draw(x, y, label)
            dpg.delete_item(self.y_axis, children_only=True)
            for idx, line in enumerate(plt_lines):
                x_orig_data, y_orig_data = line.get_xdata(), line.get_ydata()
                dpg.add_line_series(x_orig_data, y_orig_data, parent=self.y_axis, label=labels[idx])
        
        dpg.fit_axis_data(self.x_axis)
        dpg.fit_axis_data(self.y_axis)
        self.finish()


