from lightning.pytorch.callbacks import TQDMProgressBar
from nodes.tools import ViewNode_2D
import multiprocessing
import time


class ProgressBoard(TQDMProgressBar):

    def __init__(self, task, widget):
        super().__init__()  
        self.enable = True
        self.widget: ViewNode_2D = widget
        self.task = task
        self.x = 0

    def disable(self):
        self.enable = False

    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx) 
        self.x += 1
        self.widget._execute(metrics=self.get_metrics(trainer, pl_module), 
                            x=self.x)
        

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)        
        return items





class CustomDataset:

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        transform = None,
        **params
    ) -> None:
        print("HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEY")

    @staticmethod
    def copy_dataset(sender, app_data, train_params__file: tuple[2]):
        train_params, filepath_uuid = train_params__file
        print(train_params__file)
        
        data_node = dpg.get_item_user_data(
                        dpg.get_item_parent(train_params._input_attributes[0]._linked_out_attr.uuid))  
            
        filepath = dpg.get_value(filepath_uuid)
        print(data_node, filepath)
        
        

    
def subprocess(target, name, args):
    def execute(queue, target, args):
        print(queue, target, args)
        queue.put(target(**args))
    queue = multiprocessing.Queue()
    print(queue, target, args)
    p = multiprocessing.Process(target=execute, name=name, args=(queue, target, args,))
    p.start()
    p.join(10)
    if p.is_alive():
        print(f"{name} is running... let's kill it...")
        p.terminate()
        p.join()
    value = queue.get()
    print(value)
    return value