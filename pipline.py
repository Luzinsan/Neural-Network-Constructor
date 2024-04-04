from typing import Union
from torch import nn
import torch
import dearpygui.dearpygui as dpg
from nodes.tools import ViewNode_2D
from core.node import Node
from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import v2 
import os
from clearml import Task

reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
float32 = torch.float32


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
        self.widget.execute(metrics=self.get_metrics(trainer, pl_module), 
                            x=self.x)
        

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)        
        return items




class Module(LightningModule):
    def __init__(self, sequential, optimizer, lr, loss_func):
        super(Module, self).__init__()

        self.net = sequential
        self.net.apply(Module.init_cnn)
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    @staticmethod  
    def init_cnn(module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)


    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr)
        return optim
    
    def forward(self, X):
        return self.net(X)
    

    def accuracy(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = astype(argmax(Y_hat, axis=1), Y.dtype)
        compare = astype(preds == reshape(Y, -1), float32)
        return reduce_mean(compare) if averaged else compare


    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return self.loss_func(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    
    
    def metric(self, batch, mode='train', averaged=True):
        Out, Y = self(*batch[:-1]), batch[-1]
        loss = self.loss(Out, Y)
        # Logging
        self.log_dict({f"{mode}_loss":loss, 
                       f"{mode}_acc":self.accuracy(Out, Y)}, 
                       prog_bar=True, 
                       on_epoch=True)
        return loss

    def training_step(self, batch):
        return self.metric(batch, 'train')

    def validation_step(self, batch):
        # if self.global_step%100==0:
        #     # log 6 example images
        #     x, y = batch
        #     sample_imgs, truth_labels = x[:8], y[:8]
        #     axes = self.visualize(sample_imgs, truth_labels)
        #     self.logger.experiment.add_figure('example_images: predicted/ground_truth', axes)
        return self.metric(batch, 'val')
    
    
    # def visualize(self, X, ground_truth, nrows=1, ncols=8):
    #     Y_hat = torch.argmax(self(X), dim=1)
    #     labels = DATAMODULE.classes
    #     Y_hat = [labels[int(i)] for i in Y_hat]
    #     ground_truth = [labels[int(i)] for i in ground_truth]
    #     return show_images(X.squeeze(1).cpu(), nrows, ncols, 
    #                        titles=[f"{pred}\n{truth}" for pred, truth in zip(Y_hat, ground_truth)])
    
    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

class DataModule(LightningDataModule):
    def __init__(self, dataset_class, batch_size: int = 32, transforms=None):
        super().__init__()
        
        self.train = dataset_class(os.getcwd(), train=True, download=True, transform=transforms)
        self.val = dataset_class(os.getcwd(), train=False, download=True, transform=transforms)
       
        self.batch_size = batch_size
        self.text_labels = dataset_class.classes

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=7)


class Pipline:

    @staticmethod
    def flow(sender=None, app_data=None, data_node=None, fake=False):
        assert data_node
        try:
            for model_init in data_node._output_attributes[0]._children:
                self = Pipline(data_node)
                self.collect_layers(model_init._data)
                self.train(fake)
        except BaseException as err:
            raise LookupError("Error in flow")
        return self
    
    @staticmethod
    def keep_train(sender=None, app_data=None, data_node=None):
        self = data_node.train_params.pipline
        if self:
            try:
                self.max_epochs = Pipline.get_params(data_node.train_params)['Max Epoches']
                self.train()
            except BaseException as err:
                raise RuntimeError("Error in keep training")
        else:
            Pipline.flow(data_node=data_node)
        
        

    def __init__(self, init_node):
        self.pipline = [Pipline.init_dataloader(init_node)]
        init_node.train_params.set_pipline(self)
        self.train_params = Pipline.get_params(init_node.train_params)
        self.task = Task.init(
                    project_name=self.train_params.pop("Project name"),
                    task_name=self.train_params.pop("Task name")
                )
        print("\n\n\ttrain params: ", self.train_params)
        self.task.connect(self.train_params)

        self.progress_board: list = init_node._output_attributes[1]._children
        self.progress_board: Union[ProgressBoard, None] = ProgressBoard(self.task, self.progress_board[0]._data) \
                                                                    if len(self.progress_board) else None
        
    @staticmethod
    def init_normal(module: nn.Module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, mean=0, std=0.01)
            nn.init.zeros_(module.bias)

    @staticmethod
    def init_xavier(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(module.weight)
    
    @staticmethod
    def init_dataloader(data_node: Node):
        params = dict()
        for param in data_node._params:
            params[param._label] = dpg.get_value(param.uuid)
        transforms = v2.Compose([v2.Resize((28, 28)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        return DataModule(data_node._data, batch_size=params.pop('batch_size'), transforms=transforms)


    @staticmethod
    def init_layer(layer: Node) -> any:
        params = dict()
        if len(layer._params):
            init_dict = dict()
            for param in layer._params:
                key = param._label
                if key == 'Initialization':
                    init_dict = dpg.get_item_user_data(param.uuid)
                params[key] = dpg.get_value(param.uuid)
            
            if 'Initialization' in params.keys() and (init_func := init_dict[params.pop("Initialization")]):
                return layer._data(**params).apply(init_func)
            return layer._data(**params)
        return layer._data()
        

    @staticmethod
    def get_params(params_node) -> dict:
        train_params = dict()
        for param in params_node._params:
            if param._label == 'Save Weights':
                break

            match param._type:
                case 'combo':
                    choices = dpg.get_item_user_data(param.uuid)
                    train_params[param._label] = choices[dpg.get_value(param.uuid)]
                case 'text/tuple':
                    train_params[param._label] = tuple(dpg.get_value(param.uuid))
                case 'int' | 'float' | 'text':
                    train_params[param._label] = dpg.get_value(param.uuid)
                case _:
                    print("\n\t\t\tUnexpected error!")
        return train_params
    
    
    @staticmethod
    def save_weight(sender, app_data, train_params__file: tuple[2]):
        train_params, filepath_uuid = train_params__file
        self = train_params.pipline
        if self:
            filepath = dpg.get_value(filepath_uuid)
            torch.save(self.net.state_dict(), filepath)

    @staticmethod
    def load_weight(sender, app_data, train_params__file: tuple[2]):
        train_params, filepath_uuid = train_params__file
        self: Pipline = train_params.pipline
        if not self:
            data_node = dpg.get_item_user_data(
                            dpg.get_item_parent(train_params._input_attributes[0]._parent.uuid))  
            self = Pipline.flow(data_node=data_node, fake=True)
            
        filepath = dpg.get_value(filepath_uuid)
        try:
            self.net.load_state_dict(torch.load(filepath))
        except BaseException as err:
            raise FileNotFoundError("Файл параметров не найден")
        
    

    def collect_layers(self, node: Node):
        self.pipline.append(Pipline.init_layer(node))
        
        while len(node := node._output_attributes[0]._children):
            node = node[0]._data
            self.pipline.append(Pipline.init_layer(node))

        self.max_epochs = self.train_params.pop('Max Epoches')
        self.net = Module(sequential=nn.Sequential(*self.pipline[1:]), optimizer=self.train_params.pop('Optimizer'),
                          lr=self.train_params.pop('Learning Rate'), loss_func=self.train_params.pop('Loss'))
        print("pipline: ", self.net)
        

    def train(self, fake=False):
        if self.progress_board:
            self.trainer = Trainer(max_epochs=1 if fake else self.max_epochs, accelerator='gpu', 
                                   callbacks=[self.progress_board])
        else:
            self.trainer = Trainer(max_epochs=1 if fake else self.max_epochs, accelerator='gpu')
        self.trainer.fit(self.net, self.pipline[0])

