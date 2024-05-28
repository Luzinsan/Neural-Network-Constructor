from lightning import LightningModule
import torch
from torch import nn
import pdb
from core.utils import send_message


reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
float32 = torch.float32

class MultiBranch(nn.Module):

    def __init__(self, multi_branch, **kwargs):
        super(MultiBranch, self).__init__()
        self.multi_branch = multi_branch
        for key, value in kwargs.items():
            setattr(self, key, value)


    @staticmethod
    def sub_forward(branch, x):
        if isinstance(branch, list):
            res = MultiBranch.sub_forward(branch[0], x)
            for module in branch[1:]:
                res = MultiBranch.sub_forward(module, res)
            return res
        else:
            return branch(x)

    def forward(self, x):
        branchs = []
        for branch in self.multi_branch:
            branchs.append(MultiBranch.sub_forward(branch, x))
        return torch.cat(branchs, dim=self.dim)

class Module(LightningModule):
    
    debug=True
    accept_init = [nn.Linear, nn.Conv2d]
    
    def __init__(self, sequential, optimizer, lr, loss_func):
        super(Module, self).__init__()
        self.net = sequential
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func
        print('\n\t\t\tНеинициализированная сеть: ', self.net) if Module.debug else None
        

    def apply_init(self, data_source, init):
        inputs = [next(iter(data_source.train_dataloader()))[0]]
    
        if init is not None:
            self.net.apply(init)
            self.forward(*inputs)
            self.net.apply(init)


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
        return self.metric(batch, 'val')
    
    
    # def visualize(self, X, ground_truth, nrows=1, ncols=8):
    #     Y_hat = torch.argmax(self(X), dim=1)
    #     labels = DATAMODULE.classes
    #     Y_hat = [labels[int(i)] for i in Y_hat]
    #     ground_truth = [labels[int(i)] for i in ground_truth]
    #     return show_images(X.squeeze(1).cpu(), nrows, ncols, 
    #                        titles=[f"{pred}\n{truth}" for pred, truth in zip(Y_hat, ground_truth)])
    
    def layer_summary(self, X_shape):
        print("Individual input shape:\t", *X_shape)
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
