import torchvision.datasets as ds # type: ignore
from torchvision.transforms import v2 # type: ignore
import re
from torch import nn
import torch
from torch.functional import F
import pdb

class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = nn.ReLU()(self.b1_1(x))
        b2 = nn.ReLU()(self.b2_2(nn.ReLU()(self.b2_1(x))))
        b3 = nn.ReLU()(self.b3_2(nn.ReLU()(self.b3_1(x))))
        b4 = nn.ReLU()(self.b4_2(self.b4_1(x)))
        print(b1.shape,b2.shape,b3.shape,b4.shape)
        return torch.cat((b1, b2, b3, b4), dim=1)
    
# pdb.set_trace()
inc = Inception(64, (96, 128), (16, 32), 32)
res = inc.forward(torch.rand((1,1,96,96)))
print(res.shape)