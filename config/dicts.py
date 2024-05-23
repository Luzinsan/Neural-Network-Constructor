from torch import nn


dicts = {
    "LazyLinear":nn.LazyLinear,
    "LazyBatchNorm1d":nn.LazyBatchNorm1d,
    "LazyBatchNorm2d":nn.LazyBatchNorm2d,
    "LazyBatchNorm3d":nn.LazyBatchNorm3d,
    "LazyConv1d":nn.LazyConv1d,
    "LazyConv2d":nn.LazyConv2d,
    "LazyConv3d":nn.LazyConv3d,
    "BatchNorm1d":nn.BatchNorm1d,
    "BatchNorm2d":nn.BatchNorm2d,
    "BatchNorm3d":nn.BatchNorm3d,
    "Flatten":nn.Flatten,
    "AvgPool2d":nn.AvgPool2d,
    "MaxPool2d":nn.MaxPool2d,
    "AdaptiveAvgPool2d":nn.AdaptiveAvgPool2d,
    "Dropout":nn.Dropout,
    "ReLU":nn.ReLU,
    "Softmax":nn.Softmax,
    "Tanh":nn.Tanh,
    "GELU":nn.GELU,
    
}
