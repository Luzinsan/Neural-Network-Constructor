# from . import np, torch, nn, d2l

# class LinearRegression(d2l.Module):
#     """The linear regression model implemented with high-level APIs."""
#     def __init__(self, lr):
#         super().__init__()
#         d2l.LinearRegression
#         self.save_hyperparameters()
#         self.net = nn.LazyLinear(1)
#         self.net.weight.data.normal_(0, 0.01)
#         self.net.bias.data.fill_(0)
    
#     def loss(self, y_hat, y):
#         fn = nn.MSELoss()
#         return fn(y_hat, y)

#     def configure_optimizers(self):
#         return torch.optim.SGD(self.parameters(), self.lr)

#     def get_w_b(self):
#         return (self.net.weight.data, self.net.bias.data)

#     def forward(self, X):
# 	    return self.net(X)
    
