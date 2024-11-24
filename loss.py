import torch
import torch.nn as nn
import torch.nn.functional as F

# ContentLoss -> MSE between F / P
class ContentLoss(nn.Module):
    def __init__(self, x):
        super(ContentLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        loss = F.mse_loss(x,y)
        return loss

# StyleLoss -> gram matrix
class StyleLoss(nn.Module):
    def __init__(self, x):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        pass

# TotalLoss -> train에서 설정