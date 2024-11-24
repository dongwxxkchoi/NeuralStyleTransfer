import torch
import torch.nn as nn
import torch.nn.functional as F

# ContentLoss -> MSE between F / P
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        loss = F.mse_loss(x,y)
        return loss

# StyleLoss -> gram matrix
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, x: torch.Tensor):
        """
        x: torch.Tensor, shape(b,c,h,w)
        reshape(b,c,h,w) -> (b,c,h*w)
        dim (b, N, M)
        transpose
        matrix multiplication
        """

        b,c,h,w, = x.size()
        # reshape
        features = x.view(b, c, h*w) # (b, N, M)
        features_T = features.transpose(1, 2) # (b, M, N)
        # matrix multiplication
        G = torch.matmul(features, features_T)

        return G.div(4*(c**2)*(h*w)**2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        
        return loss


# TotalLoss -> train에서 설정