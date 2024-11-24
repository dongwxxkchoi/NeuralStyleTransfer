import torch
import torch.nn as nn
from torchvision.models import vgg19

conv = {
    "conv1_1": 0, # style
    "conv2_1": 5, # style
    "conv3_1": 10, # style
    "conv4_1": 19, # style
    "conv5_1": 28, # style
    "conv4_2": 21, # content
}

class StyleTransfer(nn.Module):
    def __init__(self,):
        super(StyleTransfer, self).__init__()
        #TODO: VGG19 load
        self.vgg19_model = vgg19(weights="DEFAULT")
        self.vgg19_features = self.vgg19_model.features
        
        #TODO: separate conv layers
        self.style_layer = [conv['conv1_1'], conv['conv2_1'], conv['conv3_1'], conv['conv4_1'], conv['conv5_1']]
        self.content_layer = [conv['conv4_2']]
    
    def forward(self, x):
        pass