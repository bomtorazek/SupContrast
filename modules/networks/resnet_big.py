"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.networks.resnet_dsbn import resnet18dsbn, resnet34dsbn, resnet50dsbn, resnet101dsbn
from modules.networks.resnet import resnet18, resnet34, resnet50, resnet101


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet18dsbn': [resnet18dsbn, 512],
    'resnet34dsbn': [resnet34dsbn, 512],
    'resnet50dsbn': [resnet50dsbn, 2048],
    'resnet101dsbn': [resnet101dsbn, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupHybResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, num_classes=2):
        super(SupHybResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(pretrained=True, num_classes=num_classes)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        fc = self.encoder.fc(feat) # feat.detach()
        feat = F.normalize(self.head(feat), dim=1)
        return feat, fc
        
class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, _ = model_dict[name]
        self.encoder = model_fun(pretrained=True, num_classes=num_classes)
 
    def forward(self, x):
        return self.encoder.fc(self.encoder(x))

