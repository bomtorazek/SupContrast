"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from modules.networks.resnet_dsbn import resnet18dsbn, resnet34dsbn, resnet50dsbn, resnet101dsbn


def resnet18(**kwargs):
    return models.resnet18(pretrained=True)
   

def resnet34(**kwargs):
    return models.resnet34(pretrained=True)
 

def resnet50(**kwargs):
    return models.resnet50(pretrained=True)


def resnet101(**kwargs):
    return models.resnet101(pretrained=True)
 

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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

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
        self.encoder = model_fun()
        self.encoder.fc = Identity()
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
        self.fc = nn.Linear(dim_in, num_classes)


    def forward(self, x, domain_idx = None):
        if domain_idx is not None:
            feat = self.encoder(x, domain_idx)
        else:
            feat = self.encoder(x)
        fc = self.fc(feat) # feat.detach()
        feat = F.normalize(self.head(feat), dim=1)
        return feat, fc
        


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.encoder.fc = Identity()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x,  domain_idx = None):
        if domain_idx is not None:
            return self.fc(self.encoder(x, domain_idx))
        else:
            return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
