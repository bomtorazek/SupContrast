import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from collections import OrderedDict
import operator
from itertools import islice

from modules.networks.dsbn import DomainSpecificBatchNorm2d


_pair = _ntuple(2)

__all__ = ['resnet18dsbn', 'resnet34dsbn', 'resnet50dsbn', 'resnet101dsbn', 'resnet152dsbn']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label

class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2

    
def resnet18dsbn(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet18']),
                                                          num_classes=2,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def resnet34dsbn(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet34']),
                                                          num_classes=2,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model

def resnet50dsbn(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet50']),
                                                          num_classes=2,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def resnet101dsbn(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet101']),
                                                          num_classes=2,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def resnet152dsbn(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet152']),
                                                          num_classes=2,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def _update_initial_weights_dsbn(state_dict, num_classes=2, num_domains=2, dsbn_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and dsbn_type == 'all') or
                (('bn1' in key) and dsbn_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_domains):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict


class DSBNResNet(nn.Module):
    def __init__(self, block, layers, num_domains=2, zero_init_residual=False):
        self.inplanes = 64
        self.num_domains = num_domains
        super(DSBNResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(64, self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_domains=self.num_domains)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_domains=self.num_domains)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_domains=self.num_domains)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, num_domains=self.num_domains)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, num_domains=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                DomainSpecificBatchNorm2d(planes * block.expansion, num_domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_domains=num_domains))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, num_domains=num_domains))

        return TwoInputSequential(*layers)

    def forward(self, x, domain_label):
        x = self.conv1(x)
        x, _ = self.bn1(x, domain_label)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer1(x, domain_label)
        x, _ = self.layer2(x, domain_label)
        x, _ = self.layer3(x, domain_label)
        x, _ = self.layer4(x, domain_label)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ## ---- no fc

        return x

        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = DomainSpecificBatchNorm2d(planes * 4, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label