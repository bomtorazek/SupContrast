from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import os.path as osp
from sklearn.metrics import accuracy_score
from RandAugment import rand_augment_transform
import torch.nn as nn


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_transform(opt, mean, std, scale):

    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.aug.lower() == 'sim':
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
    ])
    elif 'stacked' in opt.aug.lower():
        # stacked_ra_3_5
        spt = opt.aug.split('_')
        n = spt[-2]
        m = spt[-1]
        int(n)
        int(m)
        ra_params = dict(
            translate_const=int(opt.size * 0.125),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(opt.size//10)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(n, m),
                                   ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif 'rand' in opt.aug.lower():
        spt = opt.aug.split('_')
        n = spt[-2]
        m = spt[-1]
        int(n)
        int(m)
        ra_params = dict(
            translate_const=int(opt.size * 0.125),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=scale),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(n, m),
                                    ra_params),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError("Unsupported augmentaton name")

    return TF

def load_image_names(data_dir, util_rate, opt):
        imageset_dir = osp.join(data_dir, 'imageset/single_image.2class/fold.5-5/ratio/100%')

        with open(osp.join(imageset_dir, 'train.{}-{}.txt'.format(opt.test_fold, opt.val_fold)), 'r') as fid:
            temp = fid.read()
        train_names = temp.split('\n')[:-1]
        with open(osp.join(imageset_dir, 'validation.{}-{}.txt'.format(opt.test_fold, opt.val_fold)), 'r') as fid:
            temp = fid.read()
        val_names = temp.split('\n')[:-1]
        with open(osp.join(imageset_dir, 'test.{}.txt'.format(opt.test_fold)), 'r') as fid:
            temp = fid.read()
        test_names = temp.split('\n')[:-1]
        
        if util_rate < 1:
            num_used = int(len(train_names) * util_rate)
            np.random.seed(1)
            train_names = np.random.choice(train_names, size=num_used, replace=False)

        return train_names, val_names, test_names


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    if opt.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(),
                            lr=opt.learning_rate,
                            weight_decay=opt.weight_decay)
    
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    else:
        raise ValueError("Not supported optimizer")
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def best_accuracy(gts, probs):
    best_th = 0.0
    best_acc = 0.0
    for th in range(0,200): 
        th = th/200.0
        acc = accuracy_score(gts, probs>=th)

        if acc > best_acc:
            best_acc = acc
            best_th = th
    
    return best_acc, best_th