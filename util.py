from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import os.path as osp
from sklearn.metrics import accuracy_score




class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

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