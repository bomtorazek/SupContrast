from __future__ import print_function

import os
import sys
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.utils.data import dataset
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, SupHybResNet
from losses import SupConLoss, CrossSupConLoss
from config import parse_option
from datasets.general_dataset import GeneralDataset
from torchsampler import ImbalancedDatasetSampler
from util import load_image_names

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        scale = (0.2, 1.)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        scale = (0.2, 1.)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
        scale = (0.2, 1.)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        scale = (0.875, 1.)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale= scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = val_transform = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.ToTensor(),
        normalize,
    ])

    
    # dataset
    custom = False
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        train_names_S, _, _ = load_image_names(opt.source_data_folder, 1, opt)
        train_names_T, val_names_T, test_names_T = load_image_names(opt.data_folder, opt.train_util_rate,opt)
        
        train_dataset_S = GeneralDataset(data_dir=opt.source_data_folder, transform=TwoCropTransform(train_transform),
                                        image_names=train_names_S)
        train_dataset_T = GeneralDataset(data_dir=opt.data_folder, transform=TwoCropTransform(train_transform),
                                        image_names=train_names_T)
        val_dataset_T = GeneralDataset(data_dir=opt.data_folder, transform=val_transform,
                                        image_names=val_names_T)
        test_dataset_T = GeneralDataset(data_dir=opt.data_folder, transform=test_transform,
                                        image_names=test_names_T)
        custom = True

    # dataloader
    if custom:
        train_loader_S = torch.utils.data.DataLoader(
            train_dataset_S, batch_size=opt.batch_size, shuffle= True,
            num_workers=opt.num_workers, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset_S))
        train_loader_T = torch.utils.data.DataLoader(
            train_dataset_T, batch_size=opt.batch_size, shuffle= True,
            num_workers=opt.num_workers, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset_T))
        val_loader_T = torch.utils.data.DataLoader(
            val_dataset_T, batch_size=opt.batch_size, shuffle= False,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
        test_loader_T = torch.utils.data.DataLoader(
            test_dataset_T, batch_size=opt.batch_size, shuffle= False,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)

        return {'train_S': train_loader_S,  'train_T': train_loader_T, 'val': val_loader_T, 'test': test_loader_T}

    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model, num_classes=opt.num_cls) 
    
    if opt.method == 'Joint_Con':
        criterion = {}
        criterion['Cross'] = CrossSupConLoss(temperature=opt.temp)
        criterion['Con'] = SupConLoss(temperature=opt.temp)
        criterion['CE'] = torch.nn.CrossEntropyLoss() 
    elif opt.method == 'Joint_CE':
        criterion = torch.nn.CrossEntropyLoss()

    if opt.model_transfer is not None:
        pretrained_dict = torch.load(opt.model_transfer)['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
   
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        if opt.method == 'Joint_Con':
            criterion['CE']=criterion['CE'].cuda()
            criterion['Cross']=criterion['CE'].cuda()
            criterion['Con']=criterion['CE'].cuda()
        else:
            criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(loaders, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    loaders['train_S'] # iterator로 Late3D참고
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    # build data loader
    loaders = set_loader(opt) # tuple or just loader

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(loaders, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
