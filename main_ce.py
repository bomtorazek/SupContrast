from __future__ import print_function

import os
import sys
import argparse
import time
import math
import csv
from numpy.core.arrayprint import _void_scalar_repr

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms, datasets
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image

from datasets.general_dataset import GeneralDataset
from util import get_transform, rand_bbox, bbox2
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, best_accuracy
from util import set_optimizer, save_model
from util import denormalize
# from torchsampler import ImbalancedDatasetSampler
from util import load_image_names
from networks.resnet_big import SupCEResNet
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
from config import parse_option

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
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        scale = (0.875, 1.)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = get_transform(opt=opt, mean=mean, std=std, scale=scale)

    test_transform = val_transform = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.ToTensor(),
        normalize,
    ])
    
    custom = False
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        custom = True
        train_names_T, val_names_T, test_names_T = load_image_names(opt.data_folder, opt.train_util_rate,opt)
        
        train_dataset_T = GeneralDataset(data_dir=opt.data_folder, image_names=train_names_T,
                                        transform=train_transform)
        val_dataset_T = GeneralDataset(data_dir=opt.data_folder, image_names=val_names_T,
                                        transform=val_transform,)
        test_dataset_T = GeneralDataset(data_dir=opt.data_folder, image_names=test_names_T,
                                        transform=test_transform)

    if custom:
        train_loader_T = torch.utils.data.DataLoader(
            train_dataset_T, batch_size=opt.batch_size, shuffle= True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
        val_loader_T = torch.utils.data.DataLoader(
            val_dataset_T, batch_size=opt.batch_size, shuffle= False,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
        test_loader_T = torch.utils.data.DataLoader(
            test_dataset_T, batch_size=opt.batch_size, shuffle= False,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)

        return { 'train': train_loader_T, 'val': val_loader_T, 'test': test_loader_T}


    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=8, pin_memory=True)

        return { 'train': train_loader, 'val': val_loader}


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.num_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, CUTMIX = False,cam = None):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    if 'cut' in opt.aug:
        CUTMIX = True
        spt = opt.aug.split('_')
        cutmix_prob = float(spt[1])
        m = torch.nn.Softmax(dim=1)
        

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)


        # compute loss
        r = np.random.rand(1)
        if CUTMIX and r < cutmix_prob:

            ng_imgs = images[labels == 1]
            grayscale_cam= cam(input_tensor=ng_imgs, target_category=1) # (n_ng,h,w) numpy 0~1
            cam_mask = grayscale_cam >= 0.7 #  (n_ng,h,w)

            ng_outputs = model(ng_imgs) # FIXME double forwards
            soft_output = m(ng_outputs) # n_ng, 2

            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size()[0]).cuda()
            labels_a = labels
            labels_b = labels[rand_index]
            cutmix_labels = torch.zeros_like(labels)

            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            ng_iidx = [iid for iid in range(len(labels)) if labels[iid] == 1 ] # ng인 애들 

            for iidx in range(bsz):
                iidx_b = rand_index[iidx]
                if labels_a[iidx] == 0:
                    if labels_b[iidx] == 0: # ok <- ok: ok, random box of cutmix
                        pass
                        images[iidx, :, bbx1:bbx2, bby1:bby2] = images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                    elif labels_b[iidx] == 1: # ok <- ng: to ng, cut defect-region
                        ng_id = ng_iidx.index(iidx_b) # iidx_b가 ng_iidx에서 몇 번째인지
                        if soft_output[ng_id, 1] >= 0.9:
                            bbx1, bbx2, bby1, bby2 = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                            images[iidx, :, bbx1:bbx2, bby1:bby2] = images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                            cutmix_labels[iidx] = 1
                    else:
                        raise ValueError("fucked label")
                elif labels_a[iidx] == 1: 
                    if labels_b[iidx] == 0:  # ng <- ok: ng, cut non-defect-region
                        ng_id = ng_iidx.index(iidx)
                        if soft_output[ng_id, 1] >= 0.9:
                            bbs = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam, bbs) # 겹치면 빼기
                            images[iidx, :, bbx1:bbx2, bby1:bby2] = images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                        cutmix_labels[iidx] = 1 

                    elif labels_b[iidx] == 1: # ng <- ng: ng, cut defect-region (box)
                        ng_id = ng_iidx.index(iidx_b) # real_iidx가 ng_idx에서 몇 번째인지
                        if soft_output[ng_id, 1] >= 0.9:
                            bbx1, bbx2, bby1, bby2 = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                            images[iidx, :, bbx1:bbx2, bby1:bby2] = images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                        cutmix_labels[iidx] = 1

                    else:
                        raise ValueError("fucked label")
                else:
                    raise ValueError("fucked label")
            # ROI는 곱하기로 alpha blending 하면 된다.

        output = model(images)
        if CUTMIX and r < cutmix_prob:
            loss = criterion(output, cutmix_labels)
        else:
            loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc2 = accuracy(output, labels, topk=(1, 2))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx == len(train_loader)-1:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    probs = []
    gts = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            gts.extend(labels.tolist())
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            prob = torch.nn.functional.softmax(output, dim=1)[:,1]
            probs.extend(prob.tolist())
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            acc1, acc2 = accuracy(output, labels, topk=(1, 2))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx == len(val_loader)-1:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    gts = np.array(gts)
    probs = np.array(probs)
    auc = roc_auc_score(gts, probs)
    best_acc, best_th = best_accuracy(gts,probs)
    print('Val auc: {:.3f}'.format(auc), end = ' ')
    print('Val acc: {:.3f}'.format(best_acc) )
    
    return losses.avg, auc, best_acc, best_th

def test(test_loader, model,  opt, best_th = None):
    model.eval()

    probs = []
    gts = []

    if best_th is None:
        pretrained_dict = torch.load(os.path.join(
                opt.save_folder, 'auc_best.pth'))['model']
        model.load_state_dict(pretrained_dict)
    else:
        pretrained_dict = torch.load(os.path.join(
                opt.save_folder, 'acc_best.pth'))['model']
        model.load_state_dict(pretrained_dict)


    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            gts.extend(labels.tolist())
            labels = labels.cuda()

            # forward
            output = model(images)
            prob = torch.nn.functional.softmax(output, dim=1)[:,1]
            probs.extend(prob.tolist())
            

    gts = np.array(gts)
    probs = np.array(probs)

    if best_th is None:
        auc = roc_auc_score(gts, probs)
        return auc
    else:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(gts, probs>=best_th)
        return acc 


def main():
    best_epoch = 0
    best_auc = 0.0
    best_acc = 0.0
    opt = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # build data loader
    loaders = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)


    cam = None
    if 'cut' in opt.aug:
        cam_method = opt.aug.split('_')[2]
        if cam_method == 'PP':
            cam = GradCAMPlusPlus(model=model.module, target_layer=model.module.encoder.layer4[-1], use_cuda=True) # module if dataparallel
        elif cam_method == 'AB':
            cam = AblationCAM(model=model.module, target_layer=model.module.encoder.layer4[-1], use_cuda=True) 
        elif cam_method == 'EI':
            cam = EigenCAM(model=model.module, target_layer=model.module.encoder.layer4[-1], use_cuda=True) 
        else:
            raise NotImplementedError("not supported type of cam method")

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(loaders['train'], model, criterion, optimizer, epoch, opt, cam=cam)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_auc, val_acc, val_th = validate(loaders['val'], model, criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_auc', val_auc, epoch)
        logger.log_value('val_acc', val_acc, epoch)
        logger.log_value('val_th', val_th, epoch)


        if val_auc > best_auc:
            best_epoch = epoch
            best_auc = val_auc
            save_file = os.path.join(
                opt.save_folder, 'auc_best.pth')
            save_model(model, optimizer, opt, epoch, save_file)
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_th = val_th
            save_file = os.path.join(
                opt.save_folder, 'acc_best.pth')
            save_model(model, optimizer, opt, epoch, save_file)
        if epoch >= best_epoch + opt.patience:
            break
    # # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    best_auc = test(loaders['test'], model, opt)
    best_acc = test(loaders['test'], model, opt, best_th)
    print('Test auc: {:.2f}'.format(best_auc), end = ' ')
    print('Test acc: {:.2f}'.format(best_acc) ,end = ' ')
    print('Test th: {:.2f}'.format(best_th) ,end = ' ')

    with open("result.csv", "a") as file:
        writer = csv.writer(file)
        row = [opt.model_name, 'auc', best_auc, 'acc', best_acc, 'th', best_th]
        writer.writerow(row)




if __name__ == '__main__':
    main()
