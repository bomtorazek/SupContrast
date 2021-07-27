import time
import sys, os

import torch
from torch.nn.functional import normalize
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

from util import AverageMeter, warmup_learning_rate, accuracy, best_accuracy


def train(trainloader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_CE = AverageMeter()
    top1 = AverageMeter()

    if opt.method == 'Joint_Con':
        losses_Con = AverageMeter()
    end = time.time()


    for idx, (images, labels) in enumerate(trainloader):   
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        if 'CE' in opt.method:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            output = model(images)
            loss_CE = criterion(output, labels)
            
        elif opt.method == 'Joint_Con':
            
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)

            # if CUTMIX:
            #     r = np.random.rand(1)
            #     if r < cutmix_prob:
            #         images[0], images[1], labels = make_cutmix_ext(images=images[0], labels=labels, model=model,
            #                                         cam=cam, m=m, bsz=bsz, epoch=epoch, ext_images=images[1], type_=cutmix_type)

            images = torch.cat([images[0], images[1]], dim=0)
            labels = labels.cuda(non_blocking=True)
            labels_aug = torch.cat([labels, labels], dim=0)
            
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)


            # compute loss
            features, output = model(images)
            if opt.head == 'mlp':
                f1_T, f2_T = torch.split(features, [bsz, bsz], dim=0)
            elif opt.head == 'fc':
                f1_T, f2_T = torch.split(normalize(output,dim=1), [bsz, bsz], dim=0)
            features_T = torch.cat([f1_T.unsqueeze(1), f2_T.unsqueeze(1)], dim=1)
            
            loss_Con = criterion['Con'](features_T, labels)
            loss_CE = criterion['CE'](output, labels_aug)
        else:
            raise ValueError("check method")  

        # update metric
        losses_CE.update(loss_CE.item(), bsz)
        if opt.method == 'Joint_Con':
            losses_Con.update(loss_Con.item(), bsz)
        else:
            raise ValueError("check method")  
        acc1, _ = accuracy(output[:bsz,:], labels[:bsz], topk=(1, 2))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        total_loss = loss_CE
        if opt.method == 'Joint_Con':
            total_loss = total_loss*opt.l_ce + loss_Con 
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx == len(trainloader)-1:
        # if idx % 1 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses_CE))
            sys.stdout.flush()
    if opt.method == 'Joint_Con':
        return {'CE':losses_CE.avg, 'Con': losses_Con.avg}, top1.avg
    elif 'CE' in opt.method:
        return losses_CE.avg, top1.avg
    else:
        raise ValueError("check method")  




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
            if opt.method == 'Joint_Con':
                output = output[1]
            prob = torch.nn.functional.softmax(output, dim=1)[:,1]
            probs.extend(prob.tolist())
            
            if opt.method == 'Joint_Con':
                loss = criterion['CE'](output, labels)
            elif 'CE' in opt.method:
                loss = criterion(output, labels)
            else:
                raise ValueError("check method")  
                
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
    f1 = f1_score(gts, probs>=0.5)
    best_acc, best_th = best_accuracy(gts,probs)
    acc05 = accuracy_score(gts, probs>=0.5)
    
    print('Val auc: {:.3f}'.format(auc), end = ' ')
    print('Val bacc: {:.3f}'.format(best_acc), end = ' ')
    print('Val f1: {:.3f}'.format(f1) )
    print('Val acc: {:.3f}'.format(acc05) )
    
    return losses.avg, auc, best_acc, best_th, acc05, f1



def test(test_loader, model,  opt, metric=None, best_th = None):
    model.eval()

    probs = []
    gts = []

    if best_th is None:
        if metric == 'auc':
            pretrained_dict = torch.load(os.path.join(
                    opt.save_folder, 'auc_best.pth'))['model']
            model.load_state_dict(pretrained_dict)
        elif metric == 'acc':
            pretrained_dict = torch.load(os.path.join(
                    opt.save_folder, 'acc05_best.pth'))['model']
            model.load_state_dict(pretrained_dict)
        elif metric == 'f1':
            pretrained_dict = torch.load(os.path.join(
                    opt.save_folder, 'f1_best.pth'))['model']
            model.load_state_dict(pretrained_dict)
        elif metric == 'last':
            pretrained_dict = torch.load(os.path.join(
                    opt.save_folder, 'last.pth'))['model']
            model.load_state_dict(pretrained_dict)
        else:
            raise ValueError("not supported metric")
    else:
        pretrained_dict = torch.load(os.path.join(
                opt.save_folder, 'bacc_best.pth'))['model']
        model.load_state_dict(pretrained_dict)


    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            gts.extend(labels.tolist())
            labels = labels.cuda()

            # forward
            output = model(images)
            if opt.method == 'Joint_Con':
                output = output[1] 
            prob = torch.nn.functional.softmax(output, dim=1)[:,1]
            probs.extend(prob.tolist())
            

    gts = np.array(gts)
    probs = np.array(probs)

    if best_th is None:
        if metric == 'auc':
            auc = roc_auc_score(gts, probs)
            return auc
        elif metric == 'acc':
            acc = accuracy_score(gts, probs>=0.5)
            return acc
        elif metric == 'f1':
            return f1_score(gts, probs>=0.5)
        elif metric == 'last':
            auc = roc_auc_score(gts, probs)
            acc = accuracy_score(gts, probs>=0.5)
            f1 = f1_score(gts, probs>=0.5)
            return auc, acc, f1
            
        else:
            raise ValueError("unsupported metric")
    else:
        bacc = accuracy_score(gts, probs>=best_th)
        best_acc, best_th = best_accuracy(gts,probs)
        return bacc, best_acc, best_th