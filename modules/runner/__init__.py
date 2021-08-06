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
    mask_prob = AverageMeter()

    if opt.method == 'Joint_Con':
        losses_Con = AverageMeter()
    end = time.time()


    for idx, (images, labels, domain_idx) in enumerate(trainloader):   
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        if opt.method == 'CE':
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            output = model(images)
            loss_CE = criterion(output, labels)
            effective_bsz = bsz
            
        elif opt.method == 'Joint_Con':
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)
            t_mask = domain_idx == 0
            s_mask = domain_idx == 1
            if epoch >= opt.pseudo_epoch:
                target_images0 = images[0][t_mask]
                target_images1 = images[1][t_mask]
                source_images0 = images[0][s_mask]
                source_images1 = images[1][s_mask]

                t_num = target_images0.shape[0]
                assert target_images0.shape[0] + source_images0.shape[0] == bsz

                images = torch.cat([target_images0, source_images0, target_images1, source_images1], dim=0)
                features, output = model(images)
                features0, features1 = features.chunk(2)
                output0, output1 = output.chunk(2)

                # target_output 사용해 pseudo label 생성, 두 view에 대해 같고 threshold를 넘어야
                pseudo_soft0 = torch.softmax(output0[:t_num].detach(), dim=-1) 
                pseudo_soft1 = torch.softmax(output1[:t_num].detach(), dim=-1) 
                max_probs0, pseudo_label0 = torch.max(pseudo_soft0, dim=-1)
                max_probs1, pseudo_label1 = torch.max(pseudo_soft1, dim=-1)
                
                mask0 = max_probs0.ge(opt.pseudo_threshold)
                mask1 = max_probs1.ge(opt.pseudo_threshold)
                label_mask = pseudo_label0 == pseudo_label1
                mask = (mask0 * mask1 * label_mask)
                pseudo_label = pseudo_label0[mask]
                
                # source_label + pseudo label
                labels=labels.cuda(non_blocking=True)
                labels = torch.cat([labels[s_mask], pseudo_label], dim=0)
                # source_output + 살아남은 output
                new_output0 = torch.cat([output0[t_num:],output0[:t_num][mask]], dim=0)
                new_output1 = torch.cat([output1[t_num:],output1[:t_num][mask]], dim=0)
                output = torch.cat([new_output0, new_output1], dim=0)
                # source_feature + 살아남은 feature
                new_features0 = torch.cat([features0[t_num:],features0[:t_num][mask]], dim=0)
                new_features1 = torch.cat([features1[t_num:],features1[:t_num][mask]], dim=0)
                features = torch.cat([new_features0, new_features1], dim=0)

                assert features.shape[0] == output.shape[0]
                assert features.shape[0] == 2*labels.shape[0]


            else:
                source_images0 = images[0][s_mask]
                source_images1 = images[1][s_mask]
                labels = labels[domain_idx == 1]

                images = torch.cat([source_images0, source_images1], dim=0)
                labels = labels.cuda(non_blocking=True)
                features, output = model(images)
                mask = None

            labels_aug = torch.cat([labels, labels], dim=0)
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            f1_T, f2_T = features.chunk(2)

            features_T = torch.cat([f1_T.unsqueeze(1), f2_T.unsqueeze(1)], dim=1)
            
            loss_Con = criterion['Con'](features_T, labels)
            loss_CE = criterion['CE'](output, labels_aug)
            assert output.shape[0] %2 ==0
            effective_bsz = int(output.shape[0]/2)
        else:
            raise ValueError("check method")  

        # update metric
        if mask is not None:
            mask_prob.update(mask.float().mean().item(), bsz)
        losses_CE.update(loss_CE.item(), effective_bsz)
        if opt.method == 'Joint_Con':
            losses_Con.update(loss_Con.item(), effective_bsz)
        elif 'CE' not in opt.method:
            raise ValueError("check method")  
        acc1, _ = accuracy(output[:effective_bsz,:], labels[:effective_bsz], topk=(1, 2))
        top1.update(acc1[0], effective_bsz)

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
        return {'CE':losses_CE.avg, 'Con': losses_Con.avg}, top1.avg, mask_prob.avg
    elif 'CE' in opt.method:
        return losses_CE.avg, top1.avg, None
    else:
        raise ValueError("check method")  




def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            if opt.method == 'Joint_Con':
                output = output[1]
            
            if opt.method == 'Joint_Con':
                loss = criterion['CE'](output, labels)
            elif 'CE' in opt.method:
                loss = criterion(output, labels)
            else:
                raise ValueError("check method")  
                
            # update metric
            losses.update(loss.item(), bsz)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx == len(val_loader)-1:
                print('Val: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

    
    print('Val acc: {:.3f}'.format(top1.avg) )
    
    return losses.avg, top1.avg, top5.avg

