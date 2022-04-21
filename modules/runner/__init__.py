import time
import sys, os

import torch
from torch.nn.functional import normalize
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

from util import AverageMeter, warmup_learning_rate, accuracy, best_accuracy, mixup_data



def interleave(x, size): # for multi-gpu training, from kekmodel/FixMatch-pytorch
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):  # for multi-gpu training, from kekmodel/FixMatch-pytorch
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


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

    if opt.source_sampling is not None:
        soft = nn.Softmax(dim=1)
        sig = nn.Sigmoid()
        image_indices_list = []
        confidence_list = []
        target_sim_list = []

    for idx, samples in enumerate(trainloader):
        if opt.sampling == 'domainKang':
            images, labels, domain_tags, image_indices = samples
        else:
            images, labels = samples

        third = len(trainloader)//3
        if third !=0 and idx % third == 0:
            print(f'{idx}/{len(trainloader)}', end = ' ')

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
            
            if opt.one_crop:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                labels_aug = labels
            else:
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)
                images = torch.cat([images[0], images[1]], dim=0)
            
                # if CUTMIX:
                #     r = np.random.rand(1)
                #     if r < cutmix_prob:
                #         images[0], images[1], labels = make_cutmix_ext(images=images[0], labels=labels, model=model,
                #                                         cam=cam, m=m, bsz=bsz, epoch=epoch, ext_images=images[1], type_=cutmix_type)

                labels = labels.cuda(non_blocking=True)
                labels_aug = torch.cat([labels, labels], dim=0)
            
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)


            # compute loss
            if 'target_sim' in opt.source_sampling:
                features, output, domain_logit = model(images)
            else:
                features, output = model(images)

            if opt.one_crop:
                features_T = features.unsqueeze(1)
            else:
                f1_T, f2_T = torch.split(features, [bsz, bsz], dim=0)
                features_T = torch.cat([f1_T.unsqueeze(1), f2_T.unsqueeze(1)], dim=1)
            
            loss_Con = criterion['Con'](features_T, labels)
            loss_CE = criterion['CE'](output, labels_aug)
            if 'target_sim' in opt.source_sampling:
                domain_logit1, domain_logit2 = domain_logit.chunk(2)
                averaged_domain_logit = (domain_logit1 + domain_logit2)/2
                domain_confidence = sig(averaged_domain_logit)
                loss_domain = criterion['Domain'](domain_confidence.squeeze(1), domain_tags.type(torch.FloatTensor).cuda(non_blocking=True))

                # get indices and confidences only for source
                domain_confidence = [ conf.item() for conf_idx, conf in enumerate(domain_confidence) if domain_tags[conf_idx]==1 ]
                image_indices =  [image_idx.item() for ii, image_idx in enumerate(image_indices) if domain_tags[ii]==1]
                confidence_list.extend(domain_confidence) ## FIXME
                image_indices_list.extend(image_indices)

            if 'easy' in opt.source_sampling or 'hard' in opt.source_sampling:
                probs = soft(output) # 2*BS,2
                probs0, probs1 = probs.chunk(2)
                averaged_probs = (probs0 + probs1)/2 # BS,2
                confidence = [max(avg_prob).item() for avg_prob in averaged_probs]

                # get indices and confidences only for source
                confidence = [ conf for conf_idx, conf in enumerate(confidence) if domain_tags[conf_idx]==1 ]
                image_indices =  [image_idx.item() for ii, image_idx in enumerate(image_indices) if domain_tags[ii]==1]
                confidence_list.extend(confidence)
                image_indices_list.extend(image_indices)

        else:
            raise ValueError("check method")  

        # update metric
        losses_CE.update(loss_CE.item(), bsz)
        if opt.method == 'Joint_Con':
            losses_Con.update(loss_Con.item(), bsz)
        elif 'CE' not in opt.method:
            raise ValueError("check method")  
        acc1, _ = accuracy(output[:bsz,:], labels[:bsz], topk=(1, 2))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        total_loss = loss_CE
        if opt.method == 'Joint_Con':
            total_loss = total_loss + loss_Con *opt.l_con
            if 'target_sim' in opt.source_sampling:
                total_loss += loss_domain
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
        if opt.source_sampling is not None:
            conf_image_idx_list = [[conf, ii] for conf, ii in zip(confidence_list, image_indices_list) ]
            conf_image_idx_list.sort(key= lambda x: x[0])
            half_source_size = len(conf_image_idx_list)//2
            if opt.source_sampling == 'easy':
                conf_image_idx_list = conf_image_idx_list[half_source_size:]
            elif opt.source_sampling == 'hard':
                conf_image_idx_list = conf_image_idx_list[:half_source_size]
            elif opt.source_sampling == 'target_sim': ## FIXME
                conf_image_idx_list = conf_image_idx_list[:half_source_size]

            image_indices_list = [ii for _, ii in conf_image_idx_list]
            
            return {'CE':losses_CE.avg, 'Con': losses_Con.avg}, top1.avg, image_indices_list


        else:
            return {'CE':losses_CE.avg, 'Con': losses_Con.avg}, top1.avg
    elif 'CE' in opt.method:
        return losses_CE.avg, top1.avg
    else:
        raise ValueError("check method")  



def train_sampling(trainloader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_CE = AverageMeter()
    top1 = AverageMeter()

    if opt.method == 'Joint_Con':
        losses_Con = AverageMeter()
    end = time.time()

    trainloader_T = trainloader['target']
    trainloader_S = trainloader['source']

    iter_T = iter(trainloader_T)
    iter_S = iter(trainloader_S)

    for idx in range(len(trainloader_S)): # FIXME assume that # of source dataset is larger than # of target dataset
        try: 
            images_T, labels_T = iter_T.next()
        except:
            iter_T = iter(trainloader_T)
            images_T, labels_T = iter_T.next()

        try:
            images_S, labels_S = iter_S.next()
        except:
            iter_S = iter(trainloader_S)
            images_S, labels_S = iter_S.next()

        third = len(trainloader_S)//3
        if idx % third == 0:
            print(f'{idx}/{len(trainloader_S)}', end = ' ')
            

        data_time.update(time.time() - end)
        labels = torch.cat([labels_T, labels_S], dim=0)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        if opt.method == 'Joint_CE':
            images = torch.cat([images_T, images_S], dim=0)
            images = images.cuda(non_blocking=True)


            if opt.mixup:
                is_inter = True if opt.mixup == 'inter_class' else False
                images, labels_a, labels_b, lam = mixup_data(images, labels, inter_class=is_inter)

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            output = model(images)

            if opt.mixup:
                loss_CE = lam * criterion(output, labels_a) + (1 - lam) * criterion(output, labels_b)
            else:
                loss_CE = criterion(output, labels)
            
        elif opt.method == 'Joint_Con':
            if idx ==0:
                if opt.one_crop:
                    print(images_T.shape, 'target')
                    print(images_S.shape, 'source')
                else:    
                    print(images_T[0].shape, 'target')
                    print(images_S[0].shape, 'source')
    
        
            if opt.one_crop:
                images = torch.cat([images_T, images_S], dim=0).cuda(non_blocking=True)
                labels_aug = labels
            
            else:
                images0 = torch.cat([images_T[0], images_S[0]], dim=0)
                images1 = torch.cat([images_T[1], images_S[1]], dim=0)
                images = torch.cat([images0, images1], dim=0).cuda(non_blocking=True)
                labels_aug = torch.cat([labels, labels], dim=0)
            
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            features, output = model(images)

            if opt.one_crop:
                features = features.unsqueeze(1)
            else:
                f0, f1 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f0.unsqueeze(1), f1.unsqueeze(1)], dim=1)
            
            loss_Con = criterion['Con'](features, labels)
            loss_CE = criterion['CE'](output, labels_aug)

        else:
            raise ValueError("check method")  

        # update metric
        losses_CE.update(loss_CE.item(), bsz)
        if opt.method == 'Joint_Con':
            losses_Con.update(loss_Con.item(), bsz)
        elif 'CE' not in opt.method:
            raise ValueError("check method")  
        acc1, _ = accuracy(output[:bsz,:], labels[:bsz], topk=(1, 2))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        total_loss = loss_CE
        if opt.method == 'Joint_Con':
            total_loss = total_loss + loss_Con *opt.l_con
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx == len(trainloader_S)-1:
        # if idx % 1 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx + 1, len(trainloader_S), batch_time=batch_time,
                   data_time=data_time, loss=losses_CE), end = '')
            if opt.method == 'Joint_Con':
                print('con_loss {loss.val:.4f} ({loss.avg:.4f}'.format(loss=losses_Con))
            else:
                print('\n')
            sys.stdout.flush()
    if opt.method == 'Joint_Con':
        return {'CE':losses_CE.avg, 'Con': losses_Con.avg}, top1.avg
    elif 'CE' in opt.method:
        return losses_CE.avg, top1.avg
    else:
        raise ValueError("check method") 


def train_sampling_dsbn(trainloader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_CE = AverageMeter()
    top1 = AverageMeter()

    if opt.method == 'Joint_Con':
        losses_Con = AverageMeter()
    end = time.time()

    trainloader_T = trainloader['target']
    trainloader_S = trainloader['source']

    iter_T = iter(trainloader_T)
    iter_S = iter(trainloader_S)

    for idx in range(len(trainloader_S)): # FIXME assume that # of source dataset is larger than # of target dataset
        try: 
            images_T, labels_T = iter_T.next()
        except:
            iter_T = iter(trainloader_T)
            images_T, labels_T = iter_T.next()
        images_S, labels_S = iter_S.next()

        third = len(trainloader_S)//3
        if idx % third == 0:
            print(f'{idx}/{len(trainloader_S)}')

        data_time.update(time.time() - end)
        labels = torch.cat([labels_T, labels_S], dim=0)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # target to zeros, source to ones
        domain_idx_T = torch.zeros(labels_T.shape[0], dtype=torch.long).cuda(non_blocking=True)
        domain_idx_S = torch.ones(labels_S.shape[0], dtype=torch.long).cuda(non_blocking=True)
   
        
        if opt.method == 'Joint_CE':
            images_T = images_T.cuda(non_blocking=True)
            images_S = images_S.cuda(non_blocking=True)

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            output_T = model(images_T, domain_idx_T)
            output_S = model(images_S, domain_idx_S)
            output = torch.cat([output_T, output_S], dim=0)
            loss_CE = criterion(output, labels)
            
        elif opt.method == 'Joint_Con':
            if idx ==0:
                print(images_T[0].shape, 'target')
                print(images_S[0].shape, 'source')
                
            images_T = torch.cat([images_T[0], images_T[1]], dim=0).cuda(non_blocking=True)
            images_S = torch.cat([images_S[0], images_S[1]], dim=0).cuda(non_blocking=True)
            labels_aug = torch.cat([labels, labels], dim=0)
            
            warmup_learning_rate(opt, epoch, idx, len(trainloader), optimizer)

            # compute loss
            features_T, output_T = model(images_T, domain_idx_T)
            features_S, output_S = model(images_S, domain_idx_S)

            features_T0, features_T1 = features_T.chunk(2)
            features_S0, features_S1 = features_S.chunk(2)
            output_T0, output_T1 = output_T.chunk(2)
            output_S0, output_S1 = output_S.chunk(2)

            features = torch.cat([features_T0, features_S0, features_T1, features_S1], dim=0)
            output = torch.cat([output_T0, output_S0, output_T1, output_S1], dim=0)

            if opt.head == 'mlp':
                f0, f1 = torch.split(features, [bsz, bsz], dim=0)
            elif opt.head == 'fc':
                f0, f1 = torch.split(normalize(output,dim=1), [bsz, bsz], dim=0)
            features = torch.cat([f0.unsqueeze(1), f1.unsqueeze(1)], dim=1)
            
            loss_Con = criterion['Con'](features, labels)
            loss_CE = criterion['CE'](output, labels_aug)
        else:
            raise ValueError("check method")  

        # update metric
        losses_CE.update(loss_CE.item(), bsz)
        if opt.method == 'Joint_Con':
            losses_Con.update(loss_Con.item(), bsz)
        elif 'CE' not in opt.method:
            raise ValueError("check method")  
        acc1, _ = accuracy(output[:bsz,:], labels[:bsz], topk=(1, 2))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        total_loss = loss_CE
        if opt.method == 'Joint_Con':
            total_loss = total_loss + loss_Con *opt.l_con
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx == len(trainloader_S)-1:
        # if idx % 1 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(trainloader_S), batch_time=batch_time,
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
            if opt.dsbn:
                domain_idx_T = torch.zeros(labels.shape[0], dtype=torch.long).cuda()
                output = model(images, domain_idx_T)
            else:
                output = model(images)

            if opt.method == 'Joint_Con':
                output = output[1] # feature, output in Joint_Con
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
        if metric == 'last':
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
            if opt.dsbn:
                domain_idx_T = torch.zeros(labels.shape[0], dtype=torch.long).cuda()
                output = model(images, domain_idx_T)
            else:
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
            return auc, acc, f1, probs, gts
            
        else:
            raise ValueError("unsupported metric")
    else:
        bacc = accuracy_score(gts, probs>=best_th)
        best_acc, best_th = best_accuracy(gts,probs)
        return bacc, best_acc, best_th