from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import accuracy_score


import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def denormalize(input): # (3,128,128)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for i in range(3):
        input[i,:,:] *= std[i]
        input[i,:,:] += mean[i]

    return input


def image_from_array(img):
    #torch to img
    img = denormalize(img.cpu().data.numpy())
    img = np.transpose(img, (1,2,0))
    img = np.uint8(255 * img)
    img = Image.fromarray(img, 'RGB')

    return img


def visualize_imgs(img1, img2, img12, epoch, comb , cam = None):
    '''
    img1 (torch tensor)  [3, 128, 128]
    img2 (torch tensor)
    img12 (torch tensor)
    cam (mask)
    '''
    img1 = image_from_array(img1)
    img2 = image_from_array(img2)
    img12 = image_from_array(img12)

    fig = plt.figure(figsize=(80, 20), dpi=150)
    name = f'epoch_{epoch}_{comb}'
    fig.suptitle(name, fontsize=80)
    plt.rcParams.update({'font.size': 50})

    ax = fig.add_subplot(1, 4, 1)
    plt.imshow(img1)
    ax.set_title('Image 1')

    ax = fig.add_subplot(1, 4, 2)
    plt.imshow(img2)
    ax.set_title('Image 2')

    ax = fig.add_subplot(1, 4, 3)
    plt.imshow(img12)
    ax.set_title('Image 1+2')

    if cam is not None:
        ax = fig.add_subplot(1, 4, 4)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = np.uint8(255 * heatmap)
        plt.imshow(heatmap)
        ax.set_title('CAM')
    plt.savefig(f"./DefectMix/{name}.png")
    plt.close(fig)
    plt.close()
    fig.clf()


def bbox2(img):
    assert img.ndim == 2
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rand_bbox(size, lam, bbs = None):
    W = size[2] # 순서 잘못된 듯?
    H = size[3]


    if bbs is not None:
        bx1, bx2, by1, by2 = bbs # cam
        # step 1, get cx and cy except cam

        it = 0
        while True:
            it+=1
            if it >= 100:
                print(it)
                return 0,0,0,0
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                break
        min_dist = W**2 + H**2
        min_i = 0
        min_j = 0
        for i in range(bx1, bx2+1):
            for j in range(by1, by2+1):
                dist = (cx-i)**2 + (cy-j)**2
                if dist <= min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j


        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat) //2
        max_cut = max(abs(min_i - cx), abs(min_j - cy))
        cut = min(cut_w, max_cut)

        bbx1 = np.clip(cx - cut, 0, W)
        bby1 = np.clip(cy - cut, 0, H)
        bbx2 = np.clip(cx + cut, 0, W)
        bby2 = np.clip(cy + cut, 0, H)

        mask = np.zeros((W,H))
        mask[bx1:bx2, by1:by2] +=1
        mask[bbx1:bbx2, bby1:bby2] +=1
        if 2 in mask:
            print(bbx1,bbx2,bby1,bby2)
            print(bx1,bx2,by1,by2)
            print(min_i, min_j, cx,cy, cut_w)

        assert 2 not in mask
    else:
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)



    return bbx1, bby1, bbx2, bby2



def make_cutmix(images, labels, model, cam, m,bsz,epoch):

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

    origin_images = torch.clone(images)

    for iidx in range(bsz):
        iidx_b = rand_index[iidx]
        image1 = origin_images[iidx]
        image2 = origin_images[iidx_b]
        if labels_a[iidx] == 0:
            if labels_b[iidx] == 0: # ok <- ok: ok, random box of cutmix
                images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                # visualize_imgs(image1, image2, images[iidx],epoch, comb='ok-ok')

            elif labels_b[iidx] == 1: # ok <- ng: to ng, cut defect-region
                ng_id = ng_iidx.index(iidx_b) # iidx_b가 ng_iidx에서 몇 번째인지
                if soft_output[ng_id, 1] >= 0.9:
                    bbx1, bbx2, bby1, bby2 = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                    cutmix_labels[iidx] = 1

                    # visualize_imgs(image1, image2, images[iidx],epoch, comb='ok-ng', cam=cam_mask[ng_id,:,:])
            else:
                raise ValueError("fucked label")
        elif labels_a[iidx] == 1:
            if labels_b[iidx] == 0:  # ng <- ok: ng, cut non-defect-region
                ng_id = ng_iidx.index(iidx)
                if soft_output[ng_id, 1] >= 0.9:
                    bbs = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam, bbs) # 겹치면 빼기
                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]

                    # visualize_imgs(image1, image2, images[iidx], epoch, comb='ng-ok', cam=cam_mask[ng_id,:,:])

                cutmix_labels[iidx] = 1

            elif labels_b[iidx] == 1: # ng <- ng: ng, cut defect-region (box)
                ng_id = ng_iidx.index(iidx_b) # real_iidx가 ng_idx에서 몇 번째인지
                if soft_output[ng_id, 1] >= 0.9:
                    bbx1, bbx2, bby1, bby2 = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]

                    # visualize_imgs(image1, image2, images[iidx],epoch, comb='ng-ng', cam=cam_mask[ng_id,:,:])
                cutmix_labels[iidx] = 1

            else:
                raise ValueError("fucked label")
        else:
            raise ValueError("fucked label")
    # ROI는 곱하기로 alpha blending 하면 된다.
    return images, cutmix_labels


def make_cutmix_ext(images, labels, model, cam, m,bsz,epoch, ext_images, type_):
    ng_imgs = images[labels == 1]
    if ng_imgs.shape[0] == 0:
        return images, ext_images, labels
    grayscale_cam= cam(input_tensor=ng_imgs, target_category=1) # (n_ng,h,w) numpy 0~1
    cam_mask = grayscale_cam >= 0.7 #  (n_ng,h,w)
    _, ng_outputs = model(ng_imgs) # FIXME double forwards
    soft_output = m(ng_outputs) # n_ng, 2

    ext_ng_imgs = ext_images[labels == 1]
    ext_grayscale_cam= cam(input_tensor=ext_ng_imgs, target_category=1) # (n_ng,h,w) numpy 0~1
    ext_cam_mask = ext_grayscale_cam >= 0.7 #  (n_ng,h,w)
    _, ext_ng_outputs = model(ext_ng_imgs) # FIXME double forwards
    ext_soft_output = m(ext_ng_outputs) # n_ng, 2


    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(images.size()[0]).cuda()
    labels_a = labels
    labels_b = labels[rand_index]
    cutmix_labels = torch.zeros_like(labels)

    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    ng_iidx = [iid for iid in range(len(labels)) if labels[iid] == 1 ] # ng인 애들

    origin_images = torch.clone(images)
    ext_origin_images = torch.clone(ext_images)

    for iidx in range(bsz):
        iidx_b = rand_index[iidx]
        image1 = origin_images[iidx]
        image2 = origin_images[iidx_b]
        if labels_a[iidx] == 0:
            if labels_b[iidx] == 0: # ok <- ok: ok, random box of cutmix
                if type_ == 'oo':
                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                    ext_images[iidx, :, bbx1:bbx2, bby1:bby2] = ext_origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]

                    # visualize_imgs(image1, image2, images[iidx],epoch, comb='ok-ok')

            elif labels_b[iidx] == 1: # ok <- ng: to ng, cut defect-region
                ng_id = ng_iidx.index(iidx_b) # iidx_b가 ng_iidx에서 몇 번째인지
                if soft_output[ng_id, 1] >= 0.9 and ext_soft_output[ng_id, 1] >= 0.9:
                    bbx1, bbx2, bby1, bby2 = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                    e_bbx1, e_bbx2, e_bby1, e_bby2 = bbox2(ext_cam_mask[ng_id,:,:]) # bounding box 추출

                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                    ext_images[iidx, :, e_bbx1:e_bbx2, e_bby1:e_bby2] = ext_origin_images[iidx_b, :, e_bbx1:e_bbx2, e_bby1:e_bby2]
                    cutmix_labels[iidx] = 1

                    # visualize_imgs(image1, image2, images[iidx],epoch, comb='ok-ng', cam=cam_mask[ng_id,:,:])
            else:
                raise ValueError("fucked label")
        elif labels_a[iidx] == 1:
            if labels_b[iidx] == 0:  # ng <- ok: ng, cut non-defect-region
                ng_id = ng_iidx.index(iidx)
                if soft_output[ng_id, 1] >= 0.9 and ext_soft_output[ng_id, 1] >= 0.9 and type_ == 'xo':
                    bbs = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                    e_bbs = bbox2(ext_cam_mask[ng_id,:,:]) # bounding box 추출
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam, bbs) # 겹치면 빼기
                    e_bbx1, e_bby1, e_bbx2, e_bby2 = rand_bbox(ext_images.size(), lam, e_bbs) # 겹치면 빼기

                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                    ext_images[iidx, :, e_bbx1:e_bbx2, e_bby1:e_bby2] = ext_origin_images[iidx_b, :, e_bbx1:e_bbx2, e_bby1:e_bby2]
                    # visualize_imgs(image1, image2, images[iidx], epoch, comb='ng-ok', cam=cam_mask[ng_id,:,:])

                cutmix_labels[iidx] = 1

            elif labels_b[iidx] == 1: # ng <- ng: ng, cut defect-region (box)
                ng_id = ng_iidx.index(iidx_b) # real_iidx가 ng_idx에서 몇 번째인지
                if soft_output[ng_id, 1] >= 0.9 and ext_soft_output[ng_id, 1] >= 0.9 and type_ == 'xx':
                    bbx1, bbx2, bby1, bby2 = bbox2(cam_mask[ng_id,:,:]) # bounding box 추출
                    e_bbx1, e_bbx2, e_bby1, e_bby2 = bbox2(ext_cam_mask[ng_id,:,:]) # bounding box 추출
                    images[iidx, :, bbx1:bbx2, bby1:bby2] = origin_images[iidx_b, :, bbx1:bbx2, bby1:bby2]
                    ext_images[iidx, :, e_bbx1:e_bbx2, e_bby1:e_bby2] = ext_origin_images[iidx_b, :, e_bbx1:e_bbx2, e_bby1:e_bby2]

                    # visualize_imgs(image1, image2, images[iidx],epoch, comb='ng-ng', cam=cam_mask[ng_id,:,:])
                cutmix_labels[iidx] = 1

            else:
                raise ValueError("fucked label")
        else:
            raise ValueError("fucked label")
    # ROI는 곱하기로 alpha blending 하면 된다.
    return images, ext_images, cutmix_labels

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
    elif opt.optimizer == 'ADAMW':
        optimizer = optim.AdamW(model.parameters(),
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


def best_accuracy(gts, probs):
    best_th = 0.0
    best_acc = 0.0
    for th in range(0,200):
        th = th/200.0
        acc = accuracy_score(gts, probs>=th)

        if acc >= best_acc:
            best_acc = acc
            best_th = th

    return best_acc, best_th


