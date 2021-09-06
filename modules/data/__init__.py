import os.path as osp
from math import ceil

from torchvision import transforms
import torch
import numpy as np

from modules.data.transform import TwoCropTransform, get_transform
from modules.data.dataset import GeneralDataset, ClassBalancedDataset


def generate_imageset_by_seed(root, opt):
    train_path = osp.join(root, 'train.{}-{}.txt'.format(opt.test_fold, opt.val_fold))
    val_path = osp.join(root, 'validation.{}-{}.txt'.format(opt.test_fold, opt.val_fold))

    with open(train_path, 'r') as fid:
        temp = fid.read()
    train_names = temp.strip().split('\n')

    with open(val_path, 'r') as fid:
        temp = fid.read()
    val_names = temp.strip().split('\n')
    whole_names = train_names + val_names

    SEED = opt.ur_seed

    ls_samples = [10, 50, 100]
    ls_urs = [0.1, 0.3, 0.5, 1.0]

    ls_ur_samples = [int(ur*len(whole_names)) for ur in ls_urs]
    ls_total_samples = sorted(ls_samples + ls_ur_samples)

    train_names = whole_names
    for smp in ls_total_samples[::-1]: # descending order
        np.random.seed(SEED)
        train_names = np.random.choice(train_names, size=smp, replace=False)
        # find ur if ur else smp
        if not smp in ls_samples:
            smp = [ur for ur in ls_urs if int(ur*len(whole_names))==smp][0]
        write_path = train_path[:-4] + f'-ur{smp}-{SEED}.txt'
        with open(write_path, 'w') as f:
            for img in train_names:
                f.write(img + '\n')


def load_image_names(data_dir, util_rate, opt):
    imageset_dir = osp.join(data_dir, 'imageset/single_image.2class',opt.imgset_dir)

    if opt.ur_from_imageset and data_dir == opt.target_folder:
        # merge train + validation and choose ur% data for training set
        img_set_name = 'train.{}-{}-ur{}-{}.txt'.format(opt.test_fold, opt.val_fold, opt.train_util_rate,opt.ur_seed)

        img_set_pth = osp.join(imageset_dir, img_set_name)
        if not osp.exists(img_set_pth):
            generate_imageset_by_seed(imageset_dir, opt)

        with open(img_set_pth, 'r') as fid:
            temp = fid.read() # FIXME why fix?
        train_names = temp.strip().split('\n')
        val_names = None
        # no validation phase
        print(f"Get UR from {img_set_name}")
    else:
        # sample training set from train.x-x.txt only
        with open(osp.join(imageset_dir, 'train.{}-{}.txt'.format(opt.test_fold, opt.val_fold)), 'r') as fid:
            temp = fid.read()
        train_names = temp.strip().split('\n')

        with open(osp.join(imageset_dir, 'validation.{}-{}.txt'.format(opt.test_fold, opt.val_fold)), 'r') as fid:
            temp = fid.read()
        val_names = temp.strip().split('\n')

        if opt.whole_data_train:
            train_names = train_names + val_names
            val_names = None

        if util_rate < 1:
            num_used = int(len(train_names) * util_rate)
            np.random.seed(1)
            train_names = np.random.choice(train_names, size=num_used, replace=False)

    with open(osp.join(imageset_dir, 'test.{}.txt'.format(opt.test_fold)), 'r') as fid:
        temp = fid.read()
    test_names = temp.strip().split('\n')

    return train_names, val_names, test_names


def set_loader(opt):
    train_names_T, val_names_T, test_names_T = load_image_names(opt.target_folder, opt.train_util_rate,opt)
    print(f"# of target trainset:{len(train_names_T)}")

    # sample image
    import cv2, os
    smp_img_pth = os.path.join(opt.target_folder, 'image', train_names_T[0])
    smp_img = cv2.imread(smp_img_pth)
    h, w, c = smp_img.shape
    opt.crop_size = (int(h*7/8), int(w*7/8))

    ##----------Transforms----------##
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    scale = (0.875, 1.)

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = get_transform(opt=opt, mean=mean, std=std, scale=scale)

    if opt.aug.lower() in ['pin', 'pin-sim']:
        test_transform = val_transform = transforms.Compose([
            transforms.CenterCrop(opt.crop_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        test_transform = val_transform = transforms.Compose([
            transforms.Resize(opt.size),
            transforms.ToTensor(),
            normalize,
        ])
    if opt.method == 'Joint_Con':
        train_transform = TwoCropTransform(train_transform)


    ##----------Dataset----------##
    Dataset = ClassBalancedDataset if opt.class_balanced else GeneralDataset
    if 'Joint' in opt.method:
        #train_names_S, _, _ = load_image_names(opt.source_folder, 1.0, opt)
        tr, _, ts = load_image_names(opt.source_folder, 1.0, opt) # val = None
        train_names_S = tr + ts
        print(f"# of source trainset:{len(train_names_S)}")

        if opt.sampling == 'unbalanced':
            train_dataset = Dataset(data_dir=opt.target_folder, image_names=train_names_T,
                                            ext_data_dir=opt.source_folder, ext_image_names=train_names_S,
                                            transform=train_transform)
        else:
            train_dataset_T = Dataset(data_dir=opt.target_folder, image_names=train_names_T,
                                transform=train_transform)
            train_dataset_S = Dataset(data_dir=opt.source_folder, image_names=train_names_S,
                                transform=train_transform)

    else:
        train_dataset = Dataset(data_dir=opt.target_folder, image_names=train_names_T,
                                    transform=train_transform)

    if not opt.whole_data_train:
        val_dataset = Dataset(data_dir=opt.target_folder, image_names=val_names_T,
                                            transform=val_transform,)
        print(f"# of target valset:{len(val_names_T)}")

    test_dataset = GeneralDataset(data_dir=opt.target_folder, image_names=test_names_T,
                                    transform=test_transform)
    print(f"# of target testset:{len(test_names_T)}")

    ##----------Dataloader----------##
    # It seems that incosistent batch size harms contrastive learning, so drop the last batch
    if opt.sampling == 'unbalanced':
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, drop_last = True)
    else:
        assert opt.batch_size%2 == 0
        train_loader_T = torch.utils.data.DataLoader(
            train_dataset_T, batch_size=opt.batch_size//2, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, drop_last = True)
        train_loader_S = torch.utils.data.DataLoader(
            train_dataset_S, batch_size=opt.batch_size//2, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, drop_last = True)
        train_loader = {'target':train_loader_T, 'source':train_loader_S}

    if opt.whole_data_train:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=False, sampler=None)

    return { 'train': train_loader, 'val': val_loader, 'test': test_loader}

def adjust_batch_size(opt, train_dataset_T, train_dataset_S, epoch):
    """
    Assume that the number of target images is less than source images by default.
    """
    num_T = len(train_dataset_T)
    num_S = len(train_dataset_S)

    first_BS_T = ceil(opt.batch_size * num_T/(num_T + num_S))
    last_BS_T = opt.batch_size // 2

    BS_T = round((epoch/opt.epochs)*(last_BS_T - first_BS_T) + first_BS_T)
    BS_S = opt.batch_size - BS_T

    train_loader_T = torch.utils.data.DataLoader(
            train_dataset_T, batch_size=BS_T, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, drop_last = True)
    train_loader_S = torch.utils.data.DataLoader(
            train_dataset_S, batch_size=BS_S, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False, sampler=None, drop_last = True)

    return {'target':train_loader_T, 'source':train_loader_S}
