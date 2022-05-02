import os.path as osp
from os import listdir
import random
from math import ceil

from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torch
import numpy as np

from torchsampler import ImbalancedDatasetSampler, WeightedSampler, DomainWeightedSampler
from modules.data.transform import TwoCropTransform, get_transform
from modules.data.dataset import GeneralDataset, ClassBalancedDataset, MVTECDataset





def load_image_names(data_dir, util_rate, opt):
    if 'd_sub' in opt.dataset:
        imageset_dir = osp.join(data_dir, 'split/')
    else:
        imageset_dir = osp.join(data_dir, 'imageset/single_image.2class',opt.imgset_dir)

    if opt.ur_from_imageset and data_dir == opt.target_folder:
        img_set_name = 'train.{}-{}-ur{}-{}.txt'.format(opt.test_fold, opt.val_fold, opt.train_util_rate,opt.ur_seed)
        with open(osp.join(imageset_dir,img_set_name ), 'r') as fid:
            temp = fid.read() # FIXME why fix?
        train_names = temp.split('\n')[:-1]
        val_names = None
        print(f"Get UR from {img_set_name}")
        # no validation phase
    else:
        if 'd_sub' in opt.dataset:
            train_dir = 'train_21.txt'
            val_dir = 'train_21.txt'
        else:
            train_dir = 'train.{}-{}.txt'.format(opt.test_fold, opt.val_fold)
            val_dir = 'validation.{}-{}.txt'.format(opt.test_fold, opt.val_fold)

        with open(osp.join(imageset_dir, train_dir), 'r') as fid:
            temp = fid.read()
        train_names = temp.split('\n')[:-1]
    
        with open(osp.join(imageset_dir, val_dir), 'r') as fid:
            temp = fid.read()
        val_names = temp.split('\n')[:-1]

        if opt.whole_data_train:
            if 'd_sub' in opt.dataset:
                val_names = []
            train_names = train_names + val_names
            val_names = None

        if util_rate != 1.0:
            if util_rate <1.0:
                num_used = int(len(train_names) * util_rate)
            else:
                num_used = util_rate
            random.seed(1)
            random.shuffle(train_names)
            train_names = train_names[:num_used]

            # np.random.seed(1)
            # train_names = np.random.choice(train_names, size=num_used, replace=False)  

    if 'd_sub' in opt.dataset:
        test_dir = 'test_21.txt'
    else: test_dir = 'test.{}.txt'.format(opt.test_fold)

    with open(osp.join(imageset_dir, test_dir), 'r') as fid:
        temp = fid.read()
    test_names = temp.split('\n')[:-1]

    return train_names, val_names, test_names




def load_image_names_MVTEC(data_dir):
    train_dir = osp.join(data_dir, 'train', 'good')
    test_dir = osp.join(data_dir, 'test')
    
    train_OK = []
    test_NG = []
    test_OK = []

    for img in listdir(train_dir):
        img_dir = osp.join(train_dir, img)
        train_OK.append( (img_dir, 0))

    for folder in listdir(test_dir):
        for img in listdir(osp.join(test_dir, folder)):
            img_dir = osp.join(test_dir, folder, img)
            if folder != 'good':
                test_NG.append( (img_dir,1))
            else:
                test_OK.append( (img_dir,0) )

    random.seed(1)
    random.shuffle(test_NG)
    len_test_NG = len(test_NG)
    ratio =int(len_test_NG * 0.8)
    train_NG = test_NG[:ratio]
    test_NG = test_NG[ratio:]
    test_image = test_NG + test_OK
    train_image = train_OK + train_NG

    print( len(train_OK), 'train_OK')
    print( len(train_NG), 'train_NG')
    print( len(test_OK), 'test_OK')
    print( len(test_NG), 'test_NG')

    return train_image, None, test_image


def set_loader(opt, preprocess):
    ##----------Transforms----------##

    if preprocess is None:
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
    else:
        train_transform = val_transform = test_transform = preprocess
    if opt.method == 'Joint_Con' and not opt.one_crop:
        train_transform =TwoCropTransform(train_transform)

    
    ##----------Dataset----------##
    if "MVTEC" in opt.dataset:
        train_names_T, val_names_T, test_names_T = load_image_names_MVTEC(opt.target_folder)
    else:
        train_names_T, val_names_T, test_names_T = load_image_names(opt.target_folder, opt.train_util_rate,opt)
    print(f"# of target trainset:{len(train_names_T)}")
    if 'Joint' in opt.method:
        train_names_S, _, _ = load_image_names(opt.source_folder, 1.0, opt)
        print(f"# of source trainset:{len(train_names_S)}")
        
        if opt.sampling == 'unbalanced' or opt.sampling == 'domainKang':
            use_domain_tag = (opt.sampling == 'domainKang')
            train_dataset = GeneralDataset(data_dir=opt.target_folder, image_names=train_names_T,
                                            ext_data_dir=opt.source_folder, ext_image_names=train_names_S,
                                            transform=train_transform, \
                                            use_domain_tag=use_domain_tag, source_sampling=opt.source_sampling, opt=opt)
 
        else:
            Dataset = ClassBalancedDataset if opt.class_balanced else GeneralDataset

            train_dataset_T = Dataset(data_dir=opt.target_folder, image_names=train_names_T,
                                transform=train_transform)
            train_dataset_S = Dataset(data_dir=opt.source_folder, image_names=train_names_S,
                    transform=train_transform)

    else:
        if "MVTEC" in opt.dataset:
            train_dataset = MVTECDataset(train_names_T,transform=train_transform)
        else:
            train_dataset = GeneralDataset(data_dir=opt.target_folder, image_names=train_names_T,
                                    transform=train_transform,opt=opt)

    if not opt.whole_data_train:
        val_dataset = GeneralDataset(data_dir=opt.target_folder, image_names=val_names_T,
                                            transform=val_transform,opt=opt) 
        print(f"# of target valset:{len(val_names_T)}")

    if "MVTEC" in opt.dataset:
        test_dataset = MVTECDataset(test_names_T,transform=test_transform)
    else:
        test_dataset = GeneralDataset(data_dir=opt.target_folder, image_names=test_names_T,
                                    transform=test_transform,opt=opt)
    print(f"# of target testset:{len(test_names_T)}")
    
    ##----------Dataloader----------##
    # It seems that incosistent batch size harms contrastive learning, so drop the last batch
    if opt.sampling == 'unbalanced':
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle= True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last = True)
    elif opt.sampling == 'domainKang':
        sampler = DomainWeightedSampler(datapoints=train_dataset, num_domains=2, weights=(1,1))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle= False,
            num_workers=opt.num_workers, pin_memory=True, sampler=sampler, drop_last = True)
    else:
        shuffle = False if opt.class_balanced or 'IDS' in opt.sampling or 'Kang' in opt.sampling else True
        assert opt.batch_size%2 == 0
        if 'IDS' in opt.sampling:
            sampler_T = ImbalancedDatasetSampler(train_dataset_T)
            sampler_S = ImbalancedDatasetSampler(train_dataset_S)
        elif 'Kang' in opt.sampling:
            sampler_T = WeightedSampler([1,1])(train_dataset_T)
            sampler_S = WeightedSampler([1,1])(train_dataset_S)
        else:
            sampler_T = sampler_S = None


        train_loader_T = torch.utils.data.DataLoader(
            train_dataset_T, batch_size=opt.batch_size//2, shuffle= shuffle,
            num_workers=opt.num_workers, pin_memory=True, sampler=sampler_T, drop_last = True)
        train_loader_S = torch.utils.data.DataLoader(
            train_dataset_S, batch_size=opt.batch_size//2, shuffle= shuffle,
            num_workers=opt.num_workers, pin_memory=True, sampler=sampler_S, drop_last = True)
        train_loader = {'target':train_loader_T, 'source':train_loader_S}
        
    if opt.whole_data_train:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle= False,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle= False,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

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

    shuffle = False if opt.class_balanced else True
    train_loader_T = torch.utils.data.DataLoader(
            train_dataset_T, batch_size=BS_T, shuffle= shuffle,
            num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last = True)
    train_loader_S = torch.utils.data.DataLoader(
        train_dataset_S, batch_size=BS_S, shuffle= shuffle,
        num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last = True)
    
    return {'target':train_loader_T, 'source':train_loader_S}


def adjust_domain_weight(opt, num_T, num_S, epoch):
    """
    Assume that the number of target images is less than source images by default.
    """
    num_S = int(num_S * opt.source_rate)
    first_BS_T = ceil(opt.batch_size * num_T/(num_T + num_S))
    last_BS_T = opt.batch_size // 2

    weight_T = round((epoch/opt.epochs)*(last_BS_T - first_BS_T) + first_BS_T)
    weight_S = opt.batch_size - weight_T
    
    return weight_T, weight_S
