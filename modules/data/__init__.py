import os.path as osp

from torchvision import transforms, datasets
import torch
import numpy as np

from modules.data.transform import TwoCropTransform, get_transform
from modules.data.dataset import GeneralDataset





def load_image_names(data_dir, util_rate, opt):
    imageset_dir = osp.join(data_dir, 'imageset/single_image.2class',opt.imgset_dir)
    
    with open(osp.join(imageset_dir, 'train.{}-{}.txt'.format(opt.test_fold, opt.val_fold)), 'r') as fid:
        temp = fid.read()
    train_names = temp.split('\n')[:-1]
  
    with open(osp.join(imageset_dir, 'validation.{}-{}.txt'.format(opt.test_fold, opt.val_fold)), 'r') as fid:
        temp = fid.read()
    val_names = temp.split('\n')[:-1]

    if opt.whole_data_train:
        train_names = train_names + val_names
        
    with open(osp.join(imageset_dir, 'test.{}.txt'.format(opt.test_fold)), 'r') as fid:
        temp = fid.read()
    test_names = temp.split('\n')[:-1]

    if util_rate < 1:
        num_used = int(len(train_names) * util_rate)
        np.random.seed(1)
        train_names = np.random.choice(train_names, size=num_used, replace=False)

    print(f"# of trainset:{len(train_names)}")
    if not opt.whole_data_train:
        print(f"# of valset:{len(val_names)}")
    print(f"# of testset:{len(test_names)}")

    return train_names, val_names, test_names




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
    else: # custom dataset
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

    if opt.method == 'Joint_Con':
        train_transform =TwoCropTransform(train_transform)

    
    # dataset
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
        if 'Joint' in opt.method:
            train_names_S, _, _ = load_image_names(opt.source_data_folder, 1.0, opt)
            train_dataset = GeneralDataset(data_dir=opt.data_folder, image_names=train_names_T,
                                            ext_data_dir=opt.source_data_folder, ext_image_names=train_names_S,
                                            transform=train_transform)
        else:
            train_dataset = GeneralDataset(data_dir=opt.data_folder, image_names=train_names_T,
                                        transform=train_transform)

        if not opt.whole_data_train:
            val_dataset = GeneralDataset(data_dir=opt.data_folder, image_names=val_names_T,
                                                transform=val_transform,)
        test_dataset = GeneralDataset(data_dir=opt.data_folder, image_names=test_names_T,
                                        transform=test_transform)
        
    # dataloader
    if custom:
        droplast = True if 'Joint' in opt.method else False
        # It seems that incosistent batch size harms contrastive learning, so drop the last batch
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle= True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last = droplast)
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

    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)
        
        return { 'train': train_loader, 'val': val_loader}

