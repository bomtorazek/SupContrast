import os.path as osp

from torchvision import transforms, datasets
import torch
import numpy as np

from modules.data.transform import TwoCropTransform, get_transform
from modules.data.dataset import OFFICE, VISDA, OFFICEHOME



def set_loader(opt):
    # construct data loader
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    scale = (0.875, 1.)

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = get_transform(opt=opt, mean=mean, std=std, scale=scale)

    val_transform = transforms.Compose([
        transforms.Resize(opt.size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.method == 'Joint_Con':
        train_transform =TwoCropTransform(train_transform)


    # dataset
    if opt.dataset == 'office-home':
        DATASET = OFFICEHOME
    elif opt.dataset == 'office-31':
        DATASET = OFFICE
    elif opt.dataset == 'visda':
        DATASET = VISDA
    else:
        raise NotImplementedError("not supported datset")


    if 'Joint' in opt.method:
        train_dataset = DATASET(target_dir=opt.target_folder, source_dir=opt.source_folder,
                                        transform=train_transform, num_cls=opt.num_cls)
    else:
        train_dataset = DATASET(target_dir=opt.target_folder,
                                    transform=train_transform, num_cls=opt.num_cls )

    val_dataset = DATASET(target_dir=opt.target_folder, 
                                            transform=val_transform, num_cls=opt.num_cls)

        
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle= True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None, drop_last = True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle= False,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)


    return { 'train': train_loader, 'val': val_loader}


  

