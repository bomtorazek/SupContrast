import os
import os.path as osp
import json
from glob import glob

import torch
import numpy as np
from PIL import Image

import albumentations as AB
from albumentations.pytorch import ToTensor


class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,image_names, transform=None ):
        self._data_dir = data_dir
        self._transform = transform
        self._image_names = image_names
        

        anno_fname = 'single_image.2class.json'
        self._anno_json = json.load(open(osp.join(data_dir, 'annotation', anno_fname), 'r'))

        self._image_fpaths, self._labels = [], []
        for image_name in self._image_names:
            image_fpath = osp.join(data_dir, 'image', image_name)
            label = self._anno_json['single_image'][image_name]['class'][0]

            self._image_fpaths.append(image_fpath)
            self._labels.append(label)


    def __len__(self):
        return len(self._image_fpaths)

    def __getitem__(self, index):
        # image_dict = {}

        img_fpath = self._image_fpaths[index]
        img = (Image.open(img_fpath))
        # if len(img.shape) == 2: # gray
        #     img = np.stack([img, img, img], axis=-1)
        # image_dict['image'] = img
        # image_dict['path'] = img_fpath

        if self._transform is not None:
            image = self._transform(img)
 
        return image, self._labels[index]

    @property
    def labels(self):
        return self._labels

    @property
    def num_channels(self):
        assert self.__len__() > 0
        image_dict, _ = self.__getitem__(0)
        return image_dict['image'].shape[0]
