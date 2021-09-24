import os.path as osp
from collections import defaultdict
import random

import json
import torch
import numpy as np
from PIL import Image


class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_cls, image_names, ext_data_dir=None, ext_image_names=None, transform=None):
        self._data_dir = data_dir
        self._num_cls = num_cls
        self._transform = transform
        self._image_names = image_names
        self._ext_data_dir = ext_data_dir
        self._ext_image_names = ext_image_names


        anno_fname = 'single_image.{}class.json'.format(self._num_cls)
        self._anno_json = json.load(open(osp.join(data_dir, 'annotation', anno_fname), 'r'))

        self._image_fpaths, self._labels = [], []
        for image_name in self._image_names:
            image_fpath = osp.join(data_dir, 'image', image_name)
            label = self._anno_json['single_image'][image_name]['class'][0]

            self._image_fpaths.append(image_fpath)
            self._labels.append(label)

        if self._ext_data_dir is not None:
            self._ext_anno_json = json.load(open(osp.join(ext_data_dir, 'annotation', anno_fname), 'r'))
            for image_name in self._ext_image_names:
                image_fpath = osp.join(ext_data_dir, 'image', image_name)
                label = self._ext_anno_json['single_image'][image_name]['class'][0]

                self._image_fpaths.append(image_fpath)
                self._labels.append(label)

        # For grayscale images
        np_img = np.array(Image.open(self._image_fpaths[0]))
        if np_img.ndim == 2:
            self.is_gray = True
        elif np_img.ndim == 3:
            self.is_gray = False
        else:
            raise ValueError("This image might be RGBA, which is not supported yet.")

    def __len__(self):
        return len(self._image_fpaths)

    def __getitem__(self, index):

        img_fpath = self._image_fpaths[index]
        img = (Image.open(img_fpath))
        if self.is_gray:
            img = img.convert('RGB')

        if self._transform is not None:
            image = self._transform(img)

        return image, self._labels[index]


class ClassBalancedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_cls, image_names, transform=None):
        self._data_dir = data_dir
        self._num_cls = num_cls
        self._transform = transform
        self._image_names = image_names

        anno_fname = 'single_image.{}class.json'.format(self._num_cls)
        self._anno_json = json.load(open(osp.join(data_dir, 'annotation', anno_fname), 'r'))

        # make dict for each class
        self._image_fpaths_dict = defaultdict(list)
        for image_name in self._image_names:
            image_fpath = osp.join(data_dir, 'image', image_name)
            label = self._anno_json['single_image'][image_name]['class'][0]
            self._image_fpaths_dict[label] += [image_fpath]

        # make class-balanced fpaths
        self.most_common_cls = max(self._image_fpaths_dict, key= lambda x: len(self._image_fpaths_dict[x]))
        self.longest_length = len(self._image_fpaths_dict[self.most_common_cls])
        self.get_class_balanced()

        # For grayscale images
        np_img = np.array(Image.open(self._image_fpaths_dict[0][0]))
        if np_img.ndim == 2:
            self.is_gray = True
        elif np_img.ndim == 3:
            self.is_gray = False
        else:
            raise ValueError("This image might be RGBA, which is not supported yet.")

    def __len__(self):
        return len(self._image_fpaths)

    def __getitem__(self, index):
        img_fpath = self._image_fpaths[index]
        img = (Image.open(img_fpath))
        if self.is_gray:
            img = img.convert('RGB')

        if self._transform is not None:
            image = self._transform(img)
        return image, self._labels[index]

    def get_class_balanced(self):
        # generate labels and fpaths
        num_cls = len(self._image_fpaths_dict.keys())
        self._image_fpaths = []
        self._labels = []

        classes = list(self._image_fpaths_dict.keys())

        for idx, CLS in enumerate(classes):
            if CLS != self.most_common_cls:
                # augment lists of each class to have the same length with the longest one.
                num_augment = self.longest_length // len(self._image_fpaths_dict[CLS])
                augmented_paths = self._image_fpaths_dict[CLS] * num_augment

                num_to_choose = self.longest_length % len(self._image_fpaths_dict[CLS])
                augmented_paths += random.sample(self._image_fpaths_dict[CLS], num_to_choose)
            else:
                augmented_paths = self._image_fpaths_dict[CLS]

            self._image_fpaths += augmented_paths
            self._labels += [CLS]*len(augmented_paths)
