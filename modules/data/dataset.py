import os.path as osp
import json
import torch
import numpy as np
from PIL import Image


class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,image_names, ext_data_dir=None, ext_image_names=None, transform=None ):
        self._data_dir = data_dir
        self._transform = transform
        self._image_names = image_names
        self._ext_data_dir = ext_data_dir
        self._ext_image_names = ext_image_names
        

        anno_fname = 'single_image.2class.json'
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
        # image_dict = {}

        img_fpath = self._image_fpaths[index]
        img = (Image.open(img_fpath))
        if self.is_gray:
            img = img.convert('RGB')
        # .convert('RGB')
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


