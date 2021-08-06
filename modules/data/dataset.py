import os
import torch
from PIL import Image

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def append_samples(self, samples, dir, cls_2_id, domain_idx=0):
        for cls_folder in os.listdir(dir):
            if os.path.isdir(os.path.join(dir,cls_folder)):
                assert cls_folder in cls_2_id.keys()
                for image in os.listdir(os.path.join(dir, cls_folder)):
                    image_path = os.path.join(dir, cls_folder, image)
                    samples.append([image_path, cls_2_id[cls_folder], domain_idx])
        return samples

    def __getitem__(self, index):
        path, target, domain_idx = self.samples[index]
        img = Image.open(path).convert('RGB')
    
        if self.transform is not None:
            img = self.transform(img)

        return img, target, domain_idx

    def __len__(self):
        return len(self.samples)


class OFFICE(BasicDataset):
    def __init__(self, target_dir, source_dir = None, transform=None, num_cls=2):
        super(OFFICE, self).__init__()
        classes = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
                   'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
                   'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
                   'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
                   'tape_dispenser', 'trash_can']
        assert len(classes) == num_cls
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        self.samples = []
        self.samples = self.append_samples(self.samples, target_dir, class_to_idx, domain_idx=0)
        if source_dir is not None:
            self.samples = self.append_samples(self.samples, source_dir, class_to_idx, domain_idx=1)
        
        self.transform = transform

class VISDA(BasicDataset):
    def __init__(self, target_dir, source_dir = None, transform=None, num_cls=2):
        super(VISDA, self).__init__()
        classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                   'skateboard', 'train', 'truck']
        assert len(classes) == num_cls
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        self.samples = []
        self.samples = self.append_samples(self.samples, target_dir, class_to_idx, domain_idx=0)
        if source_dir is not None:
            self.samples = self.append_samples(self.samples, source_dir, class_to_idx, domain_idx=1)
        
        self.transform = transform


class OFFICEHOME(BasicDataset):
    def __init__(self, target_dir, source_dir = None, transform=None, num_cls=2):
        super(OFFICEHOME, self).__init__()
        classes = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
                   'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
                   'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
                   'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop',
                   'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
                   'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
                   'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
        assert len(classes) == num_cls
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        self.samples = []
        self.samples = self.append_samples(self.samples, target_dir, class_to_idx, domain_idx=0)
        if source_dir is not None:
            self.samples = self.append_samples(self.samples, source_dir, class_to_idx, domain_idx=1)
        
        self.transform = transform

