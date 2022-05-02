
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import PIL

from modules.data.aug_lib import TrivialAugment
from modules.data.RandAugment import rand_augment_transform

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def get_transform(opt, mean, std, scale):

    normalize = transforms.Normalize(mean=mean, std=std)
    if opt.aug.lower() == 'nothing': ## CLIP
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        TF = transforms.Compose([
            transforms.Resize(opt.size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(size=(opt.size, opt.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
    ])
    elif opt.aug.lower() == 'flip':
        TF = transforms.Compose([
            transforms.Resize(opt.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    elif opt.aug.lower() == 'flip_crop':
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    elif opt.aug.lower() == 'sim':
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
    ])
    elif 'stacked' in opt.aug.lower():
        # stacked_ra_3_5
        spt = opt.aug.split('_')
        n = spt[-2]
        m = spt[-1]
        int(n)
        int(m)
        ra_params = dict(
            translate_const=int(opt.size * 0.125),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur(opt.size//10)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(n, m),
                                   ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif 'rand' in opt.aug.lower():
        spt = opt.aug.split('_')
        n = spt[-2]
        m = spt[-1]
        int(n)
        int(m)
        ra_params = dict(
            translate_const=int(opt.size * 0.125),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=scale),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(n, m),
                                    ra_params),
            transforms.ToTensor(),
            normalize,
        ])
    elif 'cut' in opt.aug.lower():
        TF = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif opt.aug.lower() == 'trivial':
        TF = transforms.Compose([ 
            TrivialAugment(),
            transforms.RandomResizedCrop(size=opt.size, scale= scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    else:
        raise NotImplementedError("Unsupported augmentaton name")

    return TF