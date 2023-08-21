import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torchvision.transforms as transforms
from utils.auto_augment import auto_augment_transform


def get_transform(args, is_train_set=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # RGB

    lighting = Lighting(alphastd=0.1,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[[-0.5675,  0.7192,  0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948,  0.4203]])

    if is_train_set:
        transform_set = []
        transform_set += [transforms.RandomResizedCrop(size=args.cfg['train_cfg']['crop_size'])] # default size=224
        if args.colorjitter:
            transform_set += [transforms.ColorJitter(brightness=0.4,
                                                    contrast=0.4,
                                                    saturation=0.4)]
        transform_set += [transforms.RandomHorizontalFlip()]
        if args.autoaugment:
            transform_set += [auto_augment_transform(img_size=args.cfg['train_cfg']['crop_size'])]
        transform_set += [transforms.ToTensor()]
        if args.change_light:
            transform_set += [lighting]
        transform_set += [normalize]
        return transforms.Compose(transform_set)
    else:
        if args.cfg['test_cfg']['crop_type'] == 'resnest':
            return transforms.Compose([
                ECenterCrop(args.cfg['test_cfg']['crop_size']),
                transforms.ToTensor(),
                normalize
            ])
        elif args.cfg['test_cfg']['crop_type'] == 'normal':
            return transforms.Compose([
                        transforms.Resize(int(args.cfg['test_cfg']['crop_size']/0.875)),
                        transforms.CenterCrop(args.cfg['test_cfg']['crop_size']),
                        transforms.ToTensor(),
                        normalize,
                    ])
        elif args.cfg['test_cfg']['crop_type'] == 'tencrop':
            return transforms.Compose([
                        transforms.Resize(int(args.cfg['test_cfg']['crop_size']/0.875)),
                        transforms.TenCrop(args.cfg['test_cfg']['crop_size']),
                        TenCropToTensor(),
                        TenCropNormalize(normalize),
                    ])
        else:
            raise NotImplemented("The crop type {} is not implemented! Please select from "
                                 "[normal, resnest]".format(args.cfg['test_cfg']['crop_type']))


class Lighting(object):
    """Lighting noise (AlexNet-style PCA-based noise)

    Args:
        alphastd (float): The std of the normal distribution.
        eigval (array): The eigenvalue of RGB channel.
        eigvec (array): The eigenvec of RGB channel.

    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be processed.

        Returns:
            Tensor: Add lighting noise Tensor image.

        Code Reference: https://github.com/clovaai/CutMix-PyTorch/blob/master/utils.py
        """
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        return self.__class__.__name__ + '(alphastd={0})'.format(self.alphastd)


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize))#, interpolation=PIL.Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


class TenCropToTensor(object):

    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, imgs):

        imgs = list(imgs)
        for i in range(len(imgs)):
            imgs[i] = self.totensor(imgs[i])

        return torch.stack(imgs)


class TenCropNormalize(object):

    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, imgs):

        for i in range(len(imgs)):
            imgs[i] = self.normalize(imgs[i])

        return imgs


class RGB2BGR(object):

    def __call__(self, img):

        tmp = img[0, :, :].clone()
        img[0, :, :] = img[2, :, :]
        img[2, :, :] = tmp

        return img
