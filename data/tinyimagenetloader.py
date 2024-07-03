from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader
from .utils import TransformTwice, TransformFixMatch
from .utils import Solarize, Equalize, GaussianBlur
from .concat import ConcatDataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def find_classes_from_file(file_path):
    with open(file_path) as f:
        classes = f.readlines()
    classes = [x.strip() for x in classes]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, classes, class_to_idx):
    samples = []
    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                if 'JPEG' in path or 'jpg' in path:
                    samples.append(item)

    return samples

def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None, loader=pil_loader):

        if len(samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders \n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)

class TinyImageNetData:
    def __init__(self, args):
        # self.path = args.dataset_root
        # self.path = './data/datasets/tiny-imagenet-200/'
        self.path = args.dataset_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.image_size = args.image_size
        self.crop_pct = args.crop_pct
        self.offset = int((int(self.image_size / self.crop_pct) - self.image_size) / 2)
        self.interpolation = args.interpolation
        self.aug_pipeline = args.aug_type
        # self.tinyimagenet_mean, self.tinyimagenet_std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]    # trssl
        self.tinyimagenet_mean, self.tinyimagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # origianl
        print("Create TinyImageNet data factory")

        if self.aug_pipeline == 'vit_frost':
            print("Data Augmentation: use ViT-FRoST data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),  # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ]))
        elif self.aug_pipeline == 'vit_uno':
            print("Data Augmentation: use ViT-UNO data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),

                transforms.RandomGrayscale(p=0.2),                            # TRSSL
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),    # TRSSL

                # Solarize(p=0.1),    # aligned with cifar
                # Equalize(p=0.1),    # aligned with cifar

                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
            ])

            self.transform_supervised = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),

                transforms.RandomGrayscale(p=0.2),                            # TRSSL
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),    # TRSSL

                # Solarize(p=0.1),    # aligned with cifar
                # Equalize(p=0.1),    # aligned with cifar

                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
            ]))

            self.transform_fixmatch = TransformFixMatch(
                weak_transform=transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                        transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ]),
                strong_transform=transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                        transforms.RandomChoice([
                            transforms.RandomCrop(self.image_size, padding=self.offset),
                            transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                        ]),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),

                        transforms.RandomGrayscale(p=0.2),                            # TRSSL
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),    # TRSSL

                        # Solarize(p=0.1),    # aligned with cifar
                        # Equalize(p=0.1),    # aligned with cifar

                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ])
            )
        elif self.aug_pipeline == 'vit_uno_clip':
            print("Data Augmentation: use ViT-UNO Clip data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),

                transforms.RandomGrayscale(p=0.2),                            # TRSSL
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),    # TRSSL

                # Solarize(p=0.1),    # aligned with cifar
                # Equalize(p=0.1),    # aligned with cifar

                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
            ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),

                transforms.RandomGrayscale(p=0.2),                            # TRSSL
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),    # TRSSL

                # Solarize(p=0.1),    # aligned with cifar
                # Equalize(p=0.1),    # aligned with cifar

                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
            ]))
        else:
            print("Data Augmentation: use ResNet data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(64),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ])
            self.transform_once = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ])
            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.tinyimagenet_mean, std=self.tinyimagenet_std)
                ]))

    def find_classes_from_folder(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def find_classes_from_file(self, file_path):
        with open(file_path) as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, classes, class_to_idx):
        samples = []
        for target in classes:
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    if 'JPEG' in path or 'jpg' in path:
                        samples.append(item)

        return samples

    def get_dataset(self, split='train', aug=None, target_list=range(180), to_neighbors_dataset_indices=None):
        # img_split = 'images/'+subfolder
        if split == 'test':
            img_split = 'val'
        else:
            img_split = split

        classes_200, class_to_idx_200 = find_classes_from_file(os.path.join(self.path, 'tinyimagenet_200.txt'))
        classes_sel = [classes_200[i] for i in target_list]
        samples = make_dataset(self.path + img_split, classes_sel, class_to_idx_200)

        if aug == 'once':
            dataset = ImageFolder(transform=self.transform_once, samples=samples)
        elif aug == 'twice':
            dataset = ImageFolder(transform=self.transform_twice, samples=samples)
        elif aug == 'fixmatch':
            dataset = ImageFolder(transform=self.transform_fixmatch, samples=samples)
        elif aug == 'supervised':
            dataset = ImageFolder(transform=self.transform_supervised, samples=samples)
        else:
            dataset = ImageFolder(transform=self.transform_plain, samples=samples)

        if to_neighbors_dataset_indices is not None:
            # Dataset returns an image and one of its nearest neighbors.
            from data.scan_custom_dataset import NeighborsDataset
            dataset = NeighborsDataset(dataset, to_neighbors_dataset_indices, 50)

        return dataset

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(200),
                       to_neighbors_dataset_indices=None):
        print(f"Create TinyImageNet dataloader: split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}]")

        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                            pin_memory=False)
        return loader
