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

import pandas as pd
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
# from torch.utils.data import Dataset
import torch.utils.data as data

from data.utils import subsample_instances

from .utils import Solarize, Equalize
from .utils import TransformTwice, TransformFixMatch
import torchvision.transforms as transforms


class CarsDataset(data.Dataset):
    """
        Cars Dataset
    """
    def __init__(self, data_dir, metas, train=True, limit=0, transform=None):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        # self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


class SCarData:
    def __init__(self, args):
        # self.root = args.dataset_root
        self.car_root = args.dataset_root + "cars_{}/"
        self.meta_default_path = args.dataset_root + "devkit/cars_{}.mat"
        # self.car_root = "./data/datasets/stanford_car/cars_{}/"
        # self.meta_default_path = "./data/datasets/stanford_car/devkit/cars_{}.mat"
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.image_size = args.image_size
        self.crop_pct = args.crop_pct
        self.offset = int((int(self.image_size / self.crop_pct) - self.image_size) / 2)
        self.interpolation = args.interpolation
        self.aug_pipeline = args.aug_type

        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # imagenet

        print("Create Stanford Cars data factory")
        if self.aug_pipeline == 'vit_frost':
            print("Data Augmentation: use ViT-FRoST data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),  # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]))
        elif self.aug_pipeline == 'vit_uno':
            print("Data Augmentation: use ViT-UNO data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                # Solarize(p=0.1),
                # Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            self.transform_supervised = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                # Solarize(p=0.1),
                # Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]))

            self.transform_fixmatch = TransformFixMatch(
                weak_transform=transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                        transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.mean, std=self.std),
                ]),
                strong_transform=transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                        transforms.RandomChoice([
                            transforms.RandomCrop(self.image_size, padding=self.offset),
                            transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                        ]),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                        # Solarize(p=0.1),
                        # Equalize(p=0.1),
                        transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])
            )
        elif self.aug_pipeline == 'vit_uno_clip':
            print("Data Augmentation: use ViT-UNO Clip data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]))
        else:
            print("Data Augmentation: use ResNet data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            self.transform_once = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]))

        self.whole_training_set_once = CarsDataset(data_dir=self.car_root, metas=self.meta_default_path, transform=self.transform_once,
                                              train=True)
        self.whole_training_set = CarsDataset(data_dir=self.car_root, metas=self.meta_default_path, transform=self.transform_twice,
                                              train=True)
        self.whole_test_dataset = CarsDataset(data_dir=self.car_root, metas=self.meta_default_path, transform=self.transform_plain,
                                              train=False)

    def get_dataset(self, split='train', aug=None, target_list=range(200), to_neighbors_dataset_indices=None):
        if aug == 'once':
            whole_train_dataset = subsample_classes(deepcopy(self.whole_training_set_once), include_classes=target_list)
        else:
            whole_train_dataset = subsample_classes(deepcopy(self.whole_training_set), include_classes=target_list)
        # subsample_indices = subsample_instances(whole_train_dataset)
        # whole_train_dataset = subsample_dataset(whole_train_dataset, subsample_indices)

        train_idxs, val_idxs = get_train_val_indices(whole_train_dataset)

        train_dataset = subsample_dataset(deepcopy(whole_train_dataset), train_idxs)

        val_dataset = subsample_dataset(deepcopy(whole_train_dataset), val_idxs)
        val_dataset.transform = self.transform_plain

        # Get test set for all classes
        test_dataset = subsample_classes(deepcopy(self.whole_test_dataset), include_classes=target_list)
        # print('\n------> Printing lens...')
        # print(f'        [train]: len={len(train_dataset)}\t classes={train_dataset.data["target"].values}')
        # print(f'        [val]: len={len(val_dataset)}\t classes={val_dataset.data["target"].values}')
        # print(f'        [test]: len={len(test_dataset)}\t classes={test_dataset.data["target"].values}')
        # print('\n')

        if split == 'train':
            dataset = train_dataset
        elif split == 'val':
            dataset = val_dataset
        elif split == 'test':
            dataset = test_dataset
        else:
            raise NotImplementedError

        if to_neighbors_dataset_indices is not None:
            # Dataset returns an image and one of its nearest neighbors.
            from data.scan_custom_dataset import NeighborsDataset
            dataset = NeighborsDataset(dataset, to_neighbors_dataset_indices, 20)

        # print(dataset.transform)
        return dataset

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(98),
                       to_neighbors_dataset_indices=None):
        # get dataset
        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)

        print(f"Create Stanford Cars dataloader:\n"
              f"split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}] "
              f"classes[{len(set(dataset.target))}], "
              f"instances[{len(set(dataset.uq_idxs))}]")

        # get dataloader
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader
