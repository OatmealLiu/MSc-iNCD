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

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
# from torch.utils.data import Dataset
import torch.utils.data as data

from data.utils import subsample_instances

from .utils import Solarize, Equalize
from .utils import TransformTwice, TransformFixMatch
import torchvision.transforms as transforms


class CustomCub2011(data.Dataset):
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, no_aug_transform=False, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        # self.target_transform = target_transform

        self.return_no_aug_transform = no_aug_transform
        self.no_aug_transform = transforms.Compose([
            transforms.Resize(int(224 / 0.875), 3),
            transforms.ToTensor()])

        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        original_img = img.copy()
        not_aug_img = self.no_aug_transform(original_img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None and self.return_no_aug_transform is True:
            img = self.transform(img)
            return img, deepcopy(img[0]), target, self.uq_idxs[idx]
        elif self.transform is not None and self.return_no_aug_transform is False:
            img = self.transform(img)
            return img, target, self.uq_idxs[idx]
        else:
            return img, not_aug_img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

class CustomCub2011Data:
    def __init__(self, args):
        self.root = args.dataset_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.image_size = args.image_size
        self.crop_pct = args.crop_pct
        self.offset = int((int(self.image_size / self.crop_pct) - self.image_size) / 2)
        self.interpolation = args.interpolation
        self.aug_pipeline = args.aug_type

        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # imagenet

        print("Create CUB-200-2011 data factory")
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

        self.whole_training_set_once = CustomCub2011(root=self.root, transform=self.transform_once, train=True)
        self.whole_training_set = CustomCub2011(root=self.root, transform=self.transform_twice,
                                                no_aug_transform=True, train=True)
        self.whole_test_dataset = CustomCub2011(root=self.root, transform=self.transform_plain, train=False)

    def get_dataset(self, split='train', aug=None, target_list=range(200), to_neighbors_dataset_indices=None):
        if aug == 'once':
            whole_train_dataset = subsample_classes(deepcopy(self.whole_training_set_once), include_classes=target_list)
        else:
            whole_train_dataset = subsample_classes(deepcopy(self.whole_training_set), include_classes=target_list)

        # subsample_indices = subsample_instances(whole_train_dataset)
        # whole_train_dataset = subsample_dataset(whole_train_dataset, subsample_indices)

        train_idxs, val_idxs = get_train_val_indices(whole_train_dataset)

        train_dataset = subsample_dataset(deepcopy(whole_train_dataset), train_idxs)
        if aug is None:
            train_dataset.transform = self.transform_plain

        val_dataset = subsample_dataset(deepcopy(whole_train_dataset), val_idxs)
        val_dataset.transform = self.transform_plain
        val_dataset.return_no_aug_transform = False

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

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(100),
                       to_neighbors_dataset_indices=None):
        # get dataset
        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)
        print(f"Create CUB-200-2011 dataloader:\n"
              f"split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}] "
              f"classes[{len(set(dataset.data['target'].values))}], "
              f"instances[{len(list(dataset.uq_idxs))}]")

        # get dataloader
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader
