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
import torch
import torch.utils.data as data
from .utils import download_url, check_integrity
from .utils import TransformTwice, RandomTranslateWithReflect, TransformFixMatch
from .concat import ConcatDataset
from .utils import Solarize, Equalize, GaussianBlur
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, split='train+test',
                 transform=None, target_transform=None,
                 download=True, target_list = range(5)):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        downloaded_list = []
        if split == 'train':
            downloaded_list = self.train_list
        elif split == 'test':
            downloaded_list = self.test_list
        elif split == 'train+test':
            downloaded_list.extend(self.train_list)
            downloaded_list.extend(self.test_list)

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    #  self.targets.extend(entry['coarse_labels'])
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        ind = [i for i in range(len(self.targets)) if self.targets[i] in target_list]

        self.data = self.data[ind]
        self.targets = np.array(self.targets)
        self.targets = self.targets[ind].tolist()



    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        #  x = self.class_to_idx
        #  sorted_x = sorted(x.items(), key=lambda kv: kv[1])
        #  print(sorted_x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {
            'image': img,
            'target': target,
            'meta': {
                   'im_size': img_size,
                   'index': index
            }
        }
        return out

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        #  'key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class CIFAR10Data:
    def __init__(self, args):
        self.root = args.dataset_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.image_size = args.image_size
        self.crop_pct = args.crop_pct
        self.offset = int((int(self.image_size / self.crop_pct) - self.image_size) / 2)
        self.interpolation = args.interpolation
        self.aug_pipeline = args.aug_type

        print("Create cifar-10 data factory")

        if self.aug_pipeline == 'vit_frost':
            print("Data Augmentation: use ViT-FRoST data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                # transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                RandomTranslateWithReflect(self.offset),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                # transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                RandomTranslateWithReflect(self.offset),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
        elif self.aug_pipeline == 'vit_uno':
            print("Data Augmentation: use ViT-UNO data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))

            self.transform_fixmatch = TransformFixMatch(
                weak_transform= transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),  # 40 -> 224/0.875
                        # transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                        RandomTranslateWithReflect(self.offset),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]),
                strong_transform = transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),  # 40 -> 224/0.875
                        transforms.RandomChoice([
                            transforms.RandomCrop(self.image_size, padding=self.offset),
                            transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                        ]),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                        Solarize(p=0.1),
                        Equalize(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
            )
        elif self.aug_pipeline == 'vit_uno_clip':
            print("Data Augmentation: use ViT-UNO Clip data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            print("Data Augmentation: use ResNet data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            self.transform_once = transforms.Compose([
                RandomTranslateWithReflect(4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            self.transform_twice = TransformTwice(transforms.Compose([
                RandomTranslateWithReflect(4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]))

    def get_dataset(self, split='train', aug=None, target_list=range(5), to_neighbors_dataset_indices=None):
        if aug == 'once':
            # print(self.transform_once)
            dataset = CIFAR10(root=self.root, split=split, transform=self.transform_once, target_list=target_list)
        elif aug == 'twice':
            # print(self.transform_twice)
            dataset = CIFAR10(root=self.root, split=split, transform=self.transform_twice, target_list=target_list)
        elif aug == 'fixmatch':
            dataset = CIFAR10(root=self.root, split=split, transform=self.transform_fixmatch, target_list=target_list)
        elif aug == 'scan':
            from data.scan_custom_dataset import NeighborsDataset
            dataset = CIFAR10(root=self.root, split=split, transform=self.transform_once, target_list=target_list)
            if to_neighbors_dataset_indices is None:
                raise ValueError('Invalid neighbors dataset indices')
            dataset = NeighborsDataset(dataset, to_neighbors_dataset_indices, 20)
        elif aug == 'selflabel':
            from data.scan_custom_dataset import AugmentedDataset
            dataset = CIFAR10(root=self.root, split=split,
                              transform={'standard': self.transform_plain, 'augment': self.transform_once},
                              target_list=target_list)
            dataset = AugmentedDataset(dataset)
        else:
            # print(self.transform_plain)
            dataset = CIFAR10(root=self.root, split=split, transform=self.transform_plain, target_list=target_list)

        return dataset

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(5),
                       to_neighbors_dataset_indices=None):
        print(f"Create cifar-10 dataloader: split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}]")

        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)

        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader, len(dataset)

    def get_dataloader_mixed(self, split='train', aug=None, shuffle=True, labeled_list=range(5),
                             unlabeled_list=range(5, 10)):
        dataset_lb = self.get_dataset(split=split, aug=aug, target_list=labeled_list)
        dataset_ulb = self.get_dataset(split=split, aug=aug, target_list=unlabeled_list)
        # mix-up
        dataset_lb.targets = np.concatenate((dataset_lb.targets, dataset_ulb.targets))
        dataset_lb.data = np.concatenate((dataset_lb.data, dataset_ulb.data), 0)
        loader = data.DataLoader(dataset_lb, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader


class CIFAR100Data:
    def __init__(self, args):
        self.root = args.dataset_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.image_size = args.image_size
        self.crop_pct = args.crop_pct
        self.offset = int((int(self.image_size / self.crop_pct) - self.image_size) / 2)
        self.interpolation = args.interpolation
        self.aug_pipeline = args.aug_type

        print("Create cifar-100 data factory")
        if self.aug_pipeline == 'vit_frost':
            print("Data Augmentation: use ViT-FRoST data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),  # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]))
        elif self.aug_pipeline == 'vit_uno':
            print("Data Augmentation: use ViT-UNO data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

            self.transform_once = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]))

            self.transform_fixmatch = TransformFixMatch(
                weak_transform=transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                        transforms.RandomCrop(self.image_size, padding=self.offset),    # 224/0.875 -random-> 224
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]),
                strong_transform=transforms.Compose([
                        transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                        transforms.RandomChoice([
                            transforms.RandomCrop(self.image_size, padding=self.offset),
                            transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                        ]),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                        Solarize(p=0.1),
                        Equalize(p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            )
        elif self.aug_pipeline == 'vit_uno_clip':
            print("Data Augmentation: use ViT-UNO Clip data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.Resize(int(self.image_size / self.crop_pct), interpolation=Image.BICUBIC),    # 40 -> 224/0.875
                transforms.CenterCrop(self.image_size),                     # 224/0.875 -center-> 224
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
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
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
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
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]))
        else:
            print("Data Augmentation: use ResNet data aug. pipeline")
            self.transform_plain = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_once = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_twice = TransformTwice(transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ]))

    def get_dataset(self, split='train', aug=None, target_list=range(100), to_neighbors_dataset_indices=None):
        if aug == 'once':
            dataset = CIFAR100(root=self.root, split=split, transform=self.transform_once, target_list=target_list)
        elif aug == 'twice':
            dataset = CIFAR100(root=self.root, split=split, transform=self.transform_twice, target_list=target_list)
        elif aug == 'fixtmatch':
            dataset = CIFAR100(root=self.root, split=split, transform=self.transform_fixmatch, target_list=target_list)
        elif aug == 'scan':
            from data.scan_custom_dataset import NeighborsDataset
            dataset = CIFAR100(root=self.root, split=split, transform=self.transform_once, target_list=target_list)
            if to_neighbors_dataset_indices is None:
                raise ValueError('Invalid neighbors dataset indices')
            dataset = NeighborsDataset(dataset, to_neighbors_dataset_indices, 20)
        elif aug == 'selflabel':
            from data.scan_custom_dataset import AugmentedDataset
            dataset = CIFAR100(root=self.root, split=split,
                               transform={'standard': self.transform_plain, 'augment': self.transform_once},
                               target_list=target_list)
            dataset = AugmentedDataset(dataset)
        else:
            dataset = CIFAR100(root=self.root, split=split, transform=self.transform_plain, target_list=target_list)

        return dataset

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(100),
                       to_neighbors_dataset_indices=None):
        print(f"Create cifar-100 dataloader: split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}]")

        # get dataset
        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)

        # get dataloader
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader, len(dataset)


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
        img_size = (img.shape[0], img.shape[1])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {
            'image': img,
            'target': target,
            'meta': {
                   'im_size': img_size,
                   'index': index
            }
        }
        return out

    def __len__(self):
        return len(self.samples)

class TinyImageNetData:
    def __init__(self, args):
        # self.path = args.dataset_root
        self.path = './data/datasets/tiny-imagenet-200/'
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

    def get_dataset(self, split='train', aug=None, target_list=range(180),
                    to_neighbors_dataset_indices=None):
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
        elif aug == 'scan':
            from data.scan_custom_dataset import NeighborsDataset
            dataset = ImageFolder(transform=self.transform_once, samples=samples)
            if to_neighbors_dataset_indices is None:
                raise ValueError('Invalid neighbors dataset indices')
            dataset = NeighborsDataset(dataset, to_neighbors_dataset_indices, 50)
        elif aug == 'selflabel':
            from data.scan_custom_dataset import AugmentedDataset
            dataset = ImageFolder(transform={'standard': self.transform_plain, 'augment': self.transform_once},
                                  samples=samples)
            dataset = AugmentedDataset(dataset)
        else:
            dataset = ImageFolder(transform=self.transform_plain, samples=samples)

        return dataset

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(200),
                       to_neighbors_dataset_indices=None):
        print(f"Create TinyImageNet dataloader: split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}]")

        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                            pin_memory=False)
        return loader, len(dataset)
