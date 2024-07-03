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


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(data.Dataset):

    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.

    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = True if split == 'train' else False

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[index]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = urllib.request.urlopen(self.url)

        # download .tar.gz file
        with open(tar_path, 'wb') as f:
            f.write(data.read())

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print('Renaming %s to %s ...' % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [(p, t) for i, (p, t) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(60)):

    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]

    # TODO: Don't transform targets for now
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    all_targets = [t for i, (p, t) in enumerate(train_dataset.samples)]
    train_classes = np.unique(all_targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(all_targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

class FGVCAircraftData:
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

        print("Create FGVC Aircraft data factory")
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
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
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
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                Solarize(p=0.1),
                Equalize(p=0.1),
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
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                        Solarize(p=0.1),
                        Equalize(p=0.1),
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
        self.whole_training_set_once = FGVCAircraft(root=self.root, transform=self.transform_once, split='trainval')

        self.whole_training_set = FGVCAircraft(root=self.root, transform=self.transform_twice, split='trainval')
        self.whole_test_dataset = FGVCAircraft(root=self.root, transform=self.transform_plain, split='test')

    def get_dataset(self, split='train', aug=None, target_list=range(50), to_neighbors_dataset_indices=None):
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

    def get_dataloader(self, split='train', aug=None, shuffle=True, target_list=range(50),
                       to_neighbors_dataset_indices=None):
        # get dataset
        dataset = self.get_dataset(split=split, aug=aug, target_list=target_list,
                                   to_neighbors_dataset_indices=to_neighbors_dataset_indices)
        print(f"Create FGVC Aircraft dataloader:\n"
              f"split[{split}] aug[{aug}], target_list[{target_list}], shuffle[{shuffle}] "
              f"classes[{len(set([i[1] for i in dataset.samples]))}], "
              f"instances[{len(set(dataset.uq_idxs))}]")

        # get dataloader
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return loader
