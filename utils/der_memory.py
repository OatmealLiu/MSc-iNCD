# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data.utils import Solarize, Equalize, GaussianBlur


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, args, mode='reservoir'):
        assert mode in ('ring', 'reservoir')
        self.buffer_size = args.buffer_size
        self.device = args.device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

        # data augmentation for implicit consistency
        self.image_size = args.image_size
        self.crop_pct = args.crop_pct
        self.offset = int((int(self.image_size / self.crop_pct) - self.image_size) / 2)
        self.interpolation = args.interpolation

        if args.dataset_name == 'cifar10':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
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
        elif args.dataset_name == 'cifar100':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
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
        elif args.dataset_name == 'tinyimagenet':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(self.image_size / self.crop_pct), self.interpolation),    # 40 -> 224/0.875
                transforms.RandomChoice([
                    transforms.RandomCrop(self.image_size, padding=self.offset),
                    transforms.RandomResizedCrop(self.image_size, (0.5, 1.0))
                ]),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),

                # transforms.RandomGrayscale(p=0.2),                            # TRSSL
                # transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),    # TRSSL

                # Solarize(p=0.1),    # aligned with cifar
                # Equalize(p=0.1),    # aligned with cifar

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                # print(self.examples[index].shape)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                    # print(self.logits[index].shape)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

        # print(f"---[DEBUG]--->>> BufferSize = {min(self.num_seen_examples, self.buffer_size)}")
        # print(f"                 Example    = {self.examples[-1]}")
        # print(f"                 Logits     = {self.logits[-1]}")



    def get_data(self, size: int, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param return_index: if return index
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)

        ret_tuple = (torch.stack([self.transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_data_by_index(self, indexes) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :return:
        """
        ret_tuple = (torch.stack([self.transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :return: a tuple with all the items in the memory buffer
        """
        ret_tuple = (torch.stack([self.transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
