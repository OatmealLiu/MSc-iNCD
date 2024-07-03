from __future__ import print_function
from .der_dataloader import CIFAR10Data, CIFAR100Data, TinyImageNetData
from .der_cubloader import CustomCub2011Data
from .der_herbloader import Herb19Data

def build_data(args):
    if args.dataset_name == 'cifar10':
        return CIFAR10Data(args)
    elif args.dataset_name == 'cifar100':
        return CIFAR100Data(args)
    elif args.dataset_name == 'tinyimagenet':
        return TinyImageNetData(args)
    elif args.dataset_name == 'cub200':
        # |Y| == 200
        return CustomCub2011Data(args)
    elif args.dataset_name == 'herb19':
        # |Y| == 683
        return Herb19Data(args)
    else:
        print(f"CAN NOT FIND DATASET {args.dataset_name}")
        raise NotImplementedError