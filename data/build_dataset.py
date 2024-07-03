from __future__ import print_function
from .cifarloader import CIFAR10Data, CIFAR100Data
from .tinyimagenetloader import TinyImageNetData
from .cubloader import CustomCub2011Data
from .herbloader import Herb19Data
from .scarloader import SCarData
from .fgvcloader import FGVCAircraftData

def build_data(args):
    if args.dataset_name == 'cifar10':
        # |Y| == 10
        return CIFAR10Data(args)
    elif args.dataset_name == 'cifar100':
        # |Y| == 100
        return CIFAR100Data(args)
    elif args.dataset_name == 'tinyimagenet':
        # |Y| == 200
        return TinyImageNetData(args)
    elif args.dataset_name == 'cub200':
        # |Y| == 200
        return CustomCub2011Data(args)
    elif args.dataset_name == 'herb19':
        # |Y| == 683
        return Herb19Data(args)
    elif args.dataset_name == 'scars':
        # |Y| == 196
        return SCarData(args)
    elif args.dataset_name == 'aircraft':
        # |Y| == 100
        return FGVCAircraftData(args)
    else:
        print(f"CAN NOT FIND DATASET {args.dataset_name}")
        raise NotImplementedError