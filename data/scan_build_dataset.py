from __future__ import print_function
from .scan_dataloader import CIFAR10Data, CIFAR100Data, TinyImageNetData

def build_data(args):
    if args.dataset_name == 'cifar10':
        return CIFAR10Data(args)
    elif args.dataset_name == 'cifar100':
        return CIFAR100Data(args)
    elif args.dataset_name == 'tinyimagenet':
        return TinyImageNetData(args)
    else:
        print(f"CAN NOT FIND DATASET {args.dataset_name}")
        raise NotImplementedError