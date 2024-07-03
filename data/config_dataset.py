import math


def set_dataset_config(args, setting='msc_incd'):
    if args.dataset_name == 'cifar10':
        args.num_classes = 10
        args.dataset_root = './data/datasets/CIFAR/'
        args.val_split = 'train'
        args.test_split = 'test'

    elif args.dataset_name == 'cifar100':
        args.num_classes = 100
        args.dataset_root = './data/datasets/CIFAR/'
        args.val_split = 'train'
        args.test_split = 'test'

    elif args.dataset_name == 'tinyimagenet':
        args.num_classes = 200
        args.dataset_root = './data/datasets/tiny-imagenet-200/'
        args.val_split = 'train'
        args.test_split = 'val'

    elif args.dataset_name == 'cub200':
        args.num_classes = 200
        args.dataset_root = './data/datasets/CUB_200_2011/'
        args.val_split = 'val'
        args.test_split = 'test'

    elif args.dataset_name == 'herb19':
        args.num_classes = 683
        args.dataset_root = './data/datasets/herbarium_19/'
        args.val_split = 'val'
        args.test_split = 'test'

    elif args.dataset_name == 'scars':
        args.num_classes = 196
        args.dataset_root = './data/datasets/stanford_car/'
        args.val_split = 'val'
        args.test_split = 'test'

    elif args.dataset_name == 'aircraft':
        args.num_classes = 100
        args.dataset_root = './data/datasets/aircraft/fgvc-aircraft-2013b/'
        args.val_split = 'val'
        args.test_split = 'test'

    if setting == 'ncd':
        args.num_base = (args.num_steps-1) * math.ceil(args.num_classes/args.num_steps)
        args.num_novel = args.num_classes - args.num_base

    return args


def set_dataset_config_cluster(args, setting='msc_incd'):
    if args.dataset_name == 'cifar10':
        args.num_classes = 10
        args.dataset_root = '../global_datasets/CIFAR/'
        args.val_split = 'train'
        args.test_split = 'test'

    elif args.dataset_name == 'cifar100':
        args.num_classes = 100
        args.dataset_root = '../global_datasets/CIFAR/'
        args.val_split = 'train'
        args.test_split = 'test'

    elif args.dataset_name == 'tinyimagenet':
        args.num_classes = 200
        args.dataset_root = '../global_datasets/tiny-imagenet-200/'
        args.val_split = 'train'
        args.test_split = 'val'

    elif args.dataset_name == 'cub200':
        args.num_classes = 200
        args.dataset_root = '../global_datasets/CUB_200_2011/'
        args.val_split = 'val'
        args.test_split = 'test'

    elif args.dataset_name == 'herb19':
        args.num_classes = 683
        args.dataset_root = '../global_datasets/herbarium_19/'
        args.val_split = 'val'
        args.test_split = 'test'

    elif args.dataset_name == 'scars':
        args.num_classes = 196
        args.dataset_root = '../global_datasets/stanford_car/'
        args.val_split = 'val'
        args.test_split = 'test'

    elif args.dataset_name == 'aircraft':
        args.num_classes = 100
        args.dataset_root = '../global_datasets/aircraft/fgvc-aircraft-2013b/'
        args.val_split = 'val'
        args.test_split = 'test'

    if setting == 'ncd':
        args.num_base = (args.num_steps-1) * math.ceil(args.num_classes/args.num_steps)
        args.num_novel = args.num_classes - args.num_base

    return args

