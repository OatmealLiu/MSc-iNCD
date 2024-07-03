import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import cluster_acc, seed_torch
from utils import ramps
from utils.logging import Logger
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb
from collections.abc import Iterable
import math
from models.build_kmeans import build_kmeans
from data.build_dataset import build_data
from data.config_dataset import set_dataset_config

def test_kmeans(args, model, single_km_list, joint_km, single_train_loader_list, single_val_loader_list,
                single_test_loader_list, all_test_loader_list, step):
    print("=" * 100)
    print(f"\t\t\t\t\tCiao bella! I am Lower Bound KMeans step [{1+step}/{args.num_steps}]")
    print("=" * 100)

    # Single KMs: 25 25 25 25
    # Joint KMs : 25 50 75 100
    model.eval()

    # fit single_km for this step only
    all_train_feats = []
    print("---> Single KM Feature Extraction")
    for batch_idx, (x, label, _) in enumerate(tqdm(single_train_loader_list[-1])):
        x = x.to(args.device)
        feat = model(x)
        feat = feat.to(args.device)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        all_train_feats.append(feat.cpu().numpy())

    all_train_feats = np.concatenate(all_train_feats)

    print("---> Single KM Fitting")
    single_km_list[-1].fit(all_train_feats)

    # fit joint_km for accumulated steps
    all_train_feats = []
    print("---> Joint KM Feature Extraction")
    for single_train_loader in single_train_loader_list:
        for batch_idx, (x, label, _) in enumerate(tqdm(single_train_loader)):
            x = x.to(args.device)
            feat = model(x)
            feat = feat.to(device)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            all_train_feats.append(feat.cpu().numpy())

    all_train_feats = np.concatenate(all_train_feats)
    print("---> Joint KM Fitting")
    joint_km.fit(all_train_feats)

    args.head = 'head2'
    print('===========================================')
    print('             Head 2 Final Test             ')
    print('===========================================')
    print('Task-specific Head: test on unlabeled classes for this step only')
    acc_head2_ul, ind = test_disco(model, single_km_list[-1], joint_km, single_test_loader_list[-1], args,
                                   return_ind=True)

    args.head = 'head1'
    print('===========================================')
    print('             Head 1 Final Test             ')
    print('===========================================')
    print('Joint Head: test on unlabeled classes for this step only w/ clustering')
    acc_head1_ul_w_cluster = test_disco(model, single_km_list[-1], joint_km, single_test_loader_list[-1], args,
                                        cluster=True)

    print('Joint Head: test on unlabeled classes for this step only w/o clustering')
    acc_head1_ul_wo_cluster = test_disco(model, single_km_list[-1], joint_km, single_test_loader_list[-1], args,
                                         cluster=False, ind=ind)

    if step > 0:
        print('Joint Head: test on all previous discovered novel classes w/ clustering')
        acc_head1_prev_all_w_cluster = test_disco(model, single_km_list[-1], joint_km, all_test_loader_list[-2], args,
                                                  cluster=True)

        print('Joint Head: test on all previous discovered novel classes w/o clustering')
        acc_head1_prev_all_wo_cluster = test_disco_all(model, single_km_list[:-1], joint_km,
                                                       single_val_loader_list[:-1], single_test_loader_list[:-1], args)
    else:
        acc_head1_prev_all_w_cluster = -1
        acc_head1_prev_all_wo_cluster = -1

    print('Joint Head: test on all classes until this step w/ clustering')
    acc_head1_all_w_cluster = test_disco(model, single_km_list[-1], joint_km, all_test_loader_list[-1], args, cluster=True)

    print('Joint Head: test on all classes until this step w/o clustering')
    acc_head1_all_wo_cluster = test_disco_all(model, single_km_list, joint_km, single_val_loader_list,
                                              single_test_loader_list, args)

    # wandb metrics logging
    wandb.log({
        "test_acc/ACC_head2-current-novel_val": acc_head2_ul,
        "test_acc/ACC_head1-current-novel_test w/ clustering": acc_head1_ul_w_cluster,
        "test_acc/ACC_head1-current-novel_test w/o clustering": acc_head1_ul_wo_cluster,
        "test_acc/ACC_head1-all-prev w/ clustering": acc_head1_prev_all_w_cluster,
        "test_acc/ACC_head1-all-prev w/o clustering": acc_head1_prev_all_wo_cluster,
        "test_acc/ACC_head1-all w/ clustering": acc_head1_all_w_cluster,
        "test_acc/ACC_head1-all w/o clustering": acc_head1_all_wo_cluster,
    }, step=step)

    print('\n=========================================================')
    print(f'    Final Test Output at step [{1+step}/{args.num_steps}]')
    print('Head2')
    print(f"ACC_head2-current-novel_val = {acc_head2_ul}")

    print('Head1')
    print(f"ACC_head1-current-novel_test w/ clustering = {acc_head1_ul_w_cluster}")
    print(f"ACC_head1-current-novel_test w/o clustering= {acc_head1_ul_wo_cluster}")

    print(f"\nACC_head1-all-prev w/ clustering = {acc_head1_prev_all_w_cluster}")
    print(f"ACC_head1-all-prev w/o clustering = {acc_head1_prev_all_wo_cluster}")

    print(f"\nACC_head1-all w/ clustering = {acc_head1_all_w_cluster}")
    print(f"ACC_head1-all w/o clustering = {acc_head1_all_wo_cluster}")
    print('\n=========================================================')

    return single_km_list[-1], joint_km


def test_disco(model, single_km, joint_km_head, test_loader, args, cluster=True, ind=None, return_ind=False):
    model.eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = model(x)
        feat = feat.to(args.device)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        feat = feat.cpu().numpy()

        output1 = joint_km_head.predict(feat)
        output2 = single_km.predict(feat)

        if args.head == 'head1':
            output = output1
        else:
            output = output2

        # _, pred = output.max(1)
        pred = output
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred)

    if cluster:
        if return_ind:
            acc, ind = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        else:
            acc = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    else:
        if ind is not None:
            ind = ind[:args.num_novel_per_step, :]
            idx = np.argsort(ind[:, 1])
            id_map = ind[idx, 0]
            id_map += args.current_novel_start

            targets_new = np.copy(targets)
            for i in range(args.num_novel_per_step):
                targets_new[targets == i + args.current_novel_start] = id_map[i]
            targets = targets_new

        preds = torch.from_numpy(preds)
        targets = torch.from_numpy(targets)
        correct = preds.eq(targets).float().sum(0)
        acc = float(correct / targets.size(0))
        print('Test acc {:.4f}'.format(acc))

    if return_ind:
        return acc, ind
    else:
        return acc


def test_disco_all_val(model, single_dino_heads_list, joint_dino_head, test_loader_list, args):
    if len(single_dino_heads_list) != len(test_loader_list):
        print("-------->> ERROR: len(single_dino_heads_list) != len(test_loader_list)")
        return -1

    model.eval()
    joint_dino_head.eval()
    for single_dino_head in single_dino_heads_list:
        single_dino_head.eval()

    steps = len(test_loader_list)
    acc = 0.
    num_discovered = 0
    for s in range(steps):
        preds = np.array([])
        targets = np.array([])

        preds_ = np.array([])
        targets_ = np.array([])

        for batch_idx, (x, label, _) in enumerate(tqdm(test_loader_list[s])):
            x, label = x.to(args.device), label.to(args.device)

            # forward inference
            feat = model(x)
            output1 = joint_dino_head(feat)
            output2 = single_dino_heads_list[s](feat)

            # Task-specific head prediction
            _, pred_ = output2.max(1)
            targets_ = np.append(targets_, label.cpu().numpy())
            preds_ = np.append(preds_, pred_)

            # Joint head prediction
            _, pred = output1.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred)

        this_acc_, ind = cluster_acc(targets_.astype(int), preds_.astype(int), True)

        # organize
        this_num_novel = args.num_novel_interval if int(1 + s) < args.num_steps else args.num_novel_per_step
        this_num_base = args.num_novel_interval * s

        if args.dataset_name == 'cub200' or args.dataset_name == 'herb19':
            targets += this_num_base

        ind = ind[:this_num_novel, :]
        idx = np.argsort(ind[:, 1])
        id_map = ind[idx, 0]
        id_map += this_num_base

        targets_new = np.copy(targets)
        for i in range(args.num_novel_per_step):
            targets_new[targets == i + this_num_base] = id_map[i]
        targets = targets_new

        preds = torch.from_numpy(preds)
        targets = torch.from_numpy(targets)
        correct = preds.eq(targets).float().sum(0)
        this_acc = float(correct / targets.size(0))
        acc += this_acc * this_num_novel
        num_discovered += this_num_novel

    acc /= num_discovered
    print('Test acc w/o clustering {:.4f}'.format(acc))
    return acc


def test_disco_all(model, single_km_list, joint_km_head, val_loader_list, test_loader_list, args):
    model.eval()

    steps = len(test_loader_list)
    acc = 0.
    num_discovered = 0
    for s in range(steps):
        # assignment generation
        preds_ = np.array([])
        targets_ = np.array([])

        for batch_idx, (x, label, _) in enumerate(tqdm(val_loader_list[s])):
            x, label = x.to(args.device), label.to(args.device)

            # forward inference
            feat = model(x)
            feat = feat.to(args.device)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            feat = feat.cpu().numpy()

            output2 = single_km_list[s].predict(feat)


            # Task-specific head prediction
            # _, pred_ = output2.max(1)
            pred_ = output2
            targets_ = np.append(targets_, label.cpu().numpy())
            preds_ = np.append(preds_, pred_)

        this_acc_, ind = cluster_acc(targets_.astype(int), preds_.astype(int), True)

        # actual test
        preds = np.array([])
        targets = np.array([])
        for batch_idx, (x, label, _) in enumerate(tqdm(test_loader_list[s])):
            x, label = x.to(args.device), label.to(args.device)
            # forward inference
            feat = model(x)
            feat = feat.to(args.device)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            feat = feat.cpu().numpy()
            output1 = joint_km_head.predict(feat)

            # Joint head prediction
            # _, pred = output1.max(1)
            pred = output1
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred)

        # organize
        this_num_novel = args.num_novel_interval if int(1 + s) < args.num_steps else args.num_novel_per_step
        this_num_base = args.num_novel_interval * s

        if args.dataset_name == 'cub200' or args.dataset_name == 'herb19':
            targets += this_num_base

        ind = ind[:this_num_novel, :]
        idx = np.argsort(ind[:, 1])
        id_map = ind[idx, 0]
        id_map += this_num_base

        targets_new = np.copy(targets)
        for i in range(args.num_novel_per_step):
            targets_new[targets == i + this_num_base] = id_map[i]
        targets = targets_new

        preds = torch.from_numpy(preds)
        targets = torch.from_numpy(targets)
        correct = preds.eq(targets).float().sum(0)
        this_acc = float(correct / targets.size(0))
        acc += this_acc * this_num_novel
        num_discovered += this_num_novel

    acc /= num_discovered
    print('Test acc w/o clustering {:.4f}'.format(acc))
    return acc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--batch_size', default=256, type=int)

    # Dataset Setting
    parser.add_argument('--dataset_name', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                                 'cub200', 'herb19', 'scars',
                                                                                 'aircraft'])
    # parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    # parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--aug_type', type=str, default='vit_uno', choices=['vit_frost', 'vit_uno', 'resnet'])
    parser.add_argument('--num_workers', default=2, type=int)

    # Strategy Setting
    parser.add_argument('--num_steps', default=10, type=int)

    # Model Config
    parser.add_argument('--km_max_iter', default=300, type=int)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino')
    parser.add_argument('--grad_from_block', type=int, default=12)  # 12->do not fine tune backbone at all
    parser.add_argument('--num_mlp_layers', type=int, default=3)  # 12->do not fine tune backbone at all
    parser.add_argument('--dino_pretrain_path', type=str,
                        default='./models/dino_weights/dino_vitbase16_pretrain.pth')

    # Experimental Setting
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--exp_root', type=str, default='./outputs_kmeans/')
    parser.add_argument('--exp_marker', type=str, default='Kmeans')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')

    # ----------------------
    # Initial Configurations
    # ----------------------
    args = parser.parse_args()

    # init. dataset config.
    args = set_dataset_config(args)

    # init. config.
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    # WandB setting
    # Mark as lower bound as exp_marker
    wandb_run_name = f'KMeans_{args.dataset_name}_{args.num_steps}-Steps_{args.model_name}'
    wandb.init(project='Lower_KMeans',
               entity=args.wandb_entity,
               tags=[f'TotalStep={args.num_steps}', args.dataset_name],
               name=wandb_run_name,
               mode=args.wandb_mode)

    # ----------------------
    # Experimental Setting Initialization
    # ----------------------
    # ViT DINO B/16 Params
    # Parameters
    args.image_size = 224
    args.interpolation = 3
    args.crop_pct = 0.875
    args.pretrain_path = args.dino_pretrain_path
    # args.feat_dim = 768
    # args.mlp_out_dim = args.num_novel_per_step

    # ----------------------
    # Dataloaders Creation for this iNCD step
    # ----------------------
    args.num_novel_interval = math.ceil(args.num_classes / args.num_steps)

    data_factory = build_data(args)

    val_split = args.val_split
    test_split = args.test_split

    single_train_loader_list = []
    single_val_loader_list = []
    single_test_loader_list = []
    all_test_loader_list = []
    for s in range(args.num_steps):
        if (1+s) < args.num_steps:
            this_target_list = range(s*args.num_novel_interval, (1+s)*args.num_novel_interval)
            this_all_end = range((1+s)*args.num_novel_interval)
        else:
            this_target_list = range(s*args.num_novel_interval, args.num_classes)
            this_all_end = range(args.num_classes)

        this_train_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=True,
                                                        target_list=this_target_list)

        this_val_loader = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                      target_list=this_target_list)

        this_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                       target_list=this_target_list)

        this_all_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                           target_list=this_all_end)



        single_train_loader_list.append(this_train_loader)
        single_val_loader_list.append(this_val_loader)
        single_test_loader_list.append(this_test_loader)
        all_test_loader_list.append(this_all_test_loader)

    # ----------------------
    # ViT Model and DINO Projection Head Creations for this iNCD step
    # ----------------------
    model, single_km_list, joint_km_list = build_kmeans(args)

    for s in range(args.num_steps):
        args.current_novel_start = args.num_novel_interval * s
        args.current_novel_end = args.num_novel_interval * (s + 1) \
            if args.num_novel_interval * (s + 1) <= args.num_classes \
            else args.num_classes

        args.num_novel_per_step = args.current_novel_end - args.current_novel_start

        # Single KMs: 25 25 25 25
        # Joint KMs : 25 50 75 100
        single_km_list[s], joint_km_list[s] = test_kmeans(args, model, single_km_list[:(1+s)], joint_km_list[s],
                                                          single_train_loader_list[:(1+s)],
                                                          single_val_loader_list[:(1+s)],
                                                          single_test_loader_list[:(1+s)],
                                                          all_test_loader_list[:(1+s)],
                                                          s)
