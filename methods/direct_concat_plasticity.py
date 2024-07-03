import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
import os
import sys
import copy
import wandb

from methods.testers import test_cluster, test_ind_cluster_unlocked

class DirectConcatPlasticity:
    def __init__(self, model, single_head, joint_head, prev_pair_list, sinkhorn, train_loader, ulb_step_val_list,
                 ulb_all_prev_val, ulb_all_val, ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.model_locker = True                # T=locked, F=unlocked

        self.single_head = single_head
        self.single_head_locker = False         # T=locked, F=unlocked
        self.joint_head = joint_head
        self.prev_pair_list = prev_pair_list

        # Sinkhorn pseudo-labeling
        self.sinkhorn = sinkhorn

        # Data loaders
        # |- train: joint training loader
        self.train_loader = train_loader
        # |- val
        self.ulb_step_val_list = ulb_step_val_list
        self.ulb_all_prev_val = ulb_all_prev_val
        self.ulb_all_val = ulb_all_val
        # |- test
        self.ulb_step_test_list = ulb_step_test_list
        self.ulb_all_prev_test = ulb_all_prev_test
        self.ulb_all_test = ulb_all_test

    def calculate_ce_zero_padding(self, output, target, softmax_temp=0.1):
        # follow original UNO, temperature = 0.1
        preds = F.softmax(output / softmax_temp, dim=1)  # temperature
        preds = torch.clamp(preds, min=1e-8)
        preds = torch.log(preds)
        loss = -torch.mean(torch.sum(target * preds, dim=1))
        return loss

    def calculate_weighted_avg(self, step_acc_list, args):
        acc = 0.
        num_discovered = 0
        for s in range(len(step_acc_list)):
            this_num_novel = args.num_novel_interval if int(1 + s) < args.num_steps else args.num_novel_per_step
            acc += step_acc_list[s] * this_num_novel
            num_discovered += this_num_novel

        acc /= num_discovered
        return acc

    def unlock_backbone(self, args):
        for name, m in self.model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

        print("---> unlocked backbone")
        return False

    def lock_head(self, args):
        for m in self.single_head.parameters():
            m.requires_grad = False
        print("---> locked classifier")
        return True

    def concat_heads(self, args):
        for step in range(args.current_step):
            w_saved = self.prev_pair_list[step][1].last_layer.weight.data.clone()
            self.joint_head.last_layer.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w_saved)

        current_w = self.single_head.last_layer.weight.data.clone()
        self.joint_head.last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w)


    def train(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am Direct Concat Plasticity [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.single_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        for epoch in range(args.epochs):
            if self.model_locker and epoch >= args.unlock_epoch:
                self.model_locker = self.unlock_backbone(args)
                if args.lock_head and not self.single_head_locker:
                    self.single_head_locker = self.lock_head(args)

            # create loss statistics recorder for each loss
            loss_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.model.train()
            self.single_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    weight_temp = self.single_head.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.single_head.last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                # Single head prediction
                output_v0 = self.single_head(feat_v0)
                output_v1 = self.single_head(feat_v1)

                # cross pseudo-labeling
                target_v0 = self.sinkhorn(output_v1)
                target_v1 = self.sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                # UNO Loss Calculation
                # follow original UNO, temperature = 0.1
                loss = self.calculate_ce_zero_padding(mixed_logits, mixed_targets, softmax_temp=args.softmax_temp)

                loss_record.update(loss.item(), x_v0.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                "loss_single/uno": loss_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_record.avg))
            print('===========================================')
            self.save_backbone(args.save_model_path)
            self.save_single_head(args.save_single_head_path)

            print('------>[Single Head] This Step Test W/ Clustering')
            # Only test current step
            this_step_val_w = test_cluster(self.model, self.single_head, self.ulb_step_val_list[args.current_step],
                                           args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                "single_val_acc/this_step_ulb_val_w_clustering": this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {this_step_val_w}")
            print('======================================')

        # Finished training
        # Append the newly learned single model and single head into the model-head pair list for evaluation
        self.prev_pair_list.append((self.model, self.single_head))

    def test(self, args):
        # === Single Head ===
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.model, self.single_head,
                                                           self.ulb_step_test_list[args.current_step], args,
                                                           return_ind=False)

        # === Joint Head ===
        self.concat_heads(args)

        print('------>[Joint Head] Single Step Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_w = test_cluster(self.model, self.joint_head, self.ulb_step_test_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------>[Joint Head] Single Step Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_wo = test_ind_cluster_unlocked(test_model=self.model, test_head=self.joint_head,
                                                          ind_gen_model=self.prev_pair_list[s][0],
                                                          ind_gen_head=self.prev_pair_list[s][1],
                                                          test_loader=self.ulb_step_test_list[s], step=s, args=args,
                                                          ind_gen_loader=self.ulb_step_val_list[s])

            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        if args.current_step > 0:
            print('------>[Joint Head] All-Prev-Steps Test W/ Clustering')
            acc_all_prev_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_test, args)
        else:
            acc_all_prev_test_w_cluster = -1

        print('------>[Joint Head] All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_test, args)

        print('------>[Joint Head] All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + args.current_step)]

        if args.current_step > 0:
            acc_all_prev_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        else:
            acc_all_prev_test_wo_cluster = -1

        acc_all_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list, args)


        print('\n========================================================')
        print('             Final Test Output (test split)             ')
        print(f'[S{args.current_step}-Single Head]')
        print(f"Acc_this_step             = {acc_single_head_this_step_w_cluster}")

        print(f'\n[S{args.current_step}-Joint Head]')
        print('All-Previous-Discovered-Test')
        print(f"Acc_all_prev_W_cluster    = {acc_all_prev_test_w_cluster}")
        print(f"Acc_all_prev_WO_cluster   = {acc_all_prev_test_wo_cluster}")

        print('\nAll-Discovered-Test')
        print(f"Acc_all_W_cluster         = {acc_all_test_w_cluster}")
        print(f"Acc_all_WO_cluster        = {acc_all_test_wo_cluster}")

        print('\nStepwise-Discovered')
        print('Step Single Test w/ clustering dict')
        print(acc_step_test_w_cluster_dict)

        print('Step Single Test w/o clustering dict')
        print(acc_step_test_wo_cluster_dict)
        print('========================================================')

    def eval(self):
        pass

    def save_backbone(self, path):
        torch.save(self.model.state_dict(), path)
        print("Task-specific single backbone saved to {}.".format(path))

    def save_single_head(self, path):
        torch.save(self.single_head.state_dict(), path)
        print("Task-specific single head saved to {}.".format(path))

    def save_joint_head(self, path):
        torch.save(self.joint_head.state_dict(), path)
        print("Task-agnostic joint head saved to {}.".format(path))
