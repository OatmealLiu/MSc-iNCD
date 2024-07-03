import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb

from utils.scan_testers import test_cluster, test_ind_cluster
from utils.scan_losses import SCANLoss, ConfidenceBasedCE

class SCAN:
    def __init__(self, encoder, single_head, learned_single_heads_dict, joint_head_dict,
                 train_loader_scan, train_loader_selflabel,
                 ulb_step_val_list, ulb_all_prev_val, ulb_all_val,
                 ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.encoder = encoder
        self.single_head = single_head                          # to-be-trained
        self.learned_single_heads_dict = learned_single_heads_dict
        self.joint_head_dict = joint_head_dict

        # Data loaders
        # |- train
        self.train_loader_scan = train_loader_scan
        self.train_loader_selflabel = train_loader_selflabel
        # |- val
        self.ulb_step_val_list = ulb_step_val_list
        self.ulb_all_prev_val = ulb_all_prev_val
        self.ulb_all_val = ulb_all_val
        # |- test
        self.ulb_step_test_list = ulb_step_test_list
        self.ulb_all_prev_test = ulb_all_prev_test
        self.ulb_all_test = ulb_all_test

        self.results_scan = {'TaskSpecifc':  {'acc_this_step': -1.0},
                             'TaskAgnostic': {'acc_prev_w_cluster': -1.0,
                                              'acc_prev_wo_cluster': -1.0,
                                              'acc_all_w_cluster': -1.0,
                                              'acc_all_wo_cluster': -1.0,
                                              'acc_stepwise_w_cluster': {},
                                              'acc_stepwise_wo_cluster': {},
                                              }}

        self.results_selflabel = {'TaskSpecifc':  {'acc_this_step': -1.0},
                                  'TaskAgnostic': {'acc_prev_w_cluster': -1.0,
                                                   'acc_prev_wo_cluster': -1.0,
                                                   'acc_all_w_cluster': -1.0,
                                                   'acc_all_wo_cluster': -1.0,
                                                   'acc_stepwise_w_cluster': {},
                                                   'acc_stepwise_wo_cluster': {},
                                                   }}

    def train_scan(self, args):
        # create learnable parameter list
        param_list = list(self.encoder.parameters()) + list(self.single_head.parameters())
        # create optimizer SGD
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # create learning rate scheduler
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_scan, eta_min=args.lr * 1e-3)

        # create SCAN Loss
        criterion = SCANLoss(entropy_weight=args.w_entropy)     # 5.0 for cifar10, cifar100 and TinyImageNet
        criterion.to(args.device)

        for single_s in self.learned_single_heads_dict['scan']:
            single_s.eval()

        for epoch in range(args.epochs_scan):
            # create loss statistics recorder for each loss
            recorder_loss_total = AverageMeter()
            recorder_loss_consistency = AverageMeter()
            recorder_loss_entropy = AverageMeter()

            # switch the models to train mode
            self.encoder.train()
            self.single_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for i, batch in enumerate(tqdm(self.train_loader_scan)):
                # Forward pass
                anchors = batch['anchor'].cuda(non_blocking=True)
                neighbors = batch['neighbor'].cuda(non_blocking=True)
                #       | w/ weights normalization if args.use_norm
                if args.use_norm:
                    self.single_head.normalize_prototypes()
                #       |- feature extraction
                feat_anchors = self.encoder(anchors)
                feat_neighbors = self.encoder(neighbors)
                #       |- prediction
                output_anchors = self.single_head(feat_anchors)
                output_neighbors = self.single_head(feat_neighbors)
                #       |-Loss calculation
                loss_total, loss_consistency, loss_entropy = criterion(output_anchors, output_neighbors)

                recorder_loss_total.update(loss_total.item(), output_anchors.size(0))
                recorder_loss_consistency.update(loss_consistency.item(), output_anchors.size(0))
                recorder_loss_entropy.update(loss_entropy.item(), output_anchors.size(0))

                # Backward pass
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

            # wandb loss logging
            wandb.log({
                f"loss/total": recorder_loss_total.avg,
                f"loss/consistency": recorder_loss_consistency.avg,
                f"loss/entropy": recorder_loss_entropy.avg,
                # f"loss/SelfLabel_CE_confidence": 0.0,
            }, step=epoch)

            print('\n===========================================')
            print('\nStage-SCAN kNN-Consistency Learning Train Epoch [{}/{}]:'
                  'Avg Losses: Total={:.4f}, Consistency={:.4f}, Entropy={:.4f}'
                  .format(1+epoch, args.epochs_scan, recorder_loss_total.avg, recorder_loss_consistency.avg,
                          recorder_loss_entropy.avg))
            print('===========================================')

            # save single head
            self.save_single(path=args.save_single_path_scan)

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.encoder, self.single_head,
                                                      self.ulb_step_val_list[args.current_step],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                "val_acc/SCAN_single_head_this_step_W_cluster": acc_single_this_step_val_w,
                # "val_acc/SelfLabel_single_head_this_step_W_cluster": 0.0,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

        # Stage-wise update
        self.update_learned_heads_list(stage='scan')
        self.fill_joint_head(args, stage='scan')

    def train_selflabel(self, args):
        # create learnable parameter list
        param_list = list(self.encoder.parameters()) + list(self.single_head.parameters())
        # create optimizer SGD
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # create learning rate scheduler
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_selflabel, eta_min=args.lr * 1e-3)

        # create SCAN Loss
        criterion = ConfidenceBasedCE(args.confidence_threshold, args.apply_class_balancing)
        criterion.to(args.device)

        for single_s in self.learned_single_heads_dict['selflabel']:
            single_s.eval()

        for epoch in range(args.epochs_selflabel):
            # create loss statistics recorder for each loss
            recoder_loss = AverageMeter()

            # switch the models to train mode
            self.encoder.train()
            self.single_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for i, batch in enumerate(tqdm(self.train_loader_selflabel)):
                images = batch['image'].cuda(non_blocking=True)
                images_augmented = batch['image_augmented'].cuda(non_blocking=True)

                #       | w/ weights normalization if args.use_norm
                if args.use_norm:
                    self.single_head.normalize_prototypes()

                with torch.no_grad():
                    feat = self.encoder(images)
                    output = self.single_head(feat)

                feat_augmented = self.encoder(images_augmented)
                output_augmented = self.single_head(feat_augmented)

                loss = criterion(output, output_augmented)
                recoder_loss.update(loss, output.size(0))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # wandb loss logging
            wandb.log({
                f"loss/SelfLabel_CE_confidence": recoder_loss.avg,
                # f"loss/SCAN_total": 0.0,
                # f"loss/SCAN_consistency": 0.0,
                # f"loss/SCAN_entropy": 0.0,
            }, step=epoch+args.epochs_scan)

            print('\n===========================================')
            print('\nStage-SCAN Self-Labeling Learning Train Epoch [{}/{}]: Avg Losses: Total={:.4f}'.format(
                1+epoch, args.epochs_selflabel, recoder_loss.avg))
            print('===========================================')

            # save single head
            self.save_single(path=args.save_single_path_selflabel)

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.encoder, self.single_head,
                                                      self.ulb_step_val_list[args.current_step],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                # "val_acc/SCAN_single_head_this_step_W_cluster": 0.0,
                "val_acc/SelfLabel_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch+args.epochs_scan)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

        # Stage-wise update
        self.update_learned_heads_list(stage='selflabel')
        self.fill_joint_head(args, stage='selflabel')


    def test(self, args, stage='scan'):
        # === Single Head ===
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.encoder, self.single_head,
                                                           self.ulb_step_test_list[args.current_step], args,
                                                           return_ind=False)

        print('------>[Joint Head] Individual Steps Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_w = test_cluster(self.encoder, self.joint_head_dict[stage], self.ulb_step_test_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------>[Joint Head] Individual Steps Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_wo = test_ind_cluster(self.encoder, self.joint_head_dict[stage],
                                                 self.learned_single_heads_dict[stage][s],
                                                 self.ulb_step_test_list[s], s, args,
                                                 ind_gen_loader=self.ulb_step_val_list[s])
            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------> All-Prev-Steps Test W/ Clustering')
        if args.current_step > 0:
            acc_all_prev_test_w_cluster = test_cluster(self.encoder, self.joint_head_dict[stage],
                                                       self.ulb_all_prev_test, args)
        else:
            acc_all_prev_test_w_cluster = -1

        print('------> All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.encoder, self.joint_head_dict[stage], self.ulb_all_test, args)

        print('------> All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + args.current_step)]

        if args.current_step > 0:
            acc_all_prev_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        else:
            acc_all_prev_test_wo_cluster = -1

        acc_all_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list, args)

        # Update test results
        if stage == 'scan':
            self.results_scan['TaskSpecific'] = acc_single_head_this_step_w_cluster
            self.results_scan['TaskAgnostic']['acc_prev_w_cluster'] = acc_all_prev_test_w_cluster
            self.results_scan['TaskAgnostic']['acc_prev_wo_cluster'] = acc_all_prev_test_wo_cluster
            self.results_scan['TaskAgnostic']['acc_all_w_cluster'] = acc_all_test_w_cluster
            self.results_scan['TaskAgnostic']['acc_all_wo_cluster'] = acc_all_test_wo_cluster
            self.results_scan['TaskAgnostic']['acc_stepwise_w_cluster'] = acc_step_test_w_cluster_dict
            self.results_scan['TaskAgnostic']['acc_stepwise_wo_cluster'] = acc_step_test_wo_cluster_dict
        elif stage == 'selflabel':
            self.results_selflabel['TaskSpecific'] = acc_single_head_this_step_w_cluster
            self.results_selflabel['TaskAgnostic']['acc_prev_w_cluster'] = acc_all_prev_test_w_cluster
            self.results_selflabel['TaskAgnostic']['acc_prev_wo_cluster'] = acc_all_prev_test_wo_cluster
            self.results_selflabel['TaskAgnostic']['acc_all_w_cluster'] = acc_all_test_w_cluster
            self.results_selflabel['TaskAgnostic']['acc_all_wo_cluster'] = acc_all_test_wo_cluster
            self.results_selflabel['TaskAgnostic']['acc_stepwise_w_cluster'] = acc_step_test_w_cluster_dict
            self.results_selflabel['TaskAgnostic']['acc_stepwise_wo_cluster'] = acc_step_test_wo_cluster_dict
        else:
            raise ValueError(f'Wrong stage name: {stage}')

    def print_eval_results(self, args, stage='scan'):
        if stage == 'scan':
            results_dict = self.results_scan
        elif stage == 'selflabel':
            results_dict = self.results_selflabel
        else:
            raise ValueError(f'Wrong stage name: {stage}')

        print('\n========================================================')
        print(f'         {stage}-Stage Final Test Output (test split)             ')
        print(f'[S{args.current_step}-Single Head]')
        print(f"Acc_this_step             = {results_dict['TaskSpecific']}")

        print(f'\n[S{args.current_step}-Joint Head]')
        print('All-Previous-Discovered-Test')
        print(f"Acc_all_prev_W_cluster    = {results_dict['TaskAgnostic']['acc_prev_w_cluster']}")
        print(f"Acc_all_prev_WO_cluster   = {results_dict['TaskAgnostic']['acc_prev_wo_cluster']}")

        print('\nAll-Discovered-Test')
        print(f"Acc_all_W_cluster         = {results_dict['TaskAgnostic']['acc_all_w_cluster']}")
        print(f"Acc_all_WO_cluster        = {results_dict['TaskAgnostic']['acc_all_wo_cluster']}")

        print('\nStepwise-Discovered')
        print('Step Single Test w/ clustering dict')
        print(results_dict['TaskAgnostic']['acc_stepwise_w_cluster'])

        print('Step Single Test w/o clustering dict')
        print(results_dict['TaskAgnostic']['acc_stepwise_wo_cluster'])
        print('========================================================')

    def save_single(self, path):
        torch.save(self.single_head.state_dict(), path)
        print("Learned Single Head saved to {}.".format(path))

    def save_joint_head(self, path, stage='scan'):
        torch.save(self.joint_head_dict[stage].state_dict(), path)
        print("Joint Head saved to {}.".format(path))

    def calculate_weighted_avg(self, step_acc_list, args):
        acc = 0.
        num_discovered = 0
        for s in range(len(step_acc_list)):
            this_num_novel = args.num_novel_interval if int(1 + s) < args.num_steps else args.num_novel_per_step
            acc += step_acc_list[s] * this_num_novel
            num_discovered += this_num_novel

        acc /= num_discovered
        return acc

    def fill_joint_head(self, args, stage='scan'):
        for step in range(args.current_step):
            w_saved = self.learned_single_heads_dict[stage][step].last_layer.weight.data.clone()
            self.joint_head_dict[stage].last_layer.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w_saved)

        current_w = self.single_head.last_layer.weight.data.clone()
        self.joint_head_dict[stage].last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w)

    def update_learned_heads_list(self, stage='scan'):
        learned_single_head = copy.deepcopy(self.single_head)
        self.learned_single_heads_dict[stage].append(learned_single_head)
        print(
            f"[Stage-{stage}][Single head training completed]: extended the learned single heads list by the newly learned single head")



