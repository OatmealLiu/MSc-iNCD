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

class UnlockedUpperBound:
    def __init__(self, student_model, student_head, teacher_pair_list, joint_model, joint_head, sinkhorn,
                 train_loader_list, ulb_step_val_list, ulb_all_prev_val, ulb_all_val, ulb_step_test_list,
                 ulb_all_prev_test, ulb_all_test):
        # Models
        #   |- task-specific backbone + head to-be-trained for this NCD step
        self.student_model = student_model
        self.student_head = student_head
        #   |- task-specific (backbone, head)-pairs trained in NCD steps
        self.teacher_pair_list = teacher_pair_list
        #   |- task-agnostic joint backbone + head to-be-jointly-trained for this NCD step
        self.joint_model = joint_model
        self.joint_head = joint_head

        # Sinkhorn pseudo-labeling
        self.sinkhorn = sinkhorn

        # Data loaders
        # |- train: joint training loader
        self.train_loader_list = train_loader_list
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

    def train_single(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am Upper_unlocked-{args.stage} [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.student_model.parameters()) + list(self.student_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.student_model.train()
            self.student_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader_list[-1])):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                # if args.l2_single_cls:
                with torch.no_grad():
                    weight_temp = self.student_head.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.student_head.last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.student_model(x_v0)
                feat_v1 = self.student_model(x_v1)

                # Single head prediction
                output_v0 = self.student_head(feat_v0)
                output_v1 = self.student_head(feat_v1)

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
            # save student head
            self.save_student(model_path=args.save_student_model_path, head_path=args.save_student_head_path)

            print('------>[Single Head] This Step Test W/ Clustering')
            # Only test current step
            this_step_val_w = test_cluster(self.student_model, self.student_head,
                                           self.ulb_step_val_list[args.current_step], args, return_ind=False)
            # wandb metrics logging
            wandb.log({
                "single_val_acc/this_step_ulb_val_w_clustering": this_step_val_w,
            }, step=epoch)

    def train_joint(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am Upper_unlocked-{args.stage} [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.joint_model.parameters()) + list(self.joint_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_record = AverageMeter()  # UNO loss recorder

            # switch joint model and head to train mode
            self.joint_model.train()
            self.joint_head.train()

            # switch single teacher models and heads to eval mode
            for (teacher_model, teacher_head) in self.teacher_pair_list:
                teacher_model.eval()
                teacher_head.eval()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, data in enumerate(tzip(*self.train_loader_list)):
                # mixed raw input
                mixed_x = []
                # mixed pseudo-label generated by teacher model
                mixed_targets = []

                # normalize classifier weights
                # if args.l2_single_cls:
                with torch.no_grad():
                    weight_temp = self.joint_head.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.joint_head.last_layer.weight.copy_(weight_temp)

                for step in range(1+args.current_step):
                    (x_v0, x_v1), _, idx = data[step]
                    # send the vars to GPU
                    x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                    # Feature extraction
                    feat_v0_ = self.teacher_pair_list[step][0](x_v0)
                    feat_v1_ = self.teacher_pair_list[step][0](x_v1)

                    # Single head prediction
                    output_v0_ = self.teacher_pair_list[step][1](feat_v0_)
                    output_v1_ = self.teacher_pair_list[step][1](feat_v1_)

                    # Create pseudo label for novel classes for this step by using the Teacher model
                    target_v0 = output_v0_.detach().max(1)[1] + step * args.num_novel_interval
                    target_v1 = output_v1_.detach().max(1)[1] + step * args.num_novel_interval

                    # accumulate feature and pseudo-label
                    mixed_x.append(x_v0)
                    mixed_x.append(x_v1)
                    mixed_targets.append(target_v0)
                    mixed_targets.append(target_v1)

                mixed_x = torch.cat(mixed_x, dim=0).to(args.device)
                mixed_targets = torch.cat(mixed_targets, dim=0).to(args.device)

                # shuffle all features
                idx_shuffle = torch.randperm(mixed_x.size(0))
                mixed_x, mixed_targets = mixed_x[idx_shuffle], mixed_targets[idx_shuffle]

                # joint backbone feature extraction
                mixed_feats = self.joint_model(mixed_x)

                # joint head prediction
                outputs = self.joint_head(mixed_feats)

                # Cross-Entropy loss w/ zero-padding
                mixed_targets = torch.zeros(mixed_targets.size(0), args.current_novel_end).to(args.device).scatter_(
                                1, mixed_targets.view(-1, 1).long(), 1)
                loss = self.calculate_ce_zero_padding(outputs, mixed_targets, softmax_temp=args.softmax_temp)

                loss_record.update(loss.item(), outputs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss_joint/uno": loss_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_record.avg))
            print('===========================================')

            # print('------> All-Prev-Steps Test W/ Clustering')
            # acc_all_prev_val_w_cluster = test_cluster(self.joint_model, self.joint_head, self.ulb_all_prev_val, args)
            #
            # print('------> All-Steps Test W/ Clustering')
            # acc_all_val_w_cluster = test_cluster(self.joint_model, self.joint_head, self.ulb_all_val, args)
            #
            # # wandb metrics logging
            # wandb.log({
            #     "all_val_acc/all_prev_W_cluster": acc_all_prev_val_w_cluster,
            #     "all_val_acc/all_W_cluster": acc_all_val_w_cluster,
            # }, step=epoch)
            #
            # print('\n======================================')
            # print('             In-epoch Val Single Head Output (val split)             ')
            # print(f"Acc_all_prev_W_cluster    = {acc_all_prev_val_w_cluster}")
            # print(f"Acc_all_W_cluster         = {acc_all_val_w_cluster}")
            # print('======================================')
            self.save_joint(model_path=args.save_joint_model_path, head_path=args.save_joint_head_path)


    def test_single(self, args):
        print('------>[Single Head] Single Step Test W/ Clustering')
        acc_single_w_cluster = test_cluster(self.student_model, self.student_head,
                                            self.ulb_step_test_list[args.current_step], args)

        # print('------>[Single Head] Single Step Test W/O Clustering')
        # acc_single_wo_cluster = test_ind_cluster_unlocked(
        #     test_model=self.student_model, test_head=self.student_head,
        #     ind_gen_model=self.student_model, ind_gen_head=self.student_head,
        #     test_loader=self.ulb_step_test_list[args.current_step], step=args.current_step, args=args,
        #     ind_gen_loader=self.ulb_step_val_list[args.current_step])

        print('\n======================================')
        print('             Final Test Single Head Test Output (Test split)             ')
        print(f"Acc_single_step{args.current_step}_W_cluster    = {acc_single_w_cluster}")
        print('======================================')

    def test_joint(self, args):
        print('------>[Joint Head] Single Step Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_w = test_cluster(self.joint_model, self.joint_head, self.ulb_step_test_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------>[Joint Head] Single Step Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_wo = test_ind_cluster_unlocked(test_model=self.joint_model, test_head=self.joint_head,
                                                          ind_gen_model=self.teacher_pair_list[s][0],
                                                          ind_gen_head=self.teacher_pair_list[s][1],
                                                          test_loader=self.ulb_step_test_list[s], step=s, args=args,
                                                          ind_gen_loader=self.ulb_step_val_list[s])

            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------>[Joint Head] All-Prev-Steps Test W/ Clustering')
        acc_all_prev_test_w_cluster = test_cluster(self.joint_model, self.joint_head, self.ulb_all_prev_test, args)

        print('------>[Joint Head] All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.joint_model, self.joint_head, self.ulb_all_test, args)

        print('------>[Joint Head] All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + args.current_step)]

        acc_all_prev_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        acc_all_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list, args)


        print('\n========================================================')
        print('             Final Test Output (test split) [Joint Head]            ')
        print('All Previous Discovered-Test')
        print(f"Acc_all_prev_W_cluster    = {acc_all_prev_test_w_cluster}")
        print(f"Acc_all_prev_WO_cluster   = {acc_all_prev_test_wo_cluster}")

        print('\nAll Discovered Test')
        print(f"Acc_all_W_cluster         = {acc_all_test_w_cluster}")
        print(f"Acc_all_WO_cluster        = {acc_all_test_wo_cluster}")

        print('\nSingle Step Discovered')
        print('Step Single Test w/ clustering dict')
        print(acc_step_test_w_cluster_dict)

        print('Step Single Test w/o clustering dict')
        print(acc_step_test_wo_cluster_dict)
        print('========================================================')

    def eval(self):
        pass

    def save_student(self, model_path, head_path):
        torch.save(self.student_model.state_dict(), model_path)
        torch.save(self.student_head.state_dict(), head_path)
        print("Task-specific single student backbone saved to {}.".format(model_path))
        print("Task-specific single student head saved to {}.".format(head_path))

    def save_joint(self, model_path, head_path):
        torch.save(self.joint_model.state_dict(), model_path)
        torch.save(self.joint_head.state_dict(), head_path)
        print("Task-agnostic joint backbone saved to {}.".format(model_path))
        print("Task-agnostic joint head saved to {}.".format(head_path))
