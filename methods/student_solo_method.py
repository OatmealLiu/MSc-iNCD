import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter
from utils import ramps
from utils.logging import Logger
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb

from methods.testers import test_cluster, test_ind_cluster

class SoloStudent:
    def __init__(self, model, teachers_list, student, joint_head, sinkhorn, train_loader, ulb_step_val_list,
                 ulb_all_prev_val, ulb_all_val, ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.teachers_list = teachers_list
        self.student = student
        self.joint_head = joint_head

        # Sinkhorn algo.
        self.sinkhorn = sinkhorn

        # Data loaders
        # |- train
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

    def concat_heads(self, args):
        for step in range(args.current_step):
            w_saved = self.teachers_list[step].last_layer.weight.data.clone()
            self.joint_head.last_layer.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w_saved)

        current_w = self.student.last_layer.weight.data.clone()
        self.joint_head.last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w)

    def train_Student(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am Solo student [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.student.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=args.gamma)
        # criterion_ce = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.model.train()
            self.student.train()
            for teacher in self.teachers_list:
                teacher.eval()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                if args.l2_single_cls:
                    with torch.no_grad():
                        # weight.data.shape = # of classes x 768
                        weight_temp = self.student.last_layer.weight.data.clone()
                        weight_temp = F.normalize(weight_temp, dim=1, p=2)
                        self.student.last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                # Student head output
                output_v0_ = self.student(feat_v0)
                output_v1_ = self.student(feat_v1)

                # Sinkhorn swipe-pseudo labeling
                target_v0_ = self.sinkhorn(output_v1_)
                target_v1_ = self.sinkhorn(output_v0_)

                # Zero-padding pseudo labels
                target_v0 = torch.zeros((x_v0.size(0), args.current_novel_end)).to(args.device)
                target_v1 = torch.zeros((x_v1.size(0), args.current_novel_end)).to(args.device)

                target_v0[:, args.current_novel_start:args.current_novel_end] = target_v0_
                target_v1[:, args.current_novel_start:args.current_novel_end] = target_v1_

                # concat output
                output_v0 = []
                output_v1 = []

                # Teacher head outputs
                for teacher in self.teachers_list:
                    this_output_v0 = teacher(feat_v0)
                    this_output_v1 = teacher(feat_v1)
                    output_v0.append(this_output_v0)
                    output_v1.append(this_output_v1)

                output_v0.append(output_v0_)
                output_v1.append(output_v1_)
                output_v0 = torch.cat(output_v0, dim=1).to(args.device)
                output_v1 = torch.cat(output_v1, dim=1).to(args.device)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                # if args.student_loss == 'CE':
                #     # nomral Cross-Entropy loss w/o zero-padding
                #     loss = criterion_ce(mixed_logits, mixed_targets)
                # else:
                loss = self.calculate_ce_zero_padding(mixed_logits, mixed_targets, softmax_temp=args.softmax_temp)
                print(loss)
                loss_record.update(loss.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/student_{args.student_loss}": loss_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_record.avg))
            print('===========================================')

            # save student head
            self.save_student(path=args.save_student_path)
            # save joint head
            self.concat_heads(args)
            self.save_joint_head(None, path=args.save_joint_path)

            print('------>Acc_Student: Single Step Test W/ Clustering')
            # Only test current step
            acc_student_this_step_val_w = test_cluster(self.model, self.student,
                                                       self.ulb_step_val_list[args.current_step],
                                                       args, return_ind=False)


            print('------>Acc_Joint: Single Step Test W/ Clustering')
            acc_joint_this_step_val_w = test_cluster(self.model, self.joint_head,
                                                     self.ulb_step_val_list[args.current_step], args, return_ind=False)

            print('------>Acc_Joint: All-Prev-Steps Test W/ Clustering')
            acc_all_prev_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_val, args)

            print('------> All-Steps Test W/ Clustering')
            acc_all_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_val, args)

            # wandb metrics logging
            wandb.log({
                "val_acc/student_this_step_W_cluster": acc_student_this_step_val_w,
                "val_acc/joint_this_step_W_cluster": acc_joint_this_step_val_w,
                "val_acc/joint_all_prev_W_cluster": acc_all_prev_val_w_cluster,
                "val_acc/joint_all_W_cluster": acc_all_val_w_cluster,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_student_this_step_val_w}")

            print('Joint Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_joint_this_step_val_w}")
            print(f"Acc_all_prev_W_cluster    = {acc_all_prev_val_w_cluster}")
            print(f"Acc_all_W_cluster         = {acc_all_val_w_cluster}")
            print('======================================')

    def test(self, args):
        print('------>Acc_Student: Single Step Test W/ Clustering')
        # Only test current step
        acc_student_this_step_test_w = test_cluster(self.model, self.student,
                                                    self.ulb_step_test_list[args.current_step],
                                                    args, return_ind=False)

        self.concat_heads(args)

        print('------>Acc_Joint: Single Step Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_w = test_cluster(self.model, self.joint_head, self.ulb_step_test_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w


        print('------>Acc_Joint: All-Prev-Steps Test W/ Clustering')
        acc_all_prev_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_test, args)

        print('------> All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_test, args)

        print('\n========================================================')
        print('             Final Test Output (test split)             ')
        print('Single Head Val. Evaluation')
        print("Acc_this_step_W_cluster     = {}".format(acc_student_this_step_test_w))

        print('\nJoint Head Val. Evaluation')
        print("Acc_this_step_W_cluster    = {}".format(acc_step_test_w_cluster_dict[f'Step{args.current_step}_only']))
        print(f"Acc_all_prev_W_cluster     = {acc_all_prev_test_w_cluster}")
        print(f"Acc_all_W_cluster          = {acc_all_test_w_cluster}")

        print('\nStep Single Test w/ clustering dict')
        print(acc_step_test_w_cluster_dict)
        print('========================================================')

    def eval(self):
        pass

    def save_student(self, path):
        torch.save(self.student.state_dict(), path)
        print("Student Head saved to {}.".format(path))

    def save_joint_head(self, args, path):
        if args is not None:
            self.concat_heads(args)
        torch.save(self.joint_head.state_dict(), path)
        print("Joint Head saved to {}.".format(path))

    def return_student(self):
        return self.student

    def return_backbone(self):
        return self.model

