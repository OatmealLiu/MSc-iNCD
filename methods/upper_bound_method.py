import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, interleave
from utils import ramps
from utils.logging import Logger
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
import os
import sys
import copy
import wandb

from methods.testers import test_cluster, test_ind_cluster


class UpperBoundTeacherStudent:
    def __init__(self, model, teachers_list, student, train_loader_list, ulb_step_val_list,
                 ulb_all_prev_val, ulb_all_val, ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.teachers_list = teachers_list
        self.student = student

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

    def train_JointTeacherStudent(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am UpperBound_frozen [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.student.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_ce = nn.CrossEntropyLoss()

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

            for batch_idx, data in enumerate(tzip(*self.train_loader_list)):
                # data item format ((x_v0, x_v1), _, idx)
                mixed_feats = []
                mixed_targets = []

                # normalize classifier weights
                # if args.l2_single_cls:
                with torch.no_grad():
                    weight_temp = self.student.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.student.last_layer.weight.copy_(weight_temp)

                for step in range(1+args.current_step):
                    (x_v0, x_v1), _, idx = data[step]
                    # send the vars to GPU
                    x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                    # Feature extraction
                    feat_v0 = self.model(x_v0)
                    feat_v1 = self.model(x_v1)

                    # Single head prediction
                    output_v0 = self.teachers_list[step](feat_v0)
                    output_v1 = self.teachers_list[step](feat_v1)

                    # Create pseudo label for novel classes for this step by using the Teacher model
                    target_v0 = output_v0.detach().max(1)[1] + step * args.num_novel_interval
                    target_v1 = output_v1.detach().max(1)[1] + step * args.num_novel_interval

                    # accumulate feature and pseudo-label
                    mixed_feats.append(feat_v0)
                    mixed_feats.append(feat_v1)
                    mixed_targets.append(target_v0)
                    mixed_targets.append(target_v1)

                mixed_feats = torch.cat(mixed_feats, dim=0).to(args.device)
                mixed_targets = torch.cat(mixed_targets, dim=0).to(args.device)

                # shuffle all features
                idx_shuffle = torch.randperm(mixed_feats.size(0))
                mixed_feats, mixed_targets = mixed_feats[idx_shuffle], mixed_targets[idx_shuffle]

                output_student = self.student(mixed_feats)

                if args.student_loss == 'CE':
                    # nomral Cross-Entropy loss w/o zero-padding
                    loss = criterion_ce(output_student, mixed_targets)
                else:
                    # Cross-Entropy loss w/ zero-padding
                    mixed_targets = torch.zeros(mixed_targets.size(0), args.current_novel_end).to(args.device).scatter_(
                        1, mixed_targets.view(-1, 1).long(), 1)
                    loss = self.calculate_ce_zero_padding(output_student, mixed_targets, softmax_temp=args.softmax_temp)

                loss_record.update(loss.item(), output_student.size(0))

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

            self.save_student(path=args.save_student_path)

            # print('------> Single Step Test W/ Clustering')
            # acc_step_val_w_cluster_dict = dict(
            #     (f"step_val_acc_W_cluster/step{s}_only", -1) for s in range(args.num_steps))

            # Test every single step
            # for s in range(1 + args.current_step):
            #     this_step_val_w = test_cluster(self.model, self.student, self.ulb_step_val_list[s], args,
            #                                    return_ind=False)
            #     acc_step_val_w_cluster_dict[f"step_val_acc_W_cluster/step{s}_only"] = this_step_val_w

            # Only test current step
            # this_step_val_w = test_cluster(self.model, self.student, self.ulb_step_val_list[args.current_step], args,
            #                                return_ind=False)
            # acc_step_val_w_cluster_dict[f"step_val_acc_W_cluster/step{args.current_step}_only"] = this_step_val_w
            #
            # wandb.log(acc_step_val_w_cluster_dict, step=epoch)
            #
            # print('------> Single Step Test W/O Clustering')
            # acc_step_val_wo_cluster_dict = dict(
            #     (f"step_val_acc_WO_cluster/step{s}_only", -1) for s in range(args.num_steps))

            # Test every single step
            # for s in range(1 + args.current_step):
            #     this_step_val_wo = test_ind_cluster(self.model, self.student, self.teachers_list[s],
            #                                         self.ulb_step_val_list[s], s, args, ind_gen_loader=None)
            #     acc_step_val_wo_cluster_dict[f"step_val_acc_WO_cluster/step{s}_only"] = this_step_val_wo

            # Only test current step
            # this_step_val_wo = test_ind_cluster(self.model, self.student, self.teachers_list[args.current_step],
            #                                     self.ulb_step_val_list[args.current_step], args.current_step,
            #                                     args, ind_gen_loader=None)
            # acc_step_val_wo_cluster_dict[f"step_val_acc_WO_cluster/step{args.current_step}_only"] = this_step_val_wo

            # wandb.log(acc_step_val_wo_cluster_dict, step=epoch)

            # print('------> All-Prev-Steps Test W/ Clustering')
            # acc_all_prev_val_w_cluster = test_cluster(self.model, self.student, self.ulb_all_prev_val, args)

            # print('------> All-Steps Test W/ Clustering')
            # acc_all_val_w_cluster = test_cluster(self.model, self.student, self.ulb_all_val, args)

            # print('------> All (all/prev) Steps Test W/O Clustering')
            # step_acc_wo_cluster_list = [acc_step_val_wo_cluster_dict[f"step_val_acc_WO_cluster/step{s}_only"]
            #                             for s in range(1+args.current_step)]
            # acc_all_prev_val_wo_cluster = self.calculate_weighted_avg(step_acc_wo_cluster_list[:-1], args)
            # acc_all_val_wo_cluster = self.calculate_weighted_avg(step_acc_wo_cluster_list, args)

            # wandb metrics logging
            # wandb.log({
            #     "all_val_acc/all_prev_W_cluster": acc_all_prev_val_w_cluster,
                # "all_val_acc/all_prev_WO_cluster": acc_all_prev_val_wo_cluster,
                # "all_val_acc/all_W_cluster": acc_all_val_w_cluster,
                # "all_val_acc/all_WO_cluster": acc_all_val_wo_cluster,
            # }, step=epoch)

            # print('\n======================================')
            # print('All-Previous-Discovered')
            # print(f"Acc_all_prev_W_cluster    = {acc_all_prev_val_w_cluster}")
            # print(f"Acc_all_prev_WO_cluster   = {acc_all_prev_val_wo_cluster}")

            # print('\nAll-Discovered')
            # print(f"Acc_all_W_cluster         = {acc_all_val_w_cluster}")
            # print(f"Acc_all_WO_cluster        = {acc_all_val_wo_cluster}")

            # print('\nSingle-Discovered')
            # print('Step Single Val w/ clustering dict')
            # print(acc_step_val_w_cluster_dict)

            # print('Step Single Val w/o clustering dict')
            # print(acc_step_val_wo_cluster_dict)
            # print('======================================')

    def test(self, args):
        print('------> Single Step Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_w = test_cluster(self.model, self.student, self.ulb_step_test_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------> Single Step Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_wo = test_ind_cluster(self.model, self.student, self.teachers_list[s],
                                                 self.ulb_step_test_list[s], s, args,
                                                 ind_gen_loader=self.ulb_step_val_list[s])
            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------> All-Prev-Steps Test W/ Clustering')
        acc_all_prev_test_w_cluster = test_cluster(self.model, self.student, self.ulb_all_prev_test, args)

        print('------> All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.model, self.student, self.ulb_all_test, args)

        print('------> All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + args.current_step)]
        acc_all_prev_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        acc_all_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list, args)


        print('\n========================================================')
        print('             Final Test Output (test split)             ')
        print('All-Previous-Discovered-Test')
        print(f"Acc_all_prev_W_cluster    = {acc_all_prev_test_w_cluster}")
        print(f"Acc_all_prev_WO_cluster   = {acc_all_prev_test_wo_cluster}")

        print('\nAll-Discovered-Test')
        print(f"Acc_all_W_cluster         = {acc_all_test_w_cluster}")
        print(f"Acc_all_WO_cluster        = {acc_all_test_wo_cluster}")

        print('\nSingle-Discovered')
        print('Step Single Test w/ clustering dict')
        print(acc_step_test_w_cluster_dict)

        print('Step Single Test w/o clustering dict')
        print(acc_step_test_wo_cluster_dict)
        print('========================================================')

    def eval(self):
        pass

    def save_student(self, path):
        torch.save(self.student.state_dict(), path)
        print("Student Head saved to {}.".format(path))

    def return_student(self):
        return self.student

