import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, AverageMeter, seed_torch, PairEnum
from utils import ramps
from utils.logging import Logger
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb
from sklearn import metrics

from methods.testers import test_cluster, test_ind_cluster

def entropy(x):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ = torch.clamp(x, min=EPS)
    b = x_ * torch.log(x_)

    if len(b.size()) == 2:  # Sample-wise entropy
        return - b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class ORCA:
    def __init__(self, model, teachers_list, student, joint_head, train_loader, ulb_step_val_list,
                 ulb_all_prev_val, ulb_all_val, ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.teachers_list = teachers_list
        self.student = student
        self.joint_head = joint_head

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

    # def calculate_ce_zero_padding(self, output, target, softmax_temp=0.1):
    #     # follow original UNO, temperature = 0.1
    #     preds = F.softmax(output / softmax_temp, dim=1)  # temperature
    #     preds = torch.clamp(preds, min=1e-8)
    #     preds = torch.log(preds)
    #     loss = -torch.mean(torch.sum(target * preds, dim=1))
    #     return loss

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
        if args.use_norm:
            for step in range(args.current_step):
                w_saved = self.teachers_list[step].last_layer.weight.data.clone()
                self.joint_head.last_layer.weight.data[:, step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w_saved)

            current_w = self.student.last_layer.weight.data.clone()
            self.joint_head.last_layer.weight.data[:, args.current_novel_start:args.current_novel_end].copy_(current_w)
        else:
            for step in range(args.current_step):
                w_saved = self.teachers_list[step].last_layer.weight.data.clone()
                self.joint_head.last_layer.weight.data[step * args.num_novel_interval:(1 + step) * args.num_novel_interval].copy_(w_saved)

            current_w = self.student.last_layer.weight.data.clone()
            self.joint_head.last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w)

    def train_single(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am OCRA Teacher [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.student.parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        if args.rs_bce:
            criterion_bce = BCE()
        else:
            criterion_bce = nn.BCELoss()

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_bce_recorder = AverageMeter()
            loss_entropy_recorder = AverageMeter()

            # switch the models to train mode
            self.model.train()
            self.student.train()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)
                optimizer.zero_grad()

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                # Single head prediction
                output_v0 = self.student(feat_v0)
                output_v1 = self.student(feat_v1)

                # Softmax
                prob_v0 = F.softmax(output_v0, dim=1)
                prob_v1 = F.softmax(output_v1, dim=1)

                # Pairwise Objective
                feat_detach_v0 = feat_v0.detach()
                feat_normalized_v0 = feat_detach_v0 / torch.norm(feat_detach_v0, 2, 1, keepdim=True)
                cosine_dist_v0 = torch.mm(feat_normalized_v0, feat_normalized_v0.t())

                # # unlabel part
                vals, pos_idx = torch.topk(cosine_dist_v0, args.topk, dim=1)            # original k=2;
                pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()

                pos_prob = prob_v1[pos_idx, :]
                pos_sim = torch.bmm(prob_v0.view(prob_v0.size(0), 1, -1), pos_prob.view(prob_v0.size(0), -1, 1)).squeeze()
                ones = torch.ones_like(pos_sim).float().to(args.device)

                # RS Loss
                if args.rs_bce:
                    bce_loss = args.w_bce * criterion_bce(prob_v0.clone(), pos_prob.clone(), ones)
                else:
                    bce_loss = args.w_bce * criterion_bce(pos_sim, ones)

                if args.w_entropy > 0:
                    entropy_loss = args.w_entropy * entropy(torch.mean(prob_v0, dim=0))
                else:
                    entropy_loss = torch.tensor(0.0)

                loss = -entropy_loss + bce_loss

                # print('\n===========================================')
                # print('\nAvg Loss: bce={:.4f}, entropy={:.4f}'.format(bce_loss, entropy_loss))
                # print('===========================================')

                loss_bce_recorder.update(bce_loss.item(), x_v0.size(0))
                # ce_losses.update(ce_loss.item(), args.batch_size)
                loss_entropy_recorder.update(entropy_loss.item(), x_v0.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            # wandb loss logging
            wandb.log({
                "loss/bce": loss_bce_recorder.avg,
                "loss/entropy": loss_entropy_recorder.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: bce={:.4f}, entropy={:.4f}'.format(1 + epoch, args.epochs,
                                                                                       loss_bce_recorder.avg,
                                                                                       loss_entropy_recorder.avg))
            print('===========================================')
            # save student head
            self.save_student(path=args.save_student_path)

            print('------>[Single Head] This Step Test W/ Clustering')
            # Only test current step
            this_step_val_w = test_cluster(self.model, self.student,
                                           self.ulb_step_val_list[args.current_step], args, return_ind=False)
            # wandb metrics logging
            wandb.log({
                "val_acc/this_step_ulb_val_w_clustering": this_step_val_w,
            }, step=epoch)

        # Finish the training of the single head for this step, add the newest trained student head to teacher head
        self.teachers_list.append(self.student)

    def test(self, args):
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.model, self.student,
                                                           self.ulb_step_test_list[args.current_step], args,
                                                           return_ind=False)

        self.concat_heads(args)

        print('------>[Joint Head] Individual Steps Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_w = test_cluster(self.model, self.joint_head, self.ulb_step_test_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------>[Joint Head] Individual Steps Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + args.current_step):
            this_step_test_wo = test_ind_cluster(self.model, self.joint_head, self.teachers_list[s],
                                                 self.ulb_step_test_list[s], s, args,
                                                 ind_gen_loader=self.ulb_step_val_list[s])
            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------> All-Prev-Steps Test W/ Clustering')
        if self.ulb_all_prev_test is not None:
            acc_all_prev_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_test, args)
        else:
            acc_all_prev_test_w_cluster = -1

        print('------> All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_test, args)

        print('------> All (all/prev) Steps Test W/O Clustering')
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

    def save_joint_head(self, args, path):
        if args is not None:
            self.concat_heads(args)
        torch.save(self.joint_head.state_dict(), path)
        print("Joint Head saved to {}.".format(path))

    def return_new_teacher(self):
        return self.teachers_list[-1]

    def return_joint_head(self):
        return self.joint_head

    def return_backbone(self):
        return self.model

