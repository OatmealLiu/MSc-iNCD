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

class UncertWeightDiscrepancy:
    def __init__(self, model, teachers_list, student, joint_head, sinkhorn, train_loader, uncert_loader,
                 ulb_step_val_list, ulb_all_prev_val, ulb_all_val, ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
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
        self.uncert_loader = uncert_loader
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
        print(f"\t\t\t\t\tCiao bella! I am Uncertainty [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD >_<")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.student.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        best_acc = 0.0
        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_recorder = AverageMeter()          # UNO loss recording
            loss_weight_recorder = AverageMeter()       # Weight Discrepancy loss recording

            # TODO: first time calc.
            if args.current_step > 0:
                step_conf_list, step_uncert_list = self.estimate_uncertainty(args)

            # switch the models to train mode
            self.model.train()
            self.student.train()
            for teacher in self.teachers_list:
                teacher.eval()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            # damping parameter for loss_weight
            if args.damping:
                damping = (1 - epoch/(args.epochs-1))
            else:
                damping = 1.0

            print("---> Batch iteration")
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

                # Single head prediction
                output_v0 = self.student(feat_v0)
                output_v1 = self.student(feat_v1)

                # cross pseudo-labeling
                target_v0 = self.sinkhorn(output_v1)
                target_v1 = self.sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                # UNO Loss
                loss_uno = self.calculate_ce_zero_padding(mixed_logits, mixed_targets, softmax_temp=args.softmax_temp)

                # Weight Discrepancy Loss
                if args.current_step > 0:
                    # Weight of classifier head of the current student model
                    W_s = next(self.student.last_layer.parameters()).view(-1)

                    # Weights of previous trained classifier heads
                    W_t_list = [next(teacher.last_layer.parameters()).view(-1) for teacher in self.teachers_list]

                    # Loss calculation
                    # +1 for a positive loss
                    loss_weight = step_conf_list[0] * (torch.matmul(W_s, W_t_list[0]) / (torch.norm(W_s) * torch.norm(W_t_list[0])) + 1)
                    for s in range(len(W_t_list[1:])):
                        W_t = W_t_list[1+s]
                        conf_t = W_t_list[1+s]
                        loss_weight += conf_t * (torch.matmul(W_s, W_t) / (torch.norm(W_s) * torch.norm(W_t)) + 1)

                    # current uncertainty-guided
                    loss_weight = 2 * args.w_weight * damping * loss_weight
                    if args.current_uncert:
                        loss_weight *= step_uncert_list[-1]
                else:
                    loss_weight = torch.tensor(0.0)

                # Total Loss
                loss = loss_uno + loss_weight

                loss_uno_recorder.update(loss_uno.item(), output_v0.size(0))
                loss_weight_recorder.update(loss_weight.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/student_{args.student_loss}": loss_uno_recorder.avg,
                f"loss/weight_discrepancy": loss_weight_recorder.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg UNO Loss: {:.4f}\tWeight Loss: {:.4f}'.format(1 + epoch,
                                                                                            args.epochs,
                                                                                            loss_uno_recorder.avg,
                                                                                            loss_weight_recorder.avg))
            print('===========================================')

            # save student head
            self.save_student(path=args.save_student_path)
            # save joint head
            self.concat_heads(args)
            self.save_joint_head(None, path=args.save_joint_path)

            print('------>[Single Head Val.]: Single Step Test W/ Clustering')
            # Only test current step
            acc_student_this_step_val_w = test_cluster(self.model, self.student,
                                                       self.ulb_step_val_list[args.current_step],
                                                       args, return_ind=False)

            print('------>[Joint Head Val.]: Single Step Test W/ Clustering')
            acc_joint_this_step_val_w = test_cluster(self.model, self.joint_head,
                                                     self.ulb_step_val_list[args.current_step], args, return_ind=False)

            # if args.current_step > 0:
            #     print('------>[Joint Head Val.]: All-Prev-Steps Test W/ Clustering')
            #     acc_all_prev_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_val, args)
            # else:
            #     acc_all_prev_val_w_cluster = -1

            print('------>[Joint Head Val.]: All-Steps Test W/ Clustering')
            acc_all_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_val, args)

            # wandb metrics logging
            wandb.log({
                "val_acc/student_this_step_W_cluster": acc_student_this_step_val_w,
                "val_acc/joint_this_step_W_cluster": acc_joint_this_step_val_w,
                # "val_acc/joint_all_prev_W_cluster": acc_all_prev_val_w_cluster,
                "val_acc/joint_all_W_cluster": acc_all_val_w_cluster,
            }, step=epoch)

            print('\n======================================')
            print('[Single Head Val.] for this epoch')
            print(f"Acc_this_step_W_cluster    = {acc_student_this_step_val_w}")

            print('\n[Joint Head Val.] for this epoch')
            print(f"Acc_this_step_W_cluster    = {acc_joint_this_step_val_w}")
            # print(f"Acc_all_prev_W_cluster    = {acc_all_prev_val_w_cluster}")
            print(f"Acc_all_W_cluster         = {acc_all_val_w_cluster}")
            print('======================================')

            if acc_all_val_w_cluster > best_acc:
                best_acc = max(acc_all_val_w_cluster, best_acc)
                self.save_student(path=args.save_student_path[:-4]+'_best.pth')
                self.save_joint_head(None, path=args.save_joint_path[:-4]+'_best.pth')

    def test(self, args, best=False):
        if best:
            self.load_student(args, path=args.save_student_path[:-4]+'_best.pth')
            self.load_joint_head(args, path=args.save_joint_path[:-4]+'_best.pth')
            print('\n========================================================')
            print('             We are the best             ')

        print('------>[Single Head Test.]: Single Step Test W/ Clustering')
        # Only test current step
        acc_student_this_step_test_w = test_cluster(self.model, self.student,
                                                    self.ulb_step_test_list[args.current_step],
                                                    args, return_ind=False)

        self.concat_heads(args)

        print('------>[Joint Head Test.]: Single Step Test W/ Clustering')
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

        print('------>[Joint Head Test.]: All-Prev-Steps Test W/ Clustering')
        if args.current_step > 0:
            acc_all_prev_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_test, args)
        else:
            acc_all_prev_test_w_cluster = -1

        print('------>[Joint Head Test.]: All Test W/ Clustering')
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
        print(f"Acc_this_step             = {acc_student_this_step_test_w}")

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

    def estimate_uncertainty(self, args):
        self.model.eval()
        for teacher in self.teachers_list:
            teacher.eval()
        self.student.eval()

        step_confs_list = [np.array([]) for s in range(1+args.current_step)]
        print("---> Calc. Uncertainty and Confidence")
        with torch.no_grad():
            for batch_idx, (x, label, _) in enumerate(tqdm(self.uncert_loader)):
                x, label = x.to(args.device), label.to(args.device)

                feat = self.model(x)

                # previous heads
                for step in range(args.current_step):
                    output = self.teachers_list[step](feat)
                    conf, _ = output.max(1)
                    step_confs_list[step] = np.append(step_confs_list[step], conf.cpu().numpy())

                # student head
                output = self.student(feat)
                conf, _ = output.max(1)
                step_confs_list[args.current_step] = np.append(step_confs_list[args.current_step], conf.cpu().numpy())

        avg_conf = [np.mean(s_confs) for s_confs in step_confs_list]
        avg_uncert = [1 - mean_conf for mean_conf in avg_conf]

        # [:-1]: prev conf and uncert, [-1]: current conf and uncert
        print("Confidence list")
        print(avg_conf)
        print("Uncertainty list")
        print(avg_uncert)
        return avg_conf, avg_uncert

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

    def load_student(self, args, path):
        best_student_head_state_dict = torch.load(path, map_location=args.device)
        self.student.load_state_dict(best_student_head_state_dict)
        self.student.to(args.device)
        print(f"Loaded best student head weights from {path}")

    def load_joint_head(self, args, path):
        best_joint_head_state_dict = torch.load(path, map_location=args.device)
        self.joint_head.load_state_dict(best_joint_head_state_dict)
        self.joint_head.to(args.device)
        print(f"Loaded best joint head weights from {path}")

    def return_student(self):
        return self.student

    def return_backbone(self):
        return self.model

