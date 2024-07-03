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

class WeightDiscrepancy:
    def __init__(self, model, single_head, learned_single_heads, joint_head, sinkhorn,
                 train_loader, uncert_loader, ulb_step_val_list, ulb_all_prev_val, ulb_all_val,
                 ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.single_head = single_head                  # to-be-trained
        self.learned_single_heads = learned_single_heads
        self.joint_head = joint_head

        # Sinkhorn algo.
        self.sinkhorn = sinkhorn

        # Data loaders
        # |- train
        self.train_loader = train_loader
        # |- uncert
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
            w_saved = self.learned_single_heads[step].last_layer.weight.data.clone()
            self.joint_head.last_layer.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w_saved)

        current_w = self.single_head.last_layer.weight.data.clone()
        self.joint_head.last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w)

    def generate_affinity_map(self, args):
        self.model.eval()
        self.single_head.eval()

        for single in self.learned_single_heads:
            single.eval()

        aff_list = [[] for s in range(1+args.current_step)]
        print("---> Calc.: affinity of current training data w.r.t previous learned prototypes")
        with torch.no_grad():
            for batch_idx, (x, label, _) in enumerate(tqdm(self.uncert_loader)):
                x, label = x.to(args.device), label.to(args.device)

                feat = self.model(x)

                # previous heads
                for step in range(args.current_step):
                    output = self.learned_single_heads[step](feat)
                    aff_list[step].append(output)

                # student head
                output = self.single_head(feat)
                aff_list[args.current_step].append(output)

        aff_list = [torch.cat(aff, dim=0) for aff in aff_list]              # y^_1, ..., y^_t
        aff_list = [aff.mean(dim=0) for aff in aff_list]                    # y_avg^_1, ..., y_avg^_t
        print(aff_list)
        aff_list = [aff+torch.ones_like(aff) for aff in aff_list]           # y_avg^_1 + 1, ..., y_avg^_t + 1
        print(aff_list)
        aff_list = [aff.reshape((1, aff.size(0))) for aff in aff_list]
        aff_list = [aff.repeat((aff.size(1), 1)) for aff in aff_list]       # tile the affinity vector to a heatmap
                                                                            # in order to weight the weight discrepancy
                                                                            # matrix b.t.w. current learning prototypes
                                                                            # prev-learned prototypes
        print(aff_list)
        # for aff in aff_list[:-1]:
        #     aff.requires_grad = False
        # aff_list[:-1] contains the affinity heatmap of the current data w.r.t. prev-learned prototypes
        # aff_list[-1] contains the affinity heatmap of the current data w.r.t. current-learning prototypes
        return aff_list

    def train_init(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tInit-Stage Weight Discrepancy Method: [{1 + args.current_step}/{args.num_steps}] >_<")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.single_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        # Running best acc. eval. on val. dataset
        best_acc = 0.0
        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_recorder = AverageMeter()          # UNO loss recording

            # switch the models to train mode
            self.model.train()
            self.single_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                self.single_head.normalize_prototypes()

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

                # UNO Loss
                loss_uno = self.calculate_ce_zero_padding(mixed_logits, mixed_targets, softmax_temp=args.softmax_temp)
                loss_uno_recorder.update(loss_uno.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss_uno.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/single_head": loss_uno_recorder.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg UNO Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_uno_recorder.avg))
            print('===========================================')

            # save student head
            self.save_single(path=args.save_single_path)
            # save joint head
            self.concat_heads(args)
            self.save_joint_head(None, path=args.save_joint_path)

            print('------>[Single Head Val.]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.model, self.single_head,
                                                      self.ulb_step_val_list[args.current_step],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                "val_acc/single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

            if acc_single_this_step_val_w > best_acc:
                best_acc = max(acc_single_this_step_val_w, best_acc)
                self.save_single(path=args.save_single_path[:-4]+'_best.pth')
                self.save_joint_head(None, path=args.save_joint_path[:-4]+'_best.pth')

        self.learned_single_heads.append(self.single_head)
        print(
            "[Single head training completed]: extended the learned single heads list by the newly learned single head")

    def train_il(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tIncremental-Stage Weight Discrepancy Method: [{1 + args.current_step}/{args.num_steps}] >_<")
        print("=" * 100)

        # conf-heatmap-guided knob
        if args.conf_guided:
            # generate affinity heatmap b.t.w. current mini-batch of data and prev. prototype centroids
            # aff_list:
            #   |- aff_list[i].shape = (num_step_interval X num_step_interval)
            #   |- aff_list[:-1] contains the affinity heatmap of the current data w.r.t. prev-learned prototypes
            #   |- aff_list[-1] contains the affinity heatmap of the current data w.r.t. current-learning prototypes
            aff_list = self.generate_affinity_map(args)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.single_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        # best_acc = 0.0
        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_recorder = AverageMeter()          # UNO loss recording
            loss_weight_recorder = AverageMeter()       # Weight Discrepancy loss recording

            # switch the models to train mode
            self.model.train()
            self.single_head.train()
            for single in self.learned_single_heads:
                single.eval()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            # W_wd for weight discrepancy loss
            if args.w_wd == 'damp':
                if args.epoch_wd == 0:
                    w_wd = (1 - epoch/(args.epochs - 1)) * 0.1
                else:
                    if epoch >= args.epoch_wd:
                        w_wd = (1 - (epoch - args.epoch_wd)/(args.epoch_wd - 1)) * 0.1
                    else:
                        w_wd = 1.0
            elif args.w_wd == 'rampup':
                w_wd = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) * 0.004
            else:
                w_wd = float(args.w_wd)

            # update ramp-up coefficient for the current epoch
            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                self.single_head.normalize_prototypes()

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

                # Loss_uno
                loss_uno = self.calculate_ce_zero_padding(mixed_logits, mixed_targets, softmax_temp=args.softmax_temp)

                # Loss_weight_discrepancy
                if epoch >= args.epoch_wd:
                    # Weights of previous trained classifier heads
                    W_prev_list = [next(single.last_layer.parameters()) for single in self.learned_single_heads]

                    # Weight of classifier head of the current student model
                    W_t = next(self.single_head.last_layer.parameters())

                    # Cosine similarity matrix
                    W_cossim = [W_t.matmul(W_prev_s.T) for W_prev_s in W_prev_list]
                    # +1 to lift the CosSim from [-1,1] to [0,1]
                    W_cossim = [W_ts_dissim + torch.ones_like(W_ts_dissim).float().to(args.device)
                                for W_ts_dissim in W_cossim]

                    W_cossim = torch.cat(W_cossim, dim=1)
                    _, idx_topsim = torch.topk(W_cossim, k=args.w_topk, dim=1)
                    W_cossim_k = [W_cossim[i, idx_topsim[i]] for i in range(W_cossim.size(0))]
                    W_cossim_k = torch.cat(W_cossim_k)
                    loss_weight = torch.sum(W_cossim_k)
                    loss_weight = (1.0/args.w_topk) * w_wd * loss_weight
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
                f"loss/L_bce": loss_uno_recorder.avg,
                f"loss/L_wd": loss_weight_recorder.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg UNO Loss: {:.4f}\tWeight Loss: {:.4f}'.format(1 + epoch,
                                                                                            args.epochs,
                                                                                            loss_uno_recorder.avg,
                                                                                            loss_weight_recorder.avg))
            print('===========================================')

            # save student head
            self.save_single(path=args.save_single_path)
            # save joint head
            self.concat_heads(args)
            self.save_joint_head(None, path=args.save_joint_path)

            print('------>[Single Head Val.]: Single Step Test W/ Clustering')
            # Only test current step
            acc_student_this_step_val_w = test_cluster(self.model, self.single_head,
                                                       self.ulb_step_val_list[args.current_step],
                                                       args, return_ind=False)


            print('------>[Joint Head Val.]: Single Step Test W/ Clustering')
            acc_joint_this_step_val_w = test_cluster(self.model, self.joint_head,
                                                     self.ulb_step_val_list[args.current_step], args, return_ind=False)

            print('------>[Joint Head Val.]: All-Prev-Steps Test W/ Clustering')
            acc_all_prev_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_val, args)

            print('------>[Joint Head Val.]: All-Steps Test W/ Clustering')
            acc_all_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_val, args)

            # wandb metrics logging
            wandb.log({
                "val_acc/student_this_step_W_cluster": acc_student_this_step_val_w,
                "val_acc/joint_this_step_W_cluster": acc_joint_this_step_val_w,
                "val_acc/joint_all_prev_W_cluster": acc_all_prev_val_w_cluster,
                "val_acc/joint_all_W_cluster": acc_all_val_w_cluster,
            }, step=epoch)

            print('\n======================================')
            print('[Single Head Val.] for this epoch')
            print(f"Acc_this_step_W_cluster    = {acc_student_this_step_val_w}")

            print('\n[Joint Head Val.] for this epoch')
            print(f"Acc_this_step_W_cluster    = {acc_joint_this_step_val_w}")
            print(f"Acc_all_prev_W_cluster    = {acc_all_prev_val_w_cluster}")
            print(f"Acc_all_W_cluster         = {acc_all_val_w_cluster}")

            print('======================================')
            # if acc_all_val_w_cluster > best_acc:
            #     best_acc = max(acc_all_val_w_cluster, best_acc)
            #     self.save_single(path=args.save_single_path[:-4]+'_best.pth')
            #     self.save_joint_head(None, path=args.save_joint_path[:-4]+'_best.pth')

        self.learned_single_heads.append(self.single_head)
        print(
            "[Single head training completed]: extended the learned single heads list by the newly learned single head")

    def test(self, args):
        # if best:
        #     self.load_single(args, path=args.save_single_path[:-4]+'_best.pth')
        #     self.learned_single_heads[-1] = self.single_head
        #     self.load_joint_head(args, path=args.save_joint_path[:-4]+'_best.pth')
        #     print('\n========================================================')
        #     print('             We are the best             ')
        #     print('\n========================================================')
        # else:
        self.concat_heads(args)

        # === Single Head ===
        print('------>[Single Head Test.]: Single Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.model, self.single_head,
                                                           self.ulb_step_test_list[args.current_step],
                                                           args, return_ind=False)
        # === Joint Head ===
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
            this_step_test_wo = test_ind_cluster(self.model, self.joint_head, self.learned_single_heads[s],
                                                 self.ulb_step_test_list[s], s, args,
                                                 ind_gen_loader=self.ulb_step_val_list[s])
            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------> All-Prev-Steps Test W/ Clustering')
        if args.current_step > 0:
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

        print('\nStepwise-Discovered')
        print('Step Single Test w/ clustering dict')
        print(acc_step_test_w_cluster_dict)

        print('Step Single Test w/o clustering dict')
        print(acc_step_test_wo_cluster_dict)
        print('========================================================')

    def eval(self):
        pass

    def save_single(self, path):
        torch.save(self.single_head.state_dict(), path)
        print("Learned Single Head saved to {}.".format(path))

    def save_joint_head(self, args, path):
        if args is not None:
            self.concat_heads(args)
        torch.save(self.joint_head.state_dict(), path)
        print("Joint Head saved to {}.".format(path))

    def load_single(self, args, path):
        best_single_head_state_dict = torch.load(path, map_location=args.device)
        self.single_head.load_state_dict(best_single_head_state_dict)
        self.single_head.to(args.device)
        print(f"Loaded best single head weights from {path}")

    def load_joint_head(self, args, path):
        best_joint_head_state_dict = torch.load(path, map_location=args.device)
        self.joint_head.load_state_dict(best_joint_head_state_dict)
        self.joint_head.to(args.device)
        print(f"Loaded best joint head weights from {path}")

    def return_single(self):
        return self.single_head

    def return_backbone(self):
        return self.model

