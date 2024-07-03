import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler

from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, interleave
from utils import ramps
from utils.logging import Logger
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb

from methods.testers import test_cluster, test_ind_cluster, test_ind_cluster_unlocked
from utils.LwF_losses import LwFLoss, ZeroPaddingCrossEntropy
from utils.sinkhorn_knopp import SinkhornKnopp

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


class FRoST:
    def __init__(self, model, single_head, joint_head, prev_pair_list, feat_replayer,
                 train_loader, ulb_step_val_list, ulb_all_prev_val, ulb_all_val, ulb_step_test_list,
                 ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.single_head = single_head
        self.joint_head = joint_head
        # Prev: model-head pair list
        self.prev_pair_list = prev_pair_list

        # Feature Replayers
        self.feat_replayer = feat_replayer

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

    def train_init(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am FRoST-Init [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)
        sinkhorn = SinkhornKnopp(args)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.single_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_record = AverageMeter()                # Total loss recorder
            loss_uno_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.model.train()
            self.single_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()
            w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):

                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # weight.data.shape = # of classes x 768
                    weight_temp = self.single_head.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.single_head.last_layer.weight.copy_(weight_temp)
                # self.single_head.normalize_prototypes()

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                # Single head output
                output_v0 = self.single_head(feat_v0)
                output_v1 = self.single_head(feat_v1)

                prob_v0, prob_v1 = F.softmax(output_v0, dim=1), F.softmax(output_v1, dim=1)

                # Sinkhorn swipe-pseudo labeling
                target_v0 = sinkhorn(output_v1)
                target_v1 = sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), output_v0.size(0))

                consistency_loss = F.mse_loss(prob_v0, prob_v1)

                loss = loss_uno + w * consistency_loss
                loss_record.update(loss.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                "loss/bce": loss_uno_record.avg,
                       }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_record.avg))
            print(f"Loss_bce         = {loss_uno_record.avg}")
            print('===========================================')
            self.save_single(model_path=args.save_single_model_path, head_path=args.save_single_head_path)

            # === Single Head Val ===
            print('------>[Single Head]: Single Step Test W/ Clustering')
            acc_single_this_step_val_w = test_cluster(self.model, self.single_head,
                                                      self.ulb_step_val_list[args.current_step],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                "val/acc_single_this_step": acc_single_this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

    def train_IL(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am FRoST-IL [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)
        sinkhorn = SinkhornKnopp(args)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.single_head.parameters())\
                     + list(self.joint_head.parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = BCE()
        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_record = AverageMeter()                # Total loss recorder
            loss_ce_add_record = AverageMeter()         # CE loss recorder
            loss_uno_record = AverageMeter()            # BCE loss recorder
            consistency_loss_record = AverageMeter()    # MSE consistency loss recorder
            loss_kd_record = AverageMeter()             # LwF loss recorder (logits-KD, w/o softmax)
            loss_replay_record = AverageMeter()         # Feature-replay loss recorder

            # switch the models to train mode
            self.model.train()
            self.joint_head.train()
            self.single_head.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            # update ramp-up coefficient for the current epoch
            w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # single head
                    weight_temp = self.single_head.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.single_head.last_layer.weight.copy_(weight_temp)
                    # joint head
                    weight_temp = self.joint_head.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.joint_head.last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                j_output_v0 = self.joint_head(feat_v0)
                j_output_v1 = self.joint_head(feat_v1)

                # Single head output
                output_v0 = self.single_head(feat_v0)
                output_v1 = self.single_head(feat_v1)

                prob_v0, prob_v1 = F.softmax(output_v0, dim=1), F.softmax(output_v1, dim=1)
                label = (output_v0).detach().max(1)[1] + args.current_novel_start

                # Sinkhorn swipe-pseudo labeling
                target_v0 = sinkhorn(output_v1)
                target_v1 = sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)


                # L_ce: pseudo-label for joint head
                loss_ce_add = w * criterion1(j_output_v0, label)/args.rampup_coefficient * args.increment_coefficient
                # L_bce: ranking-statistics for single head
                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                # L_mse: consistency loss for single head
                consistency_loss = F.mse_loss(prob_v0, prob_v1)

                # L_replay: constraint classifier
                if self.feat_replayer is not None:
                    # print("----------> calculate feat replay loss")
                    replayed_feat, replayed_label = self.feat_replayer.replay_all()
                    output1_replayed = self.joint_head(replayed_feat)
                    loss_ce_replay = args.w_replay * criterion1(output1_replayed, replayed_label)
                else:
                    loss_ce_replay = torch.tensor(0.0)

                # L_feat_KD: constraint backbone
                if args.w_kd > 0:
                    prev_feat = self.prev_pair_list[-1][0](x_v0)
                    size1, size2 = prev_feat.size()
                    loss_kd = torch.dist(F.normalize(prev_feat.contiguous().view(size1 * size2, 1), dim=0),
                                         F.normalize(feat_v0.contiguous().view(size1 * size2, 1), dim=0)) * args.w_kd
                else:
                    loss_kd = torch.tensor(0.0)

                loss = loss_uno + loss_ce_add + w * consistency_loss + loss_kd + loss_ce_replay

                loss_uno_record.update(loss_uno.item(), x_v0.size(0))
                loss_ce_add_record.update(loss_ce_add.item(), x_v0.size(0))
                consistency_loss_record.update(consistency_loss.item(), x_v0.size(0))
                loss_replay_record.update(loss_ce_replay.item(), x_v0.size(0))
                loss_kd_record.update(loss_kd.item(), x_v0.size(0))
                loss_record.update(loss.item(), x_v0.size(0))


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({"loss/pseudo-ulb": loss_ce_add_record.avg,
                       "loss/uno": loss_uno_record.avg,
                       "loss/consistency": consistency_loss_record.avg,
                       "loss/feat_kd": loss_kd_record.avg,
                       "loss/replay": loss_replay_record.avg,
                       "loss/total_loss": loss_record.avg,
                       }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_record.avg))
            print(f"Loss_uno         = {loss_uno_record.avg}")
            print(f"Loss_consistency = {consistency_loss_record.avg}")
            print(f"Loss_pseudo-ulb  = {loss_ce_add_record.avg}")
            print(f"Loss_LwF         = {loss_kd_record.avg}")
            print(f"Loss_replay      = {loss_replay_record.avg}")
            print(f"Loss_total       = {loss_record.avg}")
            print('===========================================')

            self.save_single(model_path=args.save_single_model_path, head_path=args.save_single_head_path)
            self.save_joint(model_path=args.save_joint_model_path, head_path=args.save_joint_head_path)

            # === Single Head Val ===
            print('------>[Single Head]: Single Step Test W/ Clustering')
            acc_single_this_step_val_w = test_cluster(self.model, self.single_head,
                                                      self.ulb_step_val_list[args.current_step],
                                                      args, return_ind=False)

            print('------>[Joint Head]: Single Step Test W/ Clustering')
            acc_joint_this_step_val_w = test_cluster(self.model, self.joint_head,
                                                     self.ulb_step_val_list[args.current_step],
                                                     args, return_ind=False)

            print('------>[Joint Head]: All Step Test W/ Clustering')
            acc_all_val_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_val,
                                                 args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                "val/acc_single_this_step": acc_single_this_step_val_w,
                "val/acc_joint_this_step": acc_joint_this_step_val_w,
                "val/acc_joint_all": acc_all_val_w_cluster
            }, step=epoch)

            print('\n======================================')
            print('---> Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")

            print('\n---> Joint Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_joint_this_step_val_w}")
            print(f"Acc_All_W_cluster    = {acc_all_val_w_cluster}")
            print('======================================')

    def test_init(self, args):
        # === Single Head ===
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.model, self.single_head,
                                                           self.ulb_step_test_list[args.current_step], args,
                                                           return_ind=False)

        print('------>[Single Head] This Step Test W/O Clustering')
        acc_single_head_this_step_wo_cluster = test_ind_cluster(self.model, self.single_head,
                                                                self.single_head,
                                                                self.ulb_step_test_list[args.current_step], 0, args,
                                                                ind_gen_loader=self.ulb_step_val_list[args.current_step]
                                                                )

        print('\n========================================================')
        print('      Final Test Output (test split, joint head only)     ')
        print(f"\n---> [S{args.current_step} Single Head]")
        print(f"Acc_this_step w/ clustering   = {acc_single_head_this_step_w_cluster}")
        print(f"Acc_this_step w/o clustering  = {acc_single_head_this_step_wo_cluster}")
        print('========================================================')

    def test_IL(self, args):
        # === Single Head ===
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.model, self.single_head,
                                                           self.ulb_step_test_list[args.current_step], args,
                                                           return_ind=False)

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
            if s < args.current_step:
                # s < current_step: use prev-learned model and single head to generate correct index
                this_step_test_wo = test_ind_cluster_unlocked(test_model=self.model, test_head=self.joint_head,
                                                              ind_gen_model=self.prev_pair_list[s][0],
                                                              ind_gen_head=self.prev_pair_list[s][1],
                                                              test_loader=self.ulb_step_test_list[s], step=s, args=args,
                                                              ind_gen_loader=self.ulb_step_val_list[s])
                acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo
            else:
                # s == current_step: use current-learned model and single head to generate correct index
                this_step_test_wo = test_ind_cluster_unlocked(test_model=self.model, test_head=self.joint_head,
                                                              ind_gen_model=self.model,
                                                              ind_gen_head=self.single_head,
                                                              test_loader=self.ulb_step_test_list[s], step=s, args=args,
                                                              ind_gen_loader=self.ulb_step_val_list[s])
                acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------>[Joint Head] All-Prev-Steps Test W/ Clustering')
        acc_all_prev_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_prev_test, args)

        print('------>[Joint Head] All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.model, self.joint_head, self.ulb_all_test, args)

        print('------>[Joint Head] All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + args.current_step)]

        acc_all_prev_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        acc_all_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list, args)


        print('\n========================================================')
        print('      Final Test Output (test split, joint head only)     ')
        print(f"\n---> [S{args.current_step} Single Head]")
        print(f"Acc_this_step             = {acc_single_head_this_step_w_cluster}")

        print("\n---> [Joint Head]")
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

    def save_single(self, model_path, head_path):
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.single_head.state_dict(), head_path)
        print("Task-specific single student backbone saved to {}.".format(model_path))
        print("Task-specific single student head saved to {}.".format(head_path))

    def save_joint(self, model_path, head_path):
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.joint_head.state_dict(), head_path)
        print("Task-agnostic joint backbone saved to {}.".format(model_path))
        print("Task-agnostic joint head saved to {}.".format(head_path))


