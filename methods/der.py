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

from utils.der_testers import test_cluster, test_ind_cluster
from utils.der_losses import DERLoss, ZeroPaddingCrossEntropy
from utils.der_memory import Buffer
# Global
LOCKED = 0
UNLOCKED = 1


class DER:
    def __init__(self,
                 args,
                 encoder,
                 step_single_head_list,
                 joint_head_list,
                 sinkhorn,
                 eval_results_recorder,
                 step_train_loader_list,                                                # train
                 step_val_loader_list, prev_val_loader_list, all_val_loader_list,       # val
                 step_test_loader_list, prev_test_loader_list, all_test_loader_list):   # test

        self.last_step_id = args.num_steps - 1
        # Models
        #   |- Dynamic model
        self.encoder_shared = encoder                                   # shared encoder
        self.status_encoder = LOCKED
        self.step_single_head_list = step_single_head_list              # task-specific head
        self.joint_head_list = joint_head_list                          # task-agnostic eval head
                                                                        # 2 4 6 8 10

        # Pseudo-label generator
        self.sinkhorn = sinkhorn

        # Buffers
        self.step_buffer_list = [Buffer(args) for s in range(self.last_step_id)]

        # Data loaders
        # |- train
        self.step_train_loader_list = step_train_loader_list
        # |- val
        self.step_val_loader_list = step_val_loader_list
        self.prev_val_loader_list = prev_val_loader_list                # len = num_steps - 1
        self.all_val_loader_list = all_val_loader_list
        # |- test
        self.step_test_loader_list = step_test_loader_list
        self.prev_test_loader_list = prev_test_loader_list              # len = num_steps - 1
        self.all_test_loader_list = all_test_loader_list

        # Evaluation results
        self.eval_results_recorder = eval_results_recorder

    def warmup(self, args, step=0):
        print("=" * 100)
        print(f"\t\t\t\t\tDER-Stage-WarmUp [{1+step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)
        if step == 0:
            self.warmup_init(args=args)
        else:
            self.warmup_incremental(args=args, step=step)

    def train(self, args, step=0):
        print("=" * 100)
        print(f"\t\t\t\t\tDER-Stage-Train [{1+step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        if step == 0:
            self.train_init(args=args)
        else:
            self.train_incremental(args=args, il_step=step)

    def warmup_init(self, args):
        """train classifier with 100% frozen backbone"""
        # generate param list for optimizer
        param_list = list(self.step_single_head_list[0].parameters()) + list(self.encoder_shared.parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_warmup, eta_min=args.lr * 1e-3)
        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs_warmup):
            # create loss statistics recorder for each loss
            loss_uno_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.encoder_shared.eval()
            self.step_single_head_list[0].train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), no_aug_x, _, idx) in enumerate(tqdm(self.step_train_loader_list[0])):
                # send the vars to GPU
                x_v0, x_v1, no_aug_x = x_v0.to(args.device), x_v1.to(args.device), no_aug_x.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # weight.data.shape = # of classes x 768
                    weight_temp = self.step_single_head_list[0].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.step_single_head_list[0].last_layer.weight.copy_(weight_temp)
                # self.single_head.normalize_prototypes()

                # Feature extraction
                feat_v0 = self.encoder_shared(x_v0)
                feat_v1 = self.encoder_shared(x_v1)

                # Single head output
                output_v0 = self.step_single_head_list[0](feat_v0)
                output_v1 = self.step_single_head_list[0](feat_v1)

                # Sinkhorn swipe-pseudo labeling
                target_v0 = self.sinkhorn(output_v1)
                target_v1 = self.sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss_uno.backward()
                optimizer.step()

                self.step_buffer_list[0].add_data(examples=no_aug_x, logits=output_v0.data)
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"step-{0}/loss/WarmUp_uno": loss_uno_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs_warmup, loss_uno_record.avg))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.encoder_shared, self.step_single_head_list[0],
                                                      self.step_val_loader_list[0],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                f"step-{0}/val_acc/WarmUp_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

    def train_init(self, args):
        # Unlock encoder before training
        self.unlock_encoder(grad_from_block=args.grad_from_block)

        # generate param list for optimizer
        param_list = list(self.encoder_shared.parameters()) + list(self.step_single_head_list[0].parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        # Epoch train
        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.encoder_shared.train()
            self.step_single_head_list[0].train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), no_aug_x, _, idx) in enumerate(tqdm(self.step_train_loader_list[0])):
                # send the vars to GPU
                x_v0, x_v1, no_aug_x = x_v0.to(args.device), x_v1.to(args.device), no_aug_x.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # weight.data.shape = # of classes x 768
                    weight_temp = self.step_single_head_list[0].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.step_single_head_list[0].last_layer.weight.copy_(weight_temp)
                # self.single_head.normalize_prototypes()

                # Feature extraction
                feat_v0 = self.encoder_shared(x_v0)
                feat_v1 = self.encoder_shared(x_v1)

                # Single head output
                output_v0 = self.step_single_head_list[0](feat_v0)
                output_v1 = self.step_single_head_list[0](feat_v1)

                # Sinkhorn swipe-pseudo labeling
                target_v0 = self.sinkhorn(output_v1)
                target_v1 = self.sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss_uno.backward()
                optimizer.step()
                self.step_buffer_list[0].add_data(examples=no_aug_x, logits=output_v0.data)
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"step-0/loss/Train_uno": loss_uno_record.avg,
            }, step=epoch+args.epochs_warmup)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_uno_record.avg))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.encoder_shared, self.step_single_head_list[0],
                                                      self.step_val_loader_list[0],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                "step-0/val_acc/Train_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch+args.epochs_warmup)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

        # Lock encoder after training
        self.lock_encoder()

    def warmup_incremental(self, args, step=1):
        # generate param list for optimizer
        param_list = list(self.step_single_head_list[step].parameters()) + list(self.encoder_shared.parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_warmup, eta_min=args.lr * 1e-3)

        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs_warmup):
            # create loss statistics recorder for each loss
            loss_uno_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.encoder_shared.eval()
            self.step_single_head_list[step].train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), no_aug_x, _, idx) in enumerate(tqdm(self.step_train_loader_list[step])):
                # send the vars to GPU
                x_v0, x_v1, no_aug_x = x_v0.to(args.device), x_v1.to(args.device), no_aug_x.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # weight.data.shape = # of classes x 768
                    weight_temp = self.step_single_head_list[step].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.step_single_head_list[step].last_layer.weight.copy_(weight_temp)
                # self.single_head.normalize_prototypes()

                # Feature extraction
                feat_v0 = self.encoder_shared(x_v0)
                feat_v1 = self.encoder_shared(x_v1)

                # Single head output
                output_v0 = self.step_single_head_list[step](feat_v0)
                output_v1 = self.step_single_head_list[step](feat_v1)

                # Sinkhorn swipe-pseudo labeling
                target_v0 = self.sinkhorn(output_v1)
                target_v1 = self.sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss_uno.backward()
                optimizer.step()
                if step < self.last_step_id:
                    self.step_buffer_list[step].add_data(examples=no_aug_x, logits=output_v0.data)
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"step-{step}/loss/WarmUp_uno": loss_uno_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: UNO={:.4f}'.format(1 + epoch, args.epochs_warmup,
                                                                       loss_uno_record.avg,
                                                                       ))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.encoder_shared, self.step_single_head_list[step],
                                                      self.step_val_loader_list[step],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                f"step-{step}/val_acc/WarmUp_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

    def train_incremental(self, args, il_step=1):
        # Unlock encoder before training
        self.unlock_encoder(grad_from_block=args.grad_from_block)

        # generate param list for optimizer
        #   |- encoder
        param_list = list(self.encoder_shared.parameters())
        #   |- task-specific head
        for s in range(1+il_step):
            param_list += list(self.step_single_head_list[s].parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)
        criterion_der = DERLoss(alpha_der=args.alpha_der)

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_record = AverageMeter()    # UNO loss recorder
            loss_der_record = AverageMeter()    # DER loss recorder

            # switch the models to train mode
            self.encoder_shared.train()
            for s in range(1+il_step):
                self.step_single_head_list[s].train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), no_aug_x, _, idx) in enumerate(tqdm(self.step_train_loader_list[il_step])):
                # send the vars to GPU
                # we se x_v0 as data_new
                x_v0, x_v1, no_aug_x = x_v0.to(args.device), x_v1.to(args.device), no_aug_x.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    for s in range(1+il_step):
                        weight_temp = self.step_single_head_list[s].last_layer.weight.data.clone()
                        weight_temp = F.normalize(weight_temp, dim=1, p=2)
                        self.step_single_head_list[s].last_layer.weight.copy_(weight_temp)

                # self.single_head.normalize_prototypes()

                # Feature extraction
                feat_v0 = self.encoder_shared(x_v0)
                feat_v1 = self.encoder_shared(x_v1)

                # Single head output
                output_v0 = self.step_single_head_list[il_step](feat_v0)
                output_v1 = self.step_single_head_list[il_step](feat_v1)

                # Sinkhorn swipe-pseudo labeling
                target_v0 = self.sinkhorn(output_v1)
                target_v1 = self.sinkhorn(output_v0)

                mixed_logits = torch.cat([output_v0, output_v1], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), output_v0.size(0))

                # DER loss
                for prev_step in range(il_step):
                    buf_x, buf_output_old = self.step_buffer_list[prev_step].get_data(args.batch_size)
                    # print(buf_x.shape)
                    # print(buf_x[-1])
                    # print(buf_output_old.shape)
                    # print(buf_x[-1])

                    buf_feat_new = self.encoder_shared(buf_x)
                    buf_output_new = self.step_single_head_list[prev_step](buf_feat_new)
                    if prev_step == 0:
                        loss_der = criterion_der(buf_output_new, buf_output_old)
                    else:
                        loss_der += criterion_der(buf_output_new, buf_output_old)

                # Total loss
                loss_total = loss_uno + loss_der

                loss_der_record.update(loss_der.item(), output_v0.size(0))

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                if il_step < self.last_step_id:
                    self.step_buffer_list[il_step].add_data(examples=no_aug_x, logits=output_v0.data)
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"step-{il_step}/loss/Train_uno": loss_uno_record.avg,
                f"step-{il_step}/loss/Train_der": loss_der_record.avg,
            }, step=epoch+args.epochs_warmup)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: UNO={:.4f},\tDER={:.4f}'.format(1+epoch,
                                                                                    args.epochs,
                                                                                    loss_uno_record.avg,
                                                                                    loss_der_record.avg))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(self.encoder_shared, self.step_single_head_list[il_step],
                                                      self.step_val_loader_list[il_step],
                                                      args, return_ind=False)

            # wandb metrics logging
            wandb.log({
                f"step-{il_step}/val_acc/Train_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch+args.epochs_warmup)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

        # Lock encoder after training
        self.lock_encoder()

    def test(self, args, step=0):
        # create task-agnostic joint classifier
        self.fill_joint_head(args, current_step=step)

        # === Single Head ===
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(self.encoder_shared, self.step_single_head_list[step],
                                                           self.step_test_loader_list[step], args, return_ind=False)

        # === Joint Head ===
        print('------>[Joint Head] Individual Steps Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + step):
            this_step_test_w = test_cluster(self.encoder_shared, self.joint_head_list[step],
                                            self.step_test_loader_list[s], args)
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------>[Joint Head] Individual Steps Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict(
            (f"Step{s}_only", -1) for s in range(args.num_steps))
        for s in range(1 + step):
            this_step_test_wo = test_ind_cluster(self.encoder_shared, self.joint_head_list[step],
                                                 self.step_single_head_list[s],
                                                 self.step_test_loader_list[s], s, args,
                                                 ind_gen_loader=self.step_val_loader_list[s])
            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------> All-Prev-Steps Test W/ Clustering')
        if step > 0:
            acc_all_prev_test_w_cluster = test_cluster(self.encoder_shared, self.joint_head_list[step],
                                                       self.prev_test_loader_list[step-1], args)
        else:
            acc_all_prev_test_w_cluster = -1

        print('------> All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(self.encoder_shared, self.joint_head_list[step],
                                              self.all_test_loader_list[step], args)

        print('------> All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + step)]

        if step > 0:
            acc_all_prev_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        else:
            acc_all_prev_test_wo_cluster = -1

        acc_all_test_wo_cluster = self.calculate_weighted_avg(step_acc_test_wo_cluster_list, args)

        self.eval_results_recorder.update_step(step,
                                               acc_single_head_this_step_w_cluster,
                                               acc_all_prev_test_w_cluster, acc_all_prev_test_wo_cluster,
                                               acc_all_test_w_cluster, acc_all_test_wo_cluster,
                                               acc_step_test_w_cluster_dict, acc_step_test_wo_cluster_dict
                                               )

    def show_eval_result(self, step=0):
        self.eval_results_recorder.show_step(step=step)

    def save_model(self, step=0):
        pass

    def unlock_encoder(self, grad_from_block=11):
        # Only finetune layers from block 'grad_from_block' onwards
        for name, m in self.encoder_shared.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= grad_from_block:
                    m.requires_grad = True
        self.status_encoder = UNLOCKED

    def lock_encoder(self):
        for m in self.encoder_shared.parameters():
            m.requires_grad = False
        self.status_encoder = LOCKED

    def calculate_weighted_avg(self, step_acc_list, args):
        acc = 0.
        num_discovered = 0
        for s in range(len(step_acc_list)):
            this_num_novel = args.num_novel_interval
            acc += step_acc_list[s] * this_num_novel
            num_discovered += this_num_novel

        acc /= num_discovered
        return acc

    def fill_joint_head(self, args, current_step=0):
        for step in range(current_step):
            w_saved = self.step_single_head_list[step].last_layer.weight.data.clone()
            self.joint_head_list[current_step].last_layer.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w_saved)

        current_w = self.step_single_head_list[current_step].last_layer.weight.data.clone()
        self.joint_head_list[current_step].last_layer.weight.data[current_step*args.num_novel_interval:(1+current_step)*args.num_novel_interval].copy_(current_w)



