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

from utils.ResTune_testers import test_cluster, test_ind_cluster
from utils.ResTune_utils import feat2prob, target_distribution, init_prob_kmeans
from utils.ResTune_losses import BCE, HLoss, compute_sim_loss, ZeroPaddingCrossEntropy

# Global
LOCKED = 0
UNLOCKED = 1

step_model_dict = {
    'encoder':  None,
    'head_mix': None,
    'head_res': None,
}


def calculate_weighted_avg(step_acc_list, args):
    acc = 0.
    num_discovered = 0
    for s in range(len(step_acc_list)):
        this_num_novel = args.num_novel_interval
        acc += step_acc_list[s] * this_num_novel
        num_discovered += this_num_novel

    acc /= num_discovered
    return acc


class ResTune:
    def __init__(self,
                 model_dict,
                 sinkhorn,
                 eval_results_recorder,
                 step_train_loader_list,                                                # train
                 step_val_loader_list, prev_val_loader_list, all_val_loader_list,       # val
                 step_test_loader_list, prev_test_loader_list, all_test_loader_list):   # test
        # Models
        #   |- Dynamic model
        self.model_dict = model_dict                                   # shared encoder
        self.status_encoder = LOCKED

        #   |- Old model dict
        self.old_model_dict = dict((f"step{s}", copy.deepcopy(step_model_dict)) for s in range(len(self.model_dict)))

        # Pseudo-label generator
        self.sinkhorn = sinkhorn

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
        print(f"\t\t\t\t\tResTune-Stage-WarmUp [{1+step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)
        if step == 0:
            self.warmup_init(args=args)
        else:
            self.warmup_incremental(args=args, il_step=step)

    def train(self, args, step=0):
        print("=" * 100)
        print(f"\t\t\t\t\tResTune-Stage-Train [{1+step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)
        if step == 0:
            self.train_init(args=args)
        else:
            self.train_incremental(args=args, il_step=step)

    def warmup_init(self, args):
        # generate param list for optimizer
        param_list = list(self.model_dict['step0']['encoder'].parameters()) \
                     + list(self.model_dict['step0']['head_mix'].parameters())\
                     + list(self.model_dict['step0']['head_res'].parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_warmup, eta_min=args.lr * 1e-3)
        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs_warmup):
            # create loss statistics recorder for each loss
            loss_uno_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.model_dict['step0']['encoder'].eval()
            self.model_dict['step0']['head_mix'].train()
            self.model_dict['step0']['head_res'].train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.step_train_loader_list[0])):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # weight.data.shape = # of classes x 768
                    weight_temp = self.model_dict['step0']['head_mix'].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.model_dict['step0']['head_mix'].last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model_dict['step0']['encoder'](x_v0)
                feat_v1 = self.model_dict['step0']['encoder'](x_v1)

                # Single head output
                output_v0 = self.model_dict['step0']['head_mix'](feat_v0)
                output_v1 = self.model_dict['step0']['head_mix'](feat_v1)

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
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                "step-0/loss/WarmUp_uno": loss_uno_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs_warmup, loss_uno_record.avg))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(
                test_step=0,
                total_step=0,
                model_dict=self.model_dict,
                test_loader=self.step_val_loader_list[0],
                args=args,
                task_agnostic=False
            )

            # wandb metrics logging
            wandb.log({
                "step-0/val_acc/WarmUp_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

    def train_init(self, args):
        # Unlock last-2 blocks of ViT
        self.unlock_encoder(step=0, grad_from_block=args.grad_from_block)

        # generate param list for optimizer
        param_list = list(self.model_dict['step0']['encoder'].parameters()) \
                     + list(self.model_dict['step0']['head_mix'].parameters())\
                     + list(self.model_dict['step0']['head_res'].parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        # Epoch train
        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_record = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.model_dict['step0']['encoder'].train()
            self.model_dict['step0']['head_mix'].train()
            self.model_dict['step0']['head_res'].train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.step_train_loader_list[0])):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # weight.data.shape = # of classes x 768
                    weight_temp = self.model_dict['step0']['head_mix'].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.model_dict['step0']['head_mix'].last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model_dict['step0']['encoder'](x_v0)
                feat_v1 = self.model_dict['step0']['encoder'](x_v1)

                # Single head output
                output_v0 = self.model_dict['step0']['head_mix'](feat_v0)
                output_v1 = self.model_dict['step0']['head_mix'](feat_v1)

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
            acc_single_this_step_val_w = test_cluster(
                test_step=0,
                total_step=0,
                model_dict=self.model_dict,
                test_loader=self.step_val_loader_list[0],
                args=args,
                task_agnostic=False
            )

            # wandb metrics logging
            wandb.log({
                "step-0/val_acc/Train_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch+args.epochs_warmup)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

        self.lock_encoder(step=0)

    def warmup_incremental(self, args, il_step=1):
        param_list = list(self.model_dict['step0']['encoder'].parameters())\
                     + list(self.model_dict['step0']['head_mix'].parameters())\
                     + list(self.model_dict['step0']['head_res'].parameters())
        for i in range(1, il_step+1):
            param_list += list(self.model_dict[f'step{i}']['encoder'].parameters())\
                          + list(self.model_dict[f'step{i}']['head_mix'].parameters()) \
                          + list(self.model_dict[f'step{i}']['head_res'].parameters())


        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_warmup, eta_min=args.lr * 1e-3)
        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs_warmup):
            loss_uno_record = AverageMeter()  # UNO loss recorder

            for i in range(il_step+1):
                self.model_dict[f'step{i}']['encoder'].eval()
                if i == il_step:
                    self.model_dict[f'step{i}']['head_mix'].train()
                    self.model_dict[f'step{i}']['head_res'].train()
                else:
                    self.model_dict[f'step{i}']['head_mix'].eval()
                    self.model_dict[f'step{i}']['head_res'].eval()

            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.step_train_loader_list[il_step])):
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                with torch.no_grad():
                    # mix head
                    weight_temp = self.model_dict[f'step{il_step}']['head_mix'].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.model_dict[f'step{il_step}']['head_mix'].last_layer.weight.copy_(weight_temp)
                    # residual head
                    weight_temp = self.model_dict[f'step{il_step}']['head_res'].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.model_dict[f'step{il_step}']['head_res'].last_layer.weight.copy_(weight_temp)

                # Feature extraction
                # base feat
                shared_feat_v0 = self.model_dict['step0']['encoder'].get_intermediate_layers(x_v0)
                shared_feat_v1 = self.model_dict['step0']['encoder'].get_intermediate_layers(x_v1)

                # basic feat from init-step block
                mix_feat_v0 = self.model_dict['step0']['encoder'](x_v0)
                mix_feat_v1 = self.model_dict['step0']['encoder'](x_v1)
                # merge basic feat step-by-step
                for i in range(1, il_step):
                    mix_feat_v0 += self.model_dict[f'step{i}']['encoder'](shared_feat_v0)
                    mix_feat_v1 += self.model_dict[f'step{i}']['encoder'](shared_feat_v1)

                # residual feat
                res_feat_v0 = self.model_dict[f'step{il_step}']['encoder'](shared_feat_v0)
                res_feat_v1 = self.model_dict[f'step{il_step}']['encoder'](shared_feat_v1)

                # mix basic & residual feats
                mix_feat_v0 += res_feat_v0
                mix_feat_v1 += res_feat_v1

                # Mix head output
                mix_output_v0 = self.model_dict[f'step{il_step}']['head_mix'](mix_feat_v0)
                mix_output_v1 = self.model_dict[f'step{il_step}']['head_mix'](mix_feat_v1)
                # Res head output
                res_output_v0 = self.model_dict[f'step{il_step}']['head_res'](res_feat_v0)
                res_output_v1 = self.model_dict[f'step{il_step}']['head_res'](res_feat_v1)

                # Sinkhorn swipe-pseudo labeling
                mix_target_v0 = self.sinkhorn(mix_output_v1)
                mix_target_v1 = self.sinkhorn(mix_output_v0)

                res_target_v0 = self.sinkhorn(res_output_v1)
                res_target_v1 = self.sinkhorn(res_output_v0)

                mixed_logits = torch.cat([mix_output_v0, mix_output_v1, res_output_v0, res_output_v1], dim=0)
                mixed_targets = torch.cat([mix_target_v0, mix_target_v1, res_target_v0, res_target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), res_output_v0.size(0))

                optimizer.zero_grad()
                loss_uno.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"step-{il_step}/loss/WarmUp_uno": loss_uno_record.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs_warmup,
                                                                   loss_uno_record.avg))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(
                test_step=il_step,
                total_step=il_step,
                model_dict=self.model_dict,
                test_loader=self.step_val_loader_list[il_step],
                args=args,
                task_agnostic=False
            )

            # wandb metrics logging
            wandb.log({
                f"step-{il_step}/val_acc/WarmUp_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

    def train_incremental(self, args, il_step=1):
        self.unlock_encoder(step=il_step, grad_from_block=args.grad_from_block)

        param_list = list(self.model_dict['step0']['encoder'].parameters())\
                     + list(self.model_dict['step0']['head_mix'].parameters()) \
                     + list(self.model_dict['step0']['head_res'].parameters())

        for i in range(1, il_step+1):
            param_list += list(self.model_dict[f'step{i}']['encoder'].parameters()) \
                          + list(self.model_dict[f'step{i}']['head_mix'].parameters()) \
                          + list(self.model_dict[f'step{i}']['head_res'].parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_uno = ZeroPaddingCrossEntropy(temperature=args.softmax_temp)

        for epoch in range(args.epochs):
            loss_uno_record = AverageMeter()    # UNO loss recorder
            loss_kd_record = AverageMeter()     # KD loss recorder

            for i in range(il_step+1):
                self.model_dict[f'step{i}']['encoder'].train()
                self.model_dict[f'step{i}']['head_mix'].train()
                self.model_dict[f'step{i}']['head_res'].train()

            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.step_train_loader_list[il_step])):
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                with torch.no_grad():
                    # mix head
                    weight_temp = self.model_dict[f'step{il_step}']['head_mix'].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.model_dict[f'step{il_step}']['head_mix'].last_layer.weight.copy_(weight_temp)
                    # residual head
                    weight_temp = self.model_dict[f'step{il_step}']['head_res'].last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    self.model_dict[f'step{il_step}']['head_res'].last_layer.weight.copy_(weight_temp)

                # Feature extraction
                # base feat
                shared_feat_v0 = self.model_dict['step0']['encoder'].get_intermediate_layers(x_v0)
                shared_feat_v1 = self.model_dict['step0']['encoder'].get_intermediate_layers(x_v1)

                # basic feat
                mix_feat_v0 = self.model_dict['step0']['encoder'](x_v0)
                mix_feat_v1 = self.model_dict['step0']['encoder'](x_v1)
                for i in range(1, il_step):
                    mix_feat_v0 += self.model_dict[f'step{i}']['encoder'](shared_feat_v0)
                    mix_feat_v1 += self.model_dict[f'step{i}']['encoder'](shared_feat_v1)

                # residual feat
                res_feat_v0 = self.model_dict[f'step{il_step}']['encoder'](shared_feat_v0)
                res_feat_v1 = self.model_dict[f'step{il_step}']['encoder'](shared_feat_v1)

                # mixed basic & residual feat
                mix_feat_v0 += res_feat_v0
                mix_feat_v1 += res_feat_v1

                # Mix head output
                mix_output_v0 = self.model_dict[f'step{il_step}']['head_mix'](mix_feat_v0)
                mix_output_v1 = self.model_dict[f'step{il_step}']['head_mix'](mix_feat_v1)

                # Res head output
                res_output_v0 = self.model_dict[f'step{il_step}']['head_res'](res_feat_v0)
                res_output_v1 = self.model_dict[f'step{il_step}']['head_res'](res_feat_v1)

                # Sinkhorn swipe-pseudo labeling
                mix_target_v0 = self.sinkhorn(mix_output_v1)
                mix_target_v1 = self.sinkhorn(mix_output_v0)

                res_target_v0 = self.sinkhorn(res_output_v1)
                res_target_v1 = self.sinkhorn(res_output_v0)

                mixed_logits = torch.cat([mix_output_v0, mix_output_v1, res_output_v0, res_output_v1], dim=0)
                mixed_targets = torch.cat([mix_target_v0, mix_target_v1, res_target_v0, res_target_v1], dim=0)

                loss_uno = criterion_uno(mixed_logits, mixed_targets)
                loss_uno_record.update(loss_uno.item(), res_output_v0.size(0))

                # old
                old_feat = self.old_model_dict['step0']['encoder'](x_v0)
                old_output = self.old_model_dict['step0']['head_mix'](old_feat)
                soft_target = F.softmax(old_output / 2.0, dim=1)

                # new
                new_feat = self.model_dict['step0']['encoder'](x_v0)
                new_output = self.old_model_dict['step0']['head_mix'](new_feat)
                logp = F.log_softmax(new_output / 2.0, dim=1)

                # KD-Loss for init-step
                loss_kd = -torch.mean(torch.sum(soft_target * logp, dim=1))

                old_shared_feat_v0 = self.old_model_dict[f'step0']['encoder'].get_intermediate_layers(x_v0)
                for prev_step in range(1, il_step):
                    # old
                    old_feat = old_feat + self.old_model_dict[f'step{prev_step}']['encoder'](old_shared_feat_v0)
                    old_output = self.old_model_dict[f'step{prev_step}']['head_mix'](old_feat)
                    soft_target = F.softmax(old_output / 2.0, dim=1)
                    # new
                    new_feat = new_feat + self.model_dict[f'step{prev_step}']['encoder'](shared_feat_v0)
                    new_output = self.old_model_dict[f'step{prev_step}']['head_mix'](new_feat)
                    logp = F.log_softmax(new_output / 2.0, dim=1)

                    # accumulate KD-Loss
                    loss_kd += (-torch.mean(torch.sum(soft_target * logp, dim=1)) * args.w_kd)

                loss_kd_record.update(loss_kd.item(), res_output_v0.size(0))

                # Total loss
                total_loss = loss_uno + loss_kd

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"step-{il_step}/loss/Train_uno": loss_uno_record.avg,
                f"step-{il_step}/loss/Train_kd": loss_kd_record.avg,

            }, step=epoch+args.epochs_warmup)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: UNO={:.4f} KD={:.4F}'.format(1 + epoch, args.epochs,
                                                                                 loss_uno_record.avg,
                                                                                 loss_kd_record.avg))
            print('===========================================')

            print('------>[Single Head]: Single Step Test W/ Clustering')
            # Only test current step
            acc_single_this_step_val_w = test_cluster(
                test_step=il_step,
                total_step=il_step,
                model_dict=self.model_dict,
                test_loader=self.step_val_loader_list[il_step],
                args=args,
                task_agnostic=False
            )

            # wandb metrics logging
            wandb.log({
                f"step-{il_step}/val_acc/Train_single_head_this_step_W_cluster": acc_single_this_step_val_w,
            }, step=epoch+args.epochs_warmup)

            print('\n======================================')
            print('Single Head Val. Evaluation')
            print(f"Acc_this_step_W_cluster    = {acc_single_this_step_val_w}")
            print('======================================')

        self.lock_encoder(step=il_step)

    def test(self, args, step=0):
        # === Single Head ===
        print('------>[Single Head] This Step Test W/ Clustering')
        acc_single_head_this_step_w_cluster = test_cluster(
            test_step=step,
            total_step=step,
            model_dict=self.model_dict,
            test_loader=self.step_test_loader_list[step],
            args=args,
            task_agnostic=False
        )

        # === Joint Head ===
        print('------>[Joint Head] Individual Steps Test W/ Clustering')
        acc_step_test_w_cluster_dict = dict((f"Step{s}_only", -1) for s in range(args.num_steps))

        for s in range(1+step):
            this_step_test_w = test_cluster(
                test_step=s,
                total_step=step,
                model_dict=self.model_dict,
                test_loader=self.step_test_loader_list[s],
                args=args,
                task_agnostic=True
            )
            acc_step_test_w_cluster_dict[f"Step{s}_only"] = this_step_test_w

        print('------>[Joint Head] Individual Steps Test W/O Clustering')
        acc_step_test_wo_cluster_dict = dict((f"Step{s}_only", -1) for s in range(args.num_steps))

        for s in range(1 + step):
            this_step_test_wo = test_ind_cluster(
                test_step=s,
                total_step=step,
                model_dict=self.model_dict,
                test_loader=self.step_test_loader_list[s],
                ind_gen_loader=self.step_val_loader_list[s],
                args=args
            )

            acc_step_test_wo_cluster_dict[f"Step{s}_only"] = this_step_test_wo

        print('------> All-Prev-Steps Test W/ Clustering')
        if step > 0:
            acc_all_prev_test_w_cluster = test_cluster(
                test_step=1000,
                total_step=step,
                model_dict=self.model_dict,
                test_loader=self.prev_test_loader_list[step-1],
                args=args,
                task_agnostic=True
            )
        else:
            acc_all_prev_test_w_cluster = -1

        print('------> All-Steps Test W/ Clustering')
        acc_all_test_w_cluster = test_cluster(
            test_step=1000,
            total_step=step,
            model_dict=self.model_dict,
            test_loader=self.all_test_loader_list[step],
            args=args,
            task_agnostic=True
        )

        print('------> All (all/prev) Steps Test W/O Clustering')
        step_acc_test_wo_cluster_list = [acc_step_test_wo_cluster_dict[f"Step{s}_only"]
                                         for s in range(1 + step)]

        if step > 0:
            acc_all_prev_test_wo_cluster = calculate_weighted_avg(step_acc_test_wo_cluster_list[:-1], args)
        else:
            acc_all_prev_test_wo_cluster = -1

        acc_all_test_wo_cluster = calculate_weighted_avg(step_acc_test_wo_cluster_list, args)

        self.eval_results_recorder.update_step(step,
                                               acc_single_head_this_step_w_cluster,
                                               acc_all_prev_test_w_cluster, acc_all_prev_test_wo_cluster,
                                               acc_all_test_w_cluster, acc_all_test_wo_cluster,
                                               acc_step_test_w_cluster_dict, acc_step_test_wo_cluster_dict
                                               )

    def show_eval_result(self, step=0):
        self.eval_results_recorder.show_step(step=step)

    def unlock_encoder(self, step=0, grad_from_block=10):
        for name, m in self.model_dict['step0']['encoder'].named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= grad_from_block:
                    m.requires_grad = True

        if step > 0:
            for i in range(1, step+1):
                for m in self.model_dict[f'step{i}']['encoder'].parameters():
                    m.requires_grad = True
        self.status_encoder = UNLOCKED

    def lock_encoder(self, step=0):
        for i in range(step+1):
            for m in self.model_dict[f'step{i}']['encoder'].parameters():
                m.requires_grad = False
        self.status_encoder = LOCKED

    def duplicate_old_model(self, args, step=0):
        for s in range(step+1):
            for k in self.model_dict[f'step{s}'].keys():
                self.old_model_dict[f'step{s}'][k] = copy.deepcopy(self.model_dict[f'step{s}'][k])
                self.old_model_dict[f'step{s}'][k] = self.old_model_dict[f'step{s}'][k].to(args.device)
                for m in self.old_model_dict[f'step{s}'][k].parameters():
                    m.requires_grad = False
                self.old_model_dict[f'step{s}'][k].eval()



