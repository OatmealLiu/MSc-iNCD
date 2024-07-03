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
from itertools import cycle
from utils.util import cluster_acc

from methods.testers import test_cluster, test_labeled_base
# from scipy.optimize import linear_sum_assignment
# from sklearn import metrics


class UNO:
    def __init__(self, model, head_base, head_novel, head_joint,
                 sinkhorn,
                 lb_train_loader, ulb_train_loader,
                 val_loader_base, val_loader_novel, #val_loader_all,
                 test_loader_base, test_loader_novel, #test_loader_all
                 ):
        # Models
        self.model = model
        self.head_base = head_base
        self.head_novel = head_novel
        self.head_joint = head_joint

        # Sinkhorn algo.
        self.sinkhorn = sinkhorn

        # Data loaders
        # |- train
        self.lb_train_loader = lb_train_loader
        self.ulb_train_loader = ulb_train_loader

        # |- validation
        self.val_loader_base = val_loader_base
        self.val_loader_novel = val_loader_novel
        # self.val_loader_all = val_loader_all

        # |- test
        self.test_loader_base = test_loader_base
        self.test_loader_novel = test_loader_novel
        # self.test_loader_all = test_loader_all

        # Acc cache
        self.pretrain_test_acc_lb_head_base = -1.0

    def lock_encoder(self):
        for m in self.model.parameters():
            m.requires_grad = False

    def calculate_ce_zero_padding(self, output, target, softmax_temp=0.1):
        # follow original UNO, temperature = 0.1
        preds = F.softmax(output / softmax_temp, dim=1)  # temperature
        preds = torch.clamp(preds, min=1e-8)
        preds = torch.log(preds)
        loss = -torch.mean(torch.sum(target * preds, dim=1))
        return loss

    def concat_joint_head(self, args):
        w_base = self.head_base.last_layer.weight.data.clone()
        w_novel = self.head_novel.last_layer.weight.data.clone()
        self.head_joint.last_layer.weight.data[:args.num_base].copy_(w_base)
        self.head_joint.last_layer.weight.data[args.num_base:].copy_(w_novel)

    def train_pretrain(self, args):
        # generate param list for optimizer
        # |- backbone
        # |- classifier
        param_list = list(self.model.parameters()) + list(self.head_base.parameters())

        # create optimizer
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # create lr scheduler
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_pretrain, eta_min=args.lr * 1e-3)

        for epoch in range(args.epochs_pretrain):
            # CE-loss recorder for supervised pre-training stage w/ label
            loss_lb_ce_recorder = AverageMeter()

            # switch the models to train mode
            self.model.train()
            self.head_base.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, (x, labels, idx) in enumerate(tqdm(self.lb_train_loader)):
                # labeled data and its labels
                x, target_x = x.to(args.device), labels.to(args.device)

                # normalize classifier weights
                self.head_base.normalize_prototypes()

                feat_lb = self.model(x)
                output_lb = self.head_base(feat_lb)

                loss_lb_ce = F.cross_entropy(output_lb/args.softmax_temp, target_x)
                loss_lb_ce_recorder.update(loss_lb_ce.item(), x.size(0))

                optimizer.zero_grad()
                loss_lb_ce.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/CE_label": loss_lb_ce_recorder.avg,
                f"loss/UNO_mix": 0.0,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1+epoch, args.epochs_pretrain,
                                                                   loss_lb_ce_recorder.avg))
            print('===========================================')

            # save model
            self.save_backbone(path=args.save_backbone_path)
            self.save_head_base(path=args.save_head_base_path)

            print('------>[Base Head] validation')
            # Only test current step
            # Should use val dataset
            print('------>            base classes')
            acc_lb_head_base = test_labeled_base(args, self.model, self.head_base, self.val_loader_base)

            # wandb metrics logging
            wandb.log({
                "val_acc/acc_base": acc_lb_head_base,
                "val_acc/acc_novel": 0.0,
            }, step=epoch)

            print('\n======================================')
            print('On-the-fly Training Evaluation:')
            print(f"Acc_base    = {acc_lb_head_base}")
            print('======================================')

    def train_ncd(self, args):
        # lock the encoder for experimental purpose
        if args.lock_ncd_stage == 'lock':
            self.lock_encoder()

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.head_base.parameters()) \
                     + list(self.head_novel.parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_ncd, eta_min=args.lr * 1e-3)

        for epoch in range(args.epochs_ncd):
            ulb_train_loader_iter = cycle(self.ulb_train_loader)

            # create loss statistics recorder for each loss
            loss_uno_recorder = AverageMeter()  # UNO loss recorder

            # switch the models to train mode
            self.model.train()
            self.head_base.train()
            self.head_novel.train()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, (x, labels, idx) in enumerate(tqdm(self.lb_train_loader)):
                # labeled data and its labels
                x, target_x = x.to(args.device), labels.to(args.device)

                # unlabeled data
                ((ux_v0, ux_v1), _, idx) = next(ulb_train_loader_iter)
                ux_v0, ux_v1 = ux_v0.to(args.device), ux_v1.to(args.device)

                # normalize classifier weights
                self.head_base.normalize_prototypes()
                self.head_novel.normalize_prototypes()

                # Feature extraction
                feat_x = self.model(x)
                feat_ux_v0 = self.model(ux_v0)
                feat_ux_v1 = self.model(ux_v1)

                # Prediction
                output_x_base = self.head_base(feat_x)
                output_x_novel = self.head_novel(feat_x)
                output_x = torch.cat([output_x_base, output_x_novel], dim=-1)

                output_ux_v0_base = self.head_base(feat_ux_v0)
                output_ux_v0_novel = self.head_novel(feat_ux_v0)
                output_ux_v0 = torch.cat([output_ux_v0_base, output_ux_v0_novel], dim=-1)

                output_ux_v1_base = self.head_base(feat_ux_v1)
                output_ux_v1_novel = self.head_novel(feat_ux_v1)
                output_ux_v1 = torch.cat([output_ux_v1_base, output_ux_v1_novel], dim=-1)

                # Label creation
                #   |- base classes (ground truth)
                #   Zero-padding: Transform label to one-hot
                # batch_size = x.size(0)
                # target_x = torch.zeros(batch_size, args.num_base).to(args.device).scatter_(1, target_x.view(-1, 1).long(), 1)
                target_x = F.one_hot(target_x, num_classes=args.num_base).float().to(args.device)

                #   |- novel classes (pseudo-label generated by sinkhorn)
                with torch.no_grad():
                    # cross pseudo-labeling
                    target_ux_v0 = self.sinkhorn(output_ux_v1_novel)
                    target_ux_v1 = self.sinkhorn(output_ux_v0_novel)

                # Mix-up
                #   |- logits
                logits_all = torch.cat([output_x, output_ux_v0, output_ux_v1], dim=0)
                # print(logits_all.shape)
                #   |- targets
                #       |- labeled
                target_x_zp = torch.zeros_like(output_x).to(args.device)
                target_x_zp[:, :args.num_base] = target_x.type_as(target_x_zp)
                #       |- unlabeled view-0
                target_ux_v0_zp = torch.zeros_like(output_ux_v0).to(args.device)
                target_ux_v0_zp[:, args.num_base:] = target_ux_v0.type_as(target_ux_v0_zp)
                #       |- unlabeled view-1
                target_ux_v1_zp = torch.zeros_like(output_ux_v1).to(args.device)
                target_ux_v1_zp[:, args.num_base:] = target_ux_v1.type_as(target_ux_v1_zp)
                #       |- mixed-up
                targets_all = torch.cat([target_x_zp, target_ux_v0_zp, target_ux_v1_zp])

                # print(target_x_zp.shape)
                # print(target_ux_v0_zp.shape)
                # print(target_ux_v1_zp.shape)
                # print(targets_all.shape)

                loss_uno = self.calculate_ce_zero_padding(logits_all, targets_all, softmax_temp=args.softmax_temp)
                loss_uno_recorder.update(loss_uno.item(), x.size(0))

                optimizer.zero_grad()
                loss_uno.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/CE_label": 0.0,
                f"loss/UNO_mix": loss_uno_recorder.avg,
            }, step=epoch+args.epochs_pretrain)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs_ncd, loss_uno_recorder.avg))
            print('===========================================')

            # save model
            self.save_backbone(path=args.save_backbone_path)
            self.save_head_base(path=args.save_head_base_path)
            self.save_head_novel(path=args.save_head_novel_path)

            print('------>[Task-specific Head Validation]')
            print('------>        base classes')
            acc_lb_head_base = test_labeled_base(args, self.model, self.head_base, self.val_loader_base)
            print('------>        novel classes')
            acc_ulb_head_novel = test_cluster(self.model, self.head_novel, self.val_loader_novel, args,
                                              return_ind=False)

            # wandb metrics logging
            wandb.log({
                "val_acc/acc_base": acc_lb_head_base,
                "val_acc/acc_novel": acc_ulb_head_novel,
            }, step=epoch+args.epochs_pretrain)

            print('\n======================================')
            print('On-the-fly Training Evaluation:')
            print(f"Acc_base    = {acc_lb_head_base}")
            print(f"Acc_novel    = {acc_ulb_head_novel}")
            print('======================================')

    def test_pretrain(self, args):
        self.pretrain_test_acc_lb_head_base = test_labeled_base(args, self.model, self.head_base, self.test_loader_base)

        print('\n========================================================')
        print('             Final Test Output (test split)             ')
        print(f"Test-Acc_base    = {self.pretrain_test_acc_lb_head_base}")
        print('========================================================')

    def test_ncd(self, args):
        # Test split evaluation
        #   |- Task-specific evaluation
        print('------>[Task-specific Head Test Evaluation]')
        #       |- labeled base classes
        print('------>           base classes')
        test_acc_lb_head_base = test_labeled_base(args, self.model, self.head_base, self.test_loader_base)
        #       |- unlabeled novel classes
        print('------>           novel classes')
        acc_ulb_head_novel = test_cluster(self.model, self.head_novel, self.test_loader_novel, args,
                                          return_ind=False)
        #   |- Task-agnostic evaluation
        print('------>[Task-agnostic Head Test Evaluation]')
        #       |- unlabeled novel classes
        self.concat_joint_head(args)
        print('------>           novel classes (on joint head)')
        acc_ulb_head_joint = test_ind_cluster(model=self.model, test_head=self.head_joint, ind_gen_head=self.head_novel,
                                              test_loader=self.test_loader_novel, args=args,
                                              ind_gen_loader=self.val_loader_novel)

        print('\n========================================================')
        print('             Stage-I: Pre-training             ')
        print(f"Acc_base_class (head_base)      = {self.pretrain_test_acc_lb_head_base}")

        print('\n             Stage-II: Novel Class Discovery             ')
        print(f"Acc_base_class  (head_base)     = {test_acc_lb_head_base}")
        print(f"\nAcc_novel_class (head_novel)    = {acc_ulb_head_novel}")
        print(f"Acc_novel_class (head_joint)    = {acc_ulb_head_joint}")
        print('========================================================')

    def save_backbone(self, path):
        torch.save(self.model.state_dict(), path)
        print("Learned Backbone saved to {}.".format(path))

    def save_head_base(self, path):
        torch.save(self.head_base.state_dict(), path)
        print("Learned Head_base saved to {}.".format(path))

    def save_head_novel(self, path):
        torch.save(self.head_novel.state_dict(), path)
        print("Learned Head_novel saved to {}.".format(path))

    def save_head_joint(self, path):
        torch.save(self.head_joint.state_dict(), path)
        print("Learned Head_joint saved to {}.".format(path))

# def test_base(args, test_loader, model, head):
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     model.eval()
#     head.eval()
#
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, _) in enumerate(tqdm(test_loader)):
#             inputs, targets = inputs.to(args.device), targets.to(args.device)
#             feat = model(inputs)
#             outputs = head(feat)
#             loss = F.cross_entropy(outputs, targets)
#             prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
#             losses.update(loss.item(), inputs.shape[0])
#             top1.update(prec1.item(), inputs.shape[0])
#             top5.update(prec5.item(), inputs.shape[0])
#     return top1.avg


# def test_cluster(args, test_loader, model, head, offset=0):
#     gt_targets = []
#     predictions = []
#     model.eval()
#     head.eval()
#
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, _) in enumerate(tqdm(test_loader)):
#             inputs, targets = inputs.to(args.device), targets.to(args.device)
#
#             feat = model(inputs)
#             outputs = head(feat)
#
#             _, max_idx = torch.max(outputs, dim=1)
#             predictions.extend(max_idx.cpu().numpy().tolist())
#             gt_targets.extend(targets.cpu().numpy().tolist())
#
#     predictions = np.array(predictions)
#     gt_targets = np.array(gt_targets)
#
#     predictions = torch.from_numpy(predictions)
#     gt_targets = torch.from_numpy(gt_targets)
#     eval_output = hungarian_evaluate(predictions, gt_targets, offset)
#
#     return eval_output


# @torch.no_grad()
# def hungarian_evaluate(predictions, targets, offset=0):
#     # Hungarian matching
#     targets = targets - offset
#     predictions = predictions - offset
#     predictions_np = predictions.numpy()
#     num_elems = targets.size(0)
#
#     # only consider the valid predicts. rest are treated as misclassification
#     valid_idx = np.where(predictions_np >= 0)[0]
#     predictions_sel = predictions[valid_idx]
#     targets_sel = targets[valid_idx]
#     num_classes = torch.unique(targets).numel()
#     num_classes_pred = torch.unique(predictions_sel).numel()
#
#     match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred,
#                              targets_k=num_classes)  # match is data dependent
#     reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
#     for pred_i, target_i in match:
#         reordered_preds[predictions_sel == int(pred_i)] = int(target_i)
#
#     # Gather performance metrics
#     reordered_preds = reordered_preds.numpy()
#     acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(
#         num_elems)  # accuracy is normalized with the total number of samples not only the valid ones
#     nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
#     ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())
#
#     return {'acc': acc * 100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}


# @torch.no_grad()
# def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
#     # Based on implementation from IIC
#     num_samples = flat_targets.shape[0]
#
#     num_k = preds_k
#     num_correct = np.zeros((num_k, num_k))
#
#     for c1 in range(num_k):
#         for c2 in range(num_k):
#             # elementwise, so each sample contributes once
#             votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
#             num_correct[c1, c2] = votes
#
#     # num_correct is small
#     match = linear_sum_assignment(num_samples - num_correct)
#     match = np.array(list(zip(*match)))
#
#     # return as list of tuples, out_c to gt_c
#     res = []
#     for out_c, gt_c in match:
#         res.append((out_c, gt_c))
#
#     return res

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


def test_ind_cluster(model, test_head, ind_gen_head, test_loader, args, ind_gen_loader=None):
    model.eval()
    test_head.eval()
    ind_gen_head.eval()

    # organize
    ncul = args.num_novel
    ncl = args.num_base

    # ================================
    # Index generation
    # ================================
    preds_ = np.array([])
    targets_ = np.array([])
    if ind_gen_loader is None:
        ind_gen_loader = test_loader

    for batch_idx_, (x_, label_, _) in enumerate(tqdm(ind_gen_loader)):
        x_, label_ = x_.to(args.device), label_.to(args.device)

        # forward inference
        feat_ = model(x_)
        output_ = ind_gen_head(feat_)

        _, pred_ = output_.max(1)
        targets_ = np.append(targets_, label_.cpu().numpy())
        preds_ = np.append(preds_, pred_.cpu().numpy())

    if args.dataset_name != 'cub200' and args.dataset_name != 'herb19':
        targets_ -= ncl
    _, ind = cluster_acc(targets_.astype(int), preds_.astype(int), True)

    # ================================
    # Test Evaluation
    # ================================
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = model(x)
        output = test_head(feat)

        # Joint head prediction
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    idx = np.argsort(ind[:, 1])
    id_map = ind[idx, 0]
    id_map += ncl

    targets_new = np.copy(targets)
    for i in range(ncul):
        targets_new[targets == i + ncl] = id_map[i]

    targets = targets_new
    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))

    print('Test w/o clustering: acc {:.4f}'.format(acc))
    return acc
