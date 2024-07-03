import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter
import numpy as np
import wandb

from tqdm import tqdm
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc


def test_cluster(model, test_head, test_loader, args, return_ind=False):
    model.eval()
    test_head.eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = model(x)
        output1, output2, output3 = test_head(feat)

        if args.pred_method == 'average':
            # Weighted average among 3 manifolds
            output = (output1*3 + output2*2 + output1)/6
            _, pred = output.max(1)
        else:
            # Voting among 3 manifolds
            conf1, pred1 = output1.max(1)
            conf2, pred2 = output2.max(1)
            conf3, pred3 = output3.max(1)
            conf1, pred1 = conf1.reshape((1, conf1.size(0))), pred1.reshape((1, pred1.size(0)))
            conf2, pred2 = conf2.reshape((1, conf2.size(0))), pred2.reshape((1, pred2.size(0)))
            conf3, pred3 = conf3.reshape((1, conf3.size(0))), pred3.reshape((1, pred3.size(0)))

            conf = torch.cat([conf1, conf2, conf3], dim=0)
            pred = torch.cat([pred1, pred2, pred3], dim=0)
            _, voting_idx = conf.max(0)
            pred = pred[voting_idx, list(range(output1.size(0)))]

        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    acc, ind = cluster_acc(targets.astype(int), preds.astype(int), True)
    nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)

    if return_ind:
        return acc, ind
    else:
        print('Test w/ clustering: acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
        return acc


def test_ind_cluster(model, test_head, ind_gen_head, test_loader, step, args, ind_gen_loader=None):
    model.eval()
    test_head.eval()
    ind_gen_head.eval()

    # organize
    this_num_novel = args.num_novel_interval if int(1 + step) < args.num_steps else args.num_novel_per_step
    this_num_base = step * args.num_novel_interval

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
        output1_, output2_, output3_ = ind_gen_head(feat_)

        if args.pred_method == 'average':
            # Weighted average among 3 manifolds
            output_ = (output1_*3 + output2_*2 + output1_)/6
            _, pred_ = output_.max(1)
        else:
            # Voting among 3 manifolds
            conf1_, pred1_ = output1_.max(1)
            conf2_, pred2_ = output2_.max(1)
            conf3_, pred3_ = output3_.max(1)
            conf1_, pred1_ = conf1_.reshape((1, conf1_.size(0))), pred1_.reshape((1, pred1_.size(0)))
            conf2_, pred2_ = conf2_.reshape((1, conf2_.size(0))), pred2_.reshape((1, pred2_.size(0)))
            conf3_, pred3_ = conf3_.reshape((1, conf3_.size(0))), pred3_.reshape((1, pred3_.size(0)))

            conf_ = torch.cat([conf1_, conf2_, conf3_], dim=0)
            pred_ = torch.cat([pred1_, pred2_, pred3_], dim=0)
            _, voting_idx_ = conf_.max(0)
            pred_ = pred_[voting_idx_, list(range(output1_.size(0)))]

        targets_ = np.append(targets_, label_.cpu().numpy())
        preds_ = np.append(preds_, pred_.cpu().numpy())

    targets_ -= this_num_base
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
        output1, output2, output3 = test_head(feat)

        if args.pred_method == 'average':
            # Weighted average among 3 manifolds
            output = (output1*3 + output2*2 + output1)/6
            _, pred = output.max(1)
        else:
            # Voting among 3 manifolds
            conf1, pred1 = output1.max(1)
            conf2, pred2 = output2.max(1)
            conf3, pred3 = output3.max(1)
            conf1, pred1 = conf1.reshape((1, conf1.size(0))), pred1.reshape((1, pred1.size(0)))
            conf2, pred2 = conf2.reshape((1, conf2.size(0))), pred2.reshape((1, pred2.size(0)))
            conf3, pred3 = conf3.reshape((1, conf3.size(0))), pred3.reshape((1, pred3.size(0)))

            conf = torch.cat([conf1, conf2, conf3], dim=0)
            pred = torch.cat([pred1, pred2, pred3], dim=0)
            _, voting_idx = conf.max(0)
            pred = pred[voting_idx, list(range(output1.size(0)))]

        # Joint head prediction
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    idx = np.argsort(ind[:, 1])
    id_map = ind[idx, 0]
    id_map += this_num_base

    targets_new = np.copy(targets)
    for i in range(args.num_novel_per_step):
        targets_new[targets == i + this_num_base] = id_map[i]

    targets = targets_new
    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))

    print('Test w/o clustering: acc {:.4f}'.format(acc))
    return acc


class ManifoldHopper:
    def __init__(self, model, single_head, learned_single_heads, joint_head, sinkhorn,
                 train_loader, ulb_step_val_list, ulb_all_prev_val, ulb_all_val,
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
            w1_saved = self.learned_single_heads[step].last_layer1.weight.data.clone()
            w2_saved = self.learned_single_heads[step].last_layer2.weight.data.clone()
            w3_saved = self.learned_single_heads[step].last_layer3.weight.data.clone()

            self.joint_head.last_layer1.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w1_saved)
            self.joint_head.last_layer2.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w2_saved)
            self.joint_head.last_layer3.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval].copy_(w3_saved)

        current_w1 = self.single_head.last_layer1.weight.data.clone()
        current_w2 = self.single_head.last_layer2.weight.data.clone()
        current_w3 = self.single_head.last_layer3.weight.data.clone()

        self.joint_head.last_layer1.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w1)
        self.joint_head.last_layer2.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w2)
        self.joint_head.last_layer3.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w3)

    def train(self, args):
        print("=" * 100)
        print(f"\t\t\t\t\tManifold Hopper Method: [{1 + args.current_step}/{args.num_steps}] >_<")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.single_head.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        # Running best acc. eval. on val. dataset
        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_uno_recorder1 = AverageMeter()          # UNO loss recording
            loss_uno_recorder2 = AverageMeter()          # UNO loss recording
            loss_uno_recorder3 = AverageMeter()          # UNO loss recording

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
                output1_v0, output2_v0, output3_v0 = self.single_head(feat_v0)
                output1_v1, output2_v1, output3_v1 = self.single_head(feat_v1)

                # cross pseudo-labeling
                target1_v0 = self.sinkhorn(output1_v1)
                target1_v1 = self.sinkhorn(output1_v0)

                target2_v0 = self.sinkhorn(output2_v1)
                target2_v1 = self.sinkhorn(output2_v0)

                target3_v0 = self.sinkhorn(output3_v1)
                target3_v1 = self.sinkhorn(output3_v0)

                mixed_logits1 = torch.cat([output1_v0, output1_v1], dim=0)
                mixed_targets1 = torch.cat([target1_v0, target1_v1], dim=0)

                mixed_logits2 = torch.cat([output2_v0, output2_v1], dim=0)
                mixed_targets2 = torch.cat([target2_v0, target2_v1], dim=0)

                mixed_logits3 = torch.cat([output3_v0, output3_v1], dim=0)
                mixed_targets3 = torch.cat([target3_v0, target3_v1], dim=0)

                # UNO Loss
                loss_uno1 = self.calculate_ce_zero_padding(mixed_logits1, mixed_targets1, softmax_temp=args.softmax_temp)
                loss_uno2 = self.calculate_ce_zero_padding(mixed_logits2, mixed_targets2, softmax_temp=args.softmax_temp)
                loss_uno3 = self.calculate_ce_zero_padding(mixed_logits3, mixed_targets3, softmax_temp=args.softmax_temp)

                loss_uno_recorder1.update(loss_uno1.item(), output1_v0.size(0))
                loss_uno_recorder2.update(loss_uno2.item(), output2_v0.size(0))
                loss_uno_recorder3.update(loss_uno3.item(), output3_v0.size(0))

                optimizer.zero_grad()
                loss_uno1.backward()
                loss_uno2.backward()
                loss_uno3.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/loss_1": loss_uno_recorder1.avg,
                f"loss/loss_2": loss_uno_recorder2.avg,
                f"loss/loss_3": loss_uno_recorder3.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f} {:.4f} {:.4f}'.format(1 + epoch, args.epochs,
                                                                                 loss_uno_recorder1.avg,
                                                                                 loss_uno_recorder2.avg,
                                                                                 loss_uno_recorder3.avg
                                                                                 ))
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

        self.learned_single_heads.append(self.single_head)
        print(
            "[Single head training completed]: extended the learned single heads list by the newly learned single head")

    def test(self, args):
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

