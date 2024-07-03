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
import numpy as np
import os
import sys
import copy
import wandb

from methods.testers import test_cluster, test_ind_cluster

class TeacherProfessorStudent:
    def __init__(self, model, teachers_list, professors_list, student, feat_replayer, train_loader,
                 ulb_step_val_list, ulb_all_prev_val, ulb_all_val, ulb_step_test_list, ulb_all_prev_test, ulb_all_test):
        # Models
        self.model = model
        self.teachers_list = teachers_list
        self.professors_list = professors_list
        self.student = student

        # Feature Replayers
        self.feat_replayer = feat_replayer

        # Data loaders
        # |- train data for this step only
        self.train_loader = train_loader
        # |- val
        self.ulb_step_val_list = ulb_step_val_list
        self.ulb_all_prev_val = ulb_all_prev_val
        self.ulb_all_val = ulb_all_val
        # |- test
        self.ulb_step_test_list = ulb_step_test_list
        self.ulb_all_prev_test = ulb_all_prev_test
        self.ulb_all_test = ulb_all_test

    def calculate_kd(self, output_old, output_new, w_kd=10):
        softmax_output_old = F.softmax(output_old / 2, dim=1)
        log_softmax_output_new = F.log_softmax(output_new / 2, dim=1)
        loss_kd = -torch.mean(torch.sum(softmax_output_old * log_softmax_output_new, dim=1)) * w_kd
        return loss_kd

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

    def train_TeacherProfessor(self, args, epochs=100):
        """
        Adaption Step
        """
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am Teacher->Professor[{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters())
        for prof in self.professors_list:
            param_list += list(prof.parameters())

        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=args.lr * 1e-3)

        for epoch in range(int(epochs)):
            # create loss statistics recorder for each loss
            loss_record_list = [AverageMeter() for i in range(len(self.professors_list))]

            # switch the models to train mode
            self.model.train()
            for prof in self.professors_list:
                prof.train()
            for teacher in self.teachers_list:
                teacher.eval()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                loss_prof_list = []

                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                if args.l2_single_cls:
                    with torch.no_grad():
                        for prof in self.professors_list:
                            weight_temp = prof.last_layer.weight.data.clone()
                            weight_temp = F.normalize(weight_temp, dim=1, p=2)
                            prof.last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                # Single teacher head prediction
                output_v0_ = self.teachers_list[-1](feat_v0)
                output_v1_ = self.teachers_list[-1](feat_v1)

                # Create pseudo label for novel classes for this step by using the Teacher model
                target_v0 = output_v0_.detach().max(1)[1] + args.current_novel_start
                target_v1 = output_v1_.detach().max(1)[1] + args.current_novel_start

                # Current step
                output_v0 = self.professors_list[-1](feat_v0)
                output_v1 = self.professors_list[-1](feat_v1)

                mixed_output_current = torch.cat([output_v0, output_v1]).to(args.device)
                mixed_target_current = torch.cat([target_v0, target_v1]).to(args.device)
                mixed_target_current = torch.zeros(mixed_target_current.size(0), args.current_novel_end).to(
                    args.device).scatter_(1, mixed_target_current.view(-1, 1).long(), 1)

                current_loss = self.calculate_ce_zero_padding(mixed_output_current, mixed_target_current,
                                                              softmax_temp=args.softmax_temp)
                loss_prof_list.append(current_loss)
                loss_record_list[-1].update(current_loss.item(), mixed_output_current.size(0))

                # Previous step
                for i in range(len(self.professors_list[:-1])):
                    this_feat_replay, this_target_replay = self.feat_replayer.replay_step(step=i)

                    this_perm_idx = torch.randperm(this_feat_replay.size(0))
                    this_feat_replay = this_feat_replay[this_perm_idx].to(args.device)
                    this_target_replay = this_target_replay[this_perm_idx].to(args.device)

                    # professor head prediction
                    this_output_replay = self.professors_list[i](this_feat_replay)  # pred for replayed prototypes
                    this_target_replay = torch.zeros(this_target_replay.size(0), args.current_novel_end).to(
                        args.device).scatter_(1, this_target_replay.view(-1, 1).long(), 1)

                    # calculate loss
                    this_loss = self.calculate_ce_zero_padding(this_output_replay, this_target_replay,
                                                               softmax_temp=args.softmax_temp)
                    loss_prof_list.append(this_loss)

                    # record loss
                    loss_record_list[i].update(this_loss.item(), this_output_replay.size(0))

                optimizer.zero_grad()
                for loss in loss_prof_list:
                    loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            log_loss_prof_dict = dict(
                (f"loss/professor_s{i}", loss_record_list[i].avg) for i in range(len(loss_record_list)))
            wandb.log(log_loss_prof_dict, step=epoch)

            print('\n===========================================')
            for i in range(len(loss_record_list)):
                print('\nTrain Epoch [{}/{}]: Prof-S{} Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, i,
                                                                                loss_record_list[i].avg))
            print('===========================================')

            print('------> [Professor Test] Single Step Test W/ Clustering')
            acc_step_val_w_cluster_dict = dict(
                (f"Professor_step_val_acc_W_cluster/Prof{s}_step{s}_only", -1) for s in range(args.num_steps))
            for s in range(len(self.professors_list)):
                this_step_val_w = test_cluster(self.model, self.professors_list[s], self.ulb_step_val_list[s], args)
                acc_step_val_w_cluster_dict[f"Professor_step_val_acc_W_cluster/Prof{s}_step{s}_only"] = this_step_val_w

            wandb.log(acc_step_val_w_cluster_dict, step=epoch)

            print('------> [Professor Test] Single Step Test W/O Clustering')
            acc_step_val_wo_cluster_dict = dict(
                (f"Professor_step_val_acc_WO_cluster/Prof{s}_step{s}_only", -1) for s in range(args.num_steps))
            # for s in range(len(self.professors_list)):
            #     this_step_val_wo = test_ind_cluster(self.model, self.professors_list[s], self.teachers_list[s],
            #                                         self.ulb_step_val_list[s], s, args, ind_gen_loader=None)
            #     acc_step_val_wo_cluster_dict[f"Professor_step_val_acc_WO_cluster/Prof{s}_step{s}_only"] = this_step_val_wo

            wandb.log(acc_step_val_wo_cluster_dict, step=epoch)

            print('\n======================================')
            print('Step Single Val w/ clustering dict')
            print(acc_step_val_w_cluster_dict)

            print('\nStep Single Val w/o clustering dict')
            print(acc_step_val_wo_cluster_dict)
            print('======================================')
        # END: for epoch in range(args.epochs)

    def train_TeacherProfessorStudent(self, args):
        """
        Final training step
        """
        print("=" * 100)
        print(f"\t\t\t\t\tCiao bella! I am Teacher & Professor->Student [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
        print("=" * 100)

        # generate param list for optimizer
        param_list = list(self.model.parameters()) + list(self.student.parameters())
        optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

        criterion_ce = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            # create loss statistics recorder for each loss
            loss_teacher_recorder = AverageMeter()       # Loss with pseudo-label generated by teachers
            loss_prof_kd_recorder = AverageMeter()       # Loss by KD from Professors

            # switch the models to train mode
            self.model.train()
            self.student.train()

            for teacher in self.teachers_list:
                teacher.eval()

            for prof in self.professors_list:
                prof.eval()

            # update LR scheduler for the current epoch
            exp_lr_scheduler.step()

            for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(self.train_loader)):
                # send the vars to GPU
                x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

                # normalize classifier weights
                if args.l2_single_cls:
                    with torch.no_grad():
                        weight_temp = self.student.last_layer.weight.data.clone()
                        weight_temp = F.normalize(weight_temp, dim=1, p=2)
                        self.student.last_layer.weight.copy_(weight_temp)

                        for prof in self.professors_list:
                            weight_temp = prof.last_layer.weight.data.clone()
                            weight_temp = F.normalize(weight_temp, dim=1, p=2)
                            prof.last_layer.weight.copy_(weight_temp)

                # Feature extraction
                feat_v0 = self.model(x_v0)
                feat_v1 = self.model(x_v1)

                # Single head prediction
                output_v0 = self.teachers_list[-1](feat_v0)
                output_v1 = self.teachers_list[-1](feat_v1)

                # Create pseudo label for novel classes for this step by using the Teacher model
                target_v0 = output_v0.detach().max(1)[1] + args.current_novel_start
                target_v1 = output_v1.detach().max(1)[1] + args.current_novel_start

                # Replay features for previous step
                feat_replay, target_replay = self.feat_replayer.replay_all()

                mixed_feats = torch.cat([feat_v0, feat_v1, feat_replay], dim=0)
                mixed_targets = torch.cat([target_v0, target_v1, target_replay], dim=0)

                # shuffle all features
                idx_shuffle = torch.randperm(mixed_feats.size(0))
                mixed_feats, mixed_targets = mixed_feats[idx_shuffle], mixed_targets[idx_shuffle]

                output_student = self.student(mixed_feats)

                # Pseudo-loss
                if args.student_loss == 'CE':
                    # nomral Cross-Entropy loss w/o zero-padding
                    loss_teacher = criterion_ce(output_student, mixed_targets)
                else:
                    # Cross-Entropy loss w/ zero-padding
                    mixed_targets = torch.zeros(mixed_targets.size(0), args.current_novel_end).to(args.device).scatter_(
                        1, mixed_targets.view(-1, 1).long(), 1)
                    loss_teacher = self.calculate_ce_zero_padding(output_student, mixed_targets,
                                                                  softmax_temp=args.softmax_temp)

                # Professor KD loss
                # loss_prof_kd = torch.tensor(0.0).to(args.device)
                num_loss_prof_kd_smaples = 0

                # current step kd losses
                output_v0_student_current = self.student(feat_v0)
                output_v1_student_current = self.student(feat_v1)

                output_v0_prof_current = self.professors_list[-1](feat_v0)
                output_v1_prof_current = self.professors_list[-1](feat_v1)

                mixed_output_student_current = torch.cat([output_v0_student_current, output_v1_student_current], dim=0)
                mixed_output_prof_current = torch.cat([output_v0_prof_current, output_v1_prof_current], dim=0)

                # 1-st prof KD-loss then we accumulate it
                loss_prof_kd = self.calculate_kd(output_old=mixed_output_prof_current,
                                                 output_new=mixed_output_student_current,
                                                 w_kd=args.w_kd)
                num_loss_prof_kd_smaples += mixed_output_student_current.size(0)
                # print(loss_prof_kd)

                # previous step kd losses
                for i in range(len(self.professors_list[:-1])):
                    this_feat_replay, _ = self.feat_replayer.replay_step(step=i)

                    this_perm_idx = torch.randperm(this_feat_replay.size(0))
                    this_feat_replay = this_feat_replay[this_perm_idx].to(args.device)
                    # this_target_replay = this_target_replay[this_perm_idx].to(args.device)

                    # Two pred. distributions for the same feature
                    this_output_prof = self.professors_list[i](this_feat_replay)    # prof distribution
                    this_output_student = self.student(this_feat_replay)            # student distribution

                    # KW-loss calculation and accumulation
                    this_loss_prof_kd = self.calculate_kd(output_old=this_output_prof, output_new=this_output_student,
                                                          w_kd=args.w_kd)
                    loss_prof_kd += this_loss_prof_kd
                    num_loss_prof_kd_smaples += this_output_student.size(0)
                    # print(loss_prof_kd)

                # Total loss: Loss = loss_teacher + loss_prof_kd
                loss = loss_teacher + loss_prof_kd

                loss_teacher_recorder.update(loss_teacher.item(), output_student.size(0))
                loss_prof_kd_recorder.update(loss_prof_kd.item(), num_loss_prof_kd_smaples)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

            # wandb loss logging
            wandb.log({
                f"loss/teacher_{args.student_loss}": loss_teacher_recorder.avg,
                f"loss/professor_kd": loss_prof_kd_recorder.avg,
            }, step=epoch)

            print('\n===========================================')
            print('\nTrain Epoch [{}/{}]: Avg Loss: teacher_{}={:.4f}, professors_kd={:.4f}'.format(
                1 + epoch, args.epochs, args.student_loss, loss_teacher_recorder.avg, loss_prof_kd_recorder.avg))
            print('===========================================')

            self.save_student(path=args.save_student_path)

            print('------> Single Step Test W/ Clustering')
            acc_step_val_w_cluster_dict = dict(
                (f"step_val_acc_W_cluster/step{s}_only", -1) for s in range(args.num_steps))

            # Test all single steps
            # for s in range(1 + args.current_step):
            #     this_step_val_w = test_cluster(self.model, self.student, self.ulb_step_val_list[s], args,
            #                                    return_ind=False)
            #     acc_step_val_w_cluster_dict[f"step_val_acc_W_cluster/step{s}_only"] = this_step_val_w


            # Test only this step
            this_step_val_w = test_cluster(self.model, self.student, self.ulb_step_val_list[args.current_step], args,
                                           return_ind=False)
            acc_step_val_w_cluster_dict[f"step_val_acc_W_cluster/step{args.current_step}_only"] = this_step_val_w

            wandb.log(acc_step_val_w_cluster_dict, step=epoch)

            print('------> Single Step Test W/O Clustering')
            acc_step_val_wo_cluster_dict = dict(
                (f"step_val_acc_WO_cluster/step{s}_only", -1) for s in range(args.num_steps))

            # Test all single steps
            # for s in range(1 + args.current_step):
            #     this_step_val_wo = test_ind_cluster(self.model, self.student, self.teachers_list[s],
            #                                         self.ulb_step_val_list[s], s, args, ind_gen_loader=None)
            #     acc_step_val_wo_cluster_dict[f"step_val_acc_WO_cluster/step{s}_only"] = this_step_val_wo

            # Test only this step
            # this_step_val_wo = test_ind_cluster(self.model, self.student, self.teachers_list[args.current_step],
            #                                     self.ulb_step_val_list[args.current_step], args.current_step,
            #                                     args, ind_gen_loader=None)
            # acc_step_val_wo_cluster_dict[f"step_val_acc_WO_cluster/step{args.current_step}_only"] = this_step_val_wo

            wandb.log(acc_step_val_wo_cluster_dict, step=epoch)

            print('------> All-Prev-Steps Test W/ Clustering')
            acc_all_prev_val_w_cluster = test_cluster(self.model, self.student, self.ulb_all_prev_val, args)

            print('------> All-Steps Test W/ Clustering')
            acc_all_val_w_cluster = test_cluster(self.model, self.student, self.ulb_all_val, args)

            # print('------> All (all/prev) Steps Test W/O Clustering')
            # step_acc_wo_cluster_list = [acc_step_val_wo_cluster_dict[f"step_val_acc_WO_cluster/step{s}_only"]
            #                             for s in range(1+args.current_step)]
            # acc_all_prev_val_wo_cluster = self.calculate_weighted_avg(step_acc_wo_cluster_list[:-1], args)
            # acc_all_val_wo_cluster = self.calculate_weighted_avg(step_acc_wo_cluster_list, args)

            # wandb metrics logging
            wandb.log({
                "all_val_acc/all_prev_W_cluster": acc_all_prev_val_w_cluster,
                # "all_val_acc/all_prev_WO_cluster": acc_all_prev_val_wo_cluster,
                "all_val_acc/all_W_cluster": acc_all_val_w_cluster,
                # "all_val_acc/all_WO_cluster": acc_all_val_wo_cluster,
            }, step=epoch)

            print('\n======================================')
            print('All-Previous-Discovered')
            print(f"Acc_all_prev_W_cluster    = {acc_all_prev_val_w_cluster}")
            # print(f"Acc_all_prev_WO_cluster   = {acc_all_prev_val_wo_cluster}")

            print('\nAll-Discovered')
            print(f"Acc_all_W_cluster         = {acc_all_val_w_cluster}")
            # print(f"Acc_all_WO_cluster        = {acc_all_val_wo_cluster}")

            print('\nSingle-Discovered')
            print('Step Single Val w/ clustering dict')
            print(acc_step_val_w_cluster_dict)

            print('Step Single Val w/o clustering dict')
            print(acc_step_val_wo_cluster_dict)
            print('======================================')

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

    def save_professors(self, path_list):
        for s in range(len(self.professors_list)):
            torch.save(self.professors_list[s].state_dict(), path_list[s])
            print("Professor Head S-{} saved to {}".format(s, path_list[s]))

    def save_student(self, path):
        torch.save(self.student.state_dict(), path)
        print("Student Head saved to {}.".format(path))

    def return_professors(self):
        return self.professors_list

    def return_student(self):
        return self.student



