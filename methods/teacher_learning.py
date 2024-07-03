import torch
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import AverageMeter
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb
from methods.testers import test_single

def train_Teacher(model, teacher, sinkhorn, train_loader, ulb_val_loader, args):
    print("=" * 100)
    print(f"\t\t\t\t\tCiao bella! I am Teacher [{1 + args.current_step}/{args.num_steps}] for MSc-iNCD")
    print("=" * 100)

    # generate param list for optimizer
    param_list = list(model.parameters()) + list(teacher.parameters())

    optimizer = SGD(param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    for epoch in range(args.epochs):
        # create loss statistics recorder for each loss
        loss_uno_record = AverageMeter()  # UNO loss recorder

        # switch the models to train mode
        model.train()
        teacher.train()

        # update LR scheduler for the current epoch
        exp_lr_scheduler.step()

        for batch_idx, ((x_v0, x_v1), _, idx) in enumerate(tqdm(train_loader)):
            # send the vars to GPU
            x_v0, x_v1 = x_v0.to(args.device), x_v1.to(args.device)

            # normalize classifier weights
            if args.l2_single_cls:
                with torch.no_grad():
                    weight_temp = teacher.last_layer.weight.data.clone()
                    weight_temp = F.normalize(weight_temp, dim=1, p=2)
                    teacher.last_layer.weight.copy_(weight_temp)

            # Feature extraction
            feat_v0 = model(x_v0)
            feat_v1 = model(x_v1)

            # Single head prediction
            output_v0 = teacher(feat_v0)
            output_v1 = teacher(feat_v1)

            # cross pseudo-labeling
            target_v0 = sinkhorn(output_v1)
            target_v1 = sinkhorn(output_v0)

            mixed_logits = torch.cat([output_v0, output_v1], dim=0)
            mixed_targets = torch.cat([target_v0, target_v1], dim=0)

            # UNO Loss Calculation
            # follow original UNO, temperature = 0.1
            preds = F.softmax(mixed_logits / args.softmax_temp, dim=1)  # temperature
            preds = torch.clamp(preds, min=1e-8)
            preds = torch.log(preds)
            loss_uno = -torch.mean(torch.sum(mixed_targets * preds, dim=1))

            loss_uno_record.update(loss_uno.item(), output_v0.size(0))

            optimizer.zero_grad()
            loss_uno.backward()
            optimizer.step()
            # END: for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):

        # wandb loss logging
        wandb.log({
            "loss/uno": loss_uno_record.avg,
        }, step=epoch)

        print('\n===========================================')
        print('\nTrain Epoch [{}/{}]: Avg Loss: {:.4f}'.format(1 + epoch, args.epochs, loss_uno_record.avg))
        print('===========================================')

        print('Task-specific Head: test on unlabeled classes for this step only')
        args.head = 'head2'
        acc_val_w_clustering = test_single(model, teacher, ulb_val_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/ulb_val_w_clustering": acc_val_w_clustering,
        }, step=epoch)

    return teacher
    # END: for epoch in range(args.epochs)

