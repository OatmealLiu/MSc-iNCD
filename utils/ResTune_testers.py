from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc


def test_cluster(test_step, total_step, model_dict, test_loader, args, task_agnostic=False):
    # if step == 0:
    #     acc = test_cluster_init(model_dict, test_loader, args)
    # else:
    acc = test_cluster_il(test_step, total_step, model_dict, test_loader, args, task_agnostic=task_agnostic)
    return acc


def test_cluster_init(model_dict, test_loader, args):
    model_dict['step0']['encoder'].eval()
    model_dict['step0']['head_mix'].eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = model_dict['step0']['encoder'](x)
        output = model_dict['step0']['head_mix'](feat)

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    acc, ind = cluster_acc(targets.astype(int), preds.astype(int), True)
    nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)

    print('Test w/ clustering: acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc


def test_cluster_il(test_step, total_step, model_dict, test_loader, args, task_agnostic=False):
    for i in range(total_step+1):
        model_dict[f'step{i}']['encoder'].eval()
        model_dict[f'step{i}']['head_mix'].eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        output = []

        # Feature extraction
        # base feat
        shared_feat_v0 = model_dict['step0']['encoder'].get_intermediate_layers(x)

        # basic feat
        mix_feat_v0 = model_dict['step0']['encoder'](x)
        output_s = model_dict['step0']['head_mix'](mix_feat_v0)
        output.append(output_s)
        for i in range(1, total_step+1):
            mix_feat_v0 += model_dict[f'step{i}']['encoder'](shared_feat_v0)
            output_s = model_dict[f'step{i}']['head_mix'](mix_feat_v0)
            output.append(output_s)

        if task_agnostic:
            output = torch.cat(output, dim=1)
        else:
            output = output[test_step]

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    acc, ind = cluster_acc(targets.astype(int), preds.astype(int), True)
    nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)

    print('Test w/ clustering: acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc


def test_ind_cluster(test_step, total_step, model_dict, test_loader, ind_gen_loader, args):
    # if total_step == 0:
    #     acc = test_cluster_init(model_dict, test_loader, args)
    # else:
    acc = test_ind_cluster_il(test_step, total_step, model_dict, test_loader, ind_gen_loader, args)

    return acc


def test_ind_cluster_il(test_step, total_step, model_dict, test_loader, ind_gen_loader, args):
    for i in range(total_step+1):
        model_dict[f'step{i}']['encoder'].eval()
        model_dict[f'step{i}']['head_mix'].eval()

    # organize
    if test_step < args.num_steps - 1:
        this_num_novel = args.num_novel_interval
    else:
        this_num_novel = args.num_classes - test_step * args.num_novel_interval

    this_num_base = test_step * args.num_novel_interval

    # ================================
    # Index generation
    # ================================
    preds_ = np.array([])
    targets_ = np.array([])
    if ind_gen_loader is None:
        ind_gen_loader = test_loader

    for batch_idx_, (x_, label_, _) in enumerate(tqdm(ind_gen_loader)):
        x_, label_ = x_.to(args.device), label_.to(args.device)

        # Feature extraction
        # base feat
        shared_feat_v0_ = model_dict['step0']['encoder'].get_intermediate_layers(x_)

        # basic feat
        mix_feat_v0_ = model_dict['step0']['encoder'](x_)
        for i in range(1, test_step+1):
            mix_feat_v0_ += model_dict[f'step{i}']['encoder'](shared_feat_v0_)

        output_ = model_dict[f'step{test_step}']['head_mix'](mix_feat_v0_)

        _, pred_ = output_.max(1)
        targets_ = np.append(targets_, label_.cpu().numpy())
        preds_ = np.append(preds_, pred_.cpu().numpy())

    if args.dataset_name != 'cub200' and args.dataset_name != 'herb19':
        targets_ -= this_num_base
    # targets_ -= this_num_base
    _, ind = cluster_acc(targets_.astype(int), preds_.astype(int), True)

    # ================================
    # Test Evaluation
    # ================================
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        output = []
        # Feature extraction
        # base feat
        shared_feat_v0 = model_dict['step0']['encoder'].get_intermediate_layers(x)

        # basic feat
        mix_feat_v0 = model_dict['step0']['encoder'](x)
        output_s = model_dict['step0']['head_mix'](mix_feat_v0)
        output.append(output_s)

        for i in range(1, total_step+1):
            mix_feat_v0 += model_dict[f'step{i}']['encoder'](shared_feat_v0)
            output_s = model_dict[f'step{i}']['head_mix'](mix_feat_v0)
            output.append(output_s)

        output = torch.cat(output, dim=1)

        # Joint head prediction
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    if args.dataset_name == 'cub200' or args.dataset_name == 'herb19':
        targets += this_num_base

    idx = np.argsort(ind[:, 1])
    id_map = ind[idx, 0]
    id_map += this_num_base

    targets_new = np.copy(targets)
    for i in range(this_num_novel):
        targets_new[targets == i + this_num_base] = id_map[i]

    targets = targets_new
    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))

    print('Test w/o clustering: acc {:.4f}'.format(acc))
    return acc
