from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc

def test_ind_cluster_debug(model, test_head, ind_gen_head, test_loader, step, args, ind_gen_loader):
    model.eval()
    test_head.eval()
    ind_gen_head.eval()

    # ================================
    # Index generation
    # ================================
    preds_ = np.array([])
    targets_ = np.array([])

    for batch_idx_, (x_, label_, _) in enumerate(tqdm(ind_gen_loader)):
        x_, label_ = x_.to(args.device), label_.to(args.device)

        # forward inference
        feat_ = model(x_)
        output_ = ind_gen_head(feat_)

        _, pred_ = output_.max(1)
        targets_ = np.append(targets_, label_.cpu().numpy())
        preds_ = np.append(preds_, pred_.cpu().numpy())

    print(f"Single Head Ind Gen: len(targets_) = {len(targets_)}")
    print("Val. targets_[:20]")
    print(targets_[:20])
    print("Single Head Val preds_[:20]")
    print(preds_[:20])

    targets_ -= args.current_novel_start
    acc_, ind = cluster_acc(targets_.astype(int), preds_.astype(int), True)

    print(f"Single Head acc_w_cluster_val = {acc_}")

    # ================================
    # Test Evaluation
    # ================================
    preds_single_test = np.array([])
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = model(x)
        output = test_head(feat)
        output_single = ind_gen_head(feat)

        # Joint head prediction
        _, pred = output.max(1)
        _, pred_single = output_single.max(1)

        targets = np.append(targets, label.cpu().numpy())

        preds = np.append(preds, pred.cpu().numpy())
        preds_single_test = np.append(preds_single_test, pred_single.cpu().numpy())

    print(len(targets))
    print(targets[:20])

    # organize
    this_num_novel = args.num_novel_interval if int(1 + step) < args.num_steps else args.num_novel_per_step
    this_num_base = args.current_novel_start

    # ind = ind[:, 1] + this_num_base
    # ind = ind[:this_num_novel, :]
    idx = np.argsort(ind[:, 1])
    id_map = ind[idx, 0]
    id_map += this_num_base

    # idx_ = np.argsort(ind[:, 1])
    # ind = ind[idx_]
    # ind = ind[this_num_base:this_num_base+this_num_novel]
    # # ind = ind[:this_num_novel, :]
    # idx = np.argsort(ind[:, 1])
    # id_map = ind[idx, 0]
    # id_map += this_num_base

    targets_new = np.copy(targets)
    for i in range(args.num_novel_per_step):
        targets_new[targets == i + this_num_base] = id_map[i]

    targets = targets_new

    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)

    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))


    print("Target [:20]")
    print(targets[:20])

    print("Correct")
    print(correct)

    print("Single [:20]")
    print(preds_single_test[:20]+this_num_base)

    print("Joint [:20]")
    print(preds[:20])

    print("Diff")
    print((torch.from_numpy(preds_single_test[:20]) + this_num_base) - preds[:20])

    correlation_single_joint = preds.eq(torch.from_numpy(preds_single_test+this_num_base)).float().sum(0)
    print(f"Single-Joint Correlation = {correlation_single_joint}")
    print('Test w/o clustering: acc {:.4f}'.format(acc))
    return acc


def test_single(model, teacher, test_loader, args, cluster=True, ind=None, return_ind=False):
    model.eval()
    teacher.eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = model(x)
        output = teacher(feat)

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    if cluster:
        if return_ind:
            acc, ind = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        else:
            acc = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    else:
        raise NotImplementedError

    if return_ind:
        return acc, ind
    else:
        return acc

def test_cluster(model, test_head, test_loader, args, return_ind=False):
    model.eval()
    test_head.eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)
        # print(len(x[0]))
        # print(len(x[1]))
        # print(x[0].shape)
        # print(x[1].shape)
        # x = x.to(args.device)
        # label = label.to(args.device)

        # forward inference
        feat = model(x)
        output = test_head(feat)

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    acc, ind = cluster_acc(targets.astype(int), preds.astype(int), True)
    nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)

    if return_ind:
        return acc, ind
    else:
        print('Test w/ clustering: acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
        return acc

# def test_ind_cluster(model, test_head, ind_gen_head, test_loader, step, args, ind_gen_loader=None):
#     model.eval()
#     test_head.eval()
#     ind_gen_head.eval()
#
#     preds = np.array([])
#     targets = np.array([])
#
#     for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
#         x, label = x.to(args.device), label.to(args.device)
#
#         # forward inference
#         feat = model(x)
#         output = test_head(feat)
#
#         # Joint head prediction
#         _, pred = output.max(1)
#         targets = np.append(targets, label.cpu().numpy())
#         preds = np.append(preds, pred.cpu().numpy())
#
#     # index generation
#     if ind_gen_loader:
#         _, ind = test_cluster(model, ind_gen_head, ind_gen_loader, args, return_ind=True)
#     else:
#         _, ind = test_cluster(model, ind_gen_head, test_loader, args, return_ind=True)
#
#     # organize
#     this_num_novel = args.num_novel_interval if int(1 + step) < args.num_steps else args.num_novel_per_step
#     this_num_base = args.num_novel_interval * step
#
#     ind = ind[:this_num_novel, :]
#     idx = np.argsort(ind[:, 1])
#     id_map = ind[idx, 0]
#     id_map += this_num_base
#
#     targets_new = np.copy(targets)
#     for i in range(args.num_novel_per_step):
#         targets_new[targets == i + this_num_base] = id_map[i]
#
#     targets = targets_new
#
#     preds = torch.from_numpy(preds)
#     targets = torch.from_numpy(targets)
#
#     correct = preds.eq(targets).float().sum(0)
#     acc = float(correct / targets.size(0))
#
#     print('Test w/o clustering: acc {:.4f}'.format(acc))
#     return acc


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
        output_ = ind_gen_head(feat_)

        _, pred_ = output_.max(1)
        targets_ = np.append(targets_, label_.cpu().numpy())
        preds_ = np.append(preds_, pred_.cpu().numpy())

    # print('\n\n')
    # print(targets_)
    if args.dataset_name != 'cub200' and args.dataset_name != 'herb19':
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
        output = test_head(feat)

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
    for i in range(args.num_novel_per_step):
        targets_new[targets == i + this_num_base] = id_map[i]

    targets = targets_new
    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))

    print('Test w/o clustering: acc {:.4f}'.format(acc))
    return acc


def test_ind_cluster_unlocked(test_model, test_head, ind_gen_model, ind_gen_head, test_loader, step, args,
                              ind_gen_loader=None):
    test_model.eval()
    test_head.eval()
    ind_gen_model.eval()
    ind_gen_head.eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

        # forward inference
        feat = test_model(x)
        output = test_head(feat)

        # Joint head prediction
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    # index generation
    if ind_gen_loader:
        _, ind = test_cluster(ind_gen_model, ind_gen_head, ind_gen_loader, args, return_ind=True)
    else:
        _, ind = test_cluster(ind_gen_model, ind_gen_head, test_loader, args, return_ind=True)

    # organize
    this_num_novel = args.num_novel_interval if int(1 + step) < args.num_steps else args.num_novel_per_step
    this_num_base = args.num_novel_interval * step

    ind = ind[:this_num_novel, :]
    idx = np.argsort(ind[:, 1])
    id_map = ind[idx, 0]
    id_map += this_num_base

    if args.dataset_name == 'cub200' or args.dataset_name == 'herb19':
        targets += this_num_base

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

def test_labeled_base(args, model, head, test_loader):
    model.eval()
    head.eval()

    preds = np.array([])
    targets = np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)
        feat = model(x)
        output = head(feat)

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))
    print('Labeled-Test acc {:.4f}'.format(acc))

    return acc