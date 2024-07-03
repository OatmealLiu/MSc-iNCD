from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc

def test_cluster(model, test_head, test_loader, args, return_ind=False):
    model.eval()
    test_head.eval()

    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)

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


def test_ind_cluster(model, test_head, ind_gen_head, test_loader, step, args, ind_gen_loader=None):
    model.eval()
    test_head.eval()
    ind_gen_head.eval()

    # organize
    if step < args.num_steps - 1:
        this_num_novel = args.num_novel_interval
    else:
        this_num_novel = args.num_classes - step * args.num_novel_interval

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
    for i in range(this_num_novel):
        targets_new[targets == i + this_num_base] = id_map[i]

    targets = targets_new
    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    acc = float(correct / targets.size(0))

    print('Test w/o clustering: acc {:.4f}'.format(acc))
    return acc
