import torch
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.util import cluster_acc
import numpy as np
from tqdm import tqdm

def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def init_prob_kmeans(encoder, eval_loader, args):
    torch.manual_seed(args.seed)
    # model = model.to(args.device)
    # cluster parameter initiate
    encoder.eval()
    targets = np.zeros(len(eval_loader.dataset))
    feats = np.zeros((len(eval_loader.dataset), 768))
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(eval_loader)):
            x = x.to(args.device)
            feat = encoder(x)
            idx = idx.data.cpu().numpy()
            feats[idx, :] = feat.data.cpu().numpy()
            targets[idx] = label.data.cpu().numpy()

    # evaluate clustering performance
    pca = PCA(n_components=args.num_novel_interval)
    feats = pca.fit_transform(feats)
    kmeans = KMeans(n_clusters=args.num_novel_interval)
    y_pred = kmeans.fit_predict(feats)

    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs