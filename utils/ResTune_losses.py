import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroPaddingCrossEntropy(nn.Module):
    def __init__(self, temperature=0.1):
        super(ZeroPaddingCrossEntropy, self).__init__()
        self.T = temperature

    def forward(self, output, target):
        preds = F.softmax(output/self.T, dim=1)
        preds = torch.clamp(preds, min=1e-8)
        preds = torch.log(preds)
        loss = -torch.mean(torch.sum(target * preds, dim=1))
        return loss


class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, neg=True, batch=False):
        b = self.softmax(x) * self.logsoft(x)
        if batch:
            return -1.0 * b.sum(1)
        if neg:
            return -1.0 * b.sum()/x.size(0)
        else:
            return b.sum()/x.size(0)


def compute_sim_loss(feat_old, feat_new, sim_criterion, args):
    feat_old = F.normalize(feat_old, p=2, dim=1)
    feat_new = F.normalize(feat_new, p=2, dim=1)
    one_batch = torch.ones(feat_new.size(0)).to(args.device)
    sim_loss = sim_criterion(feat_old, feat_new, one_batch)
    return sim_loss