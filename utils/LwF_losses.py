import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, exp=1.0, size_average=True, eps=1e-5):
        super(CrossEntropyLoss, self).__init__()
        self.exp = exp
        self.size_average = size_average
        self.eps = eps

    def forward(self, output_old, output_new):
        output_old = F.softmax(output_old, dim=1)
        output_new = F.softmax(output_new, dim=1)

        if self.exp != 1:
            output_old = output_old.pow(self.exp)
            output_old = output_old / output_old.sum(1).view(-1, 1).expand_as(output_old)
            output_new = output_new.pow(self.exp)
            output_new = output_new / output_new.sum(1).view(-1, 1).expand_as(output_new)

        output_old = output_old + self.eps / output_old.size(1)
        output_old = output_old / output_old.sum(1).view(-1, 1).expand_as(output_old)
        ce = -(output_new * output_old.log()).sum(1)
        if self.size_average:
            ce = ce.mean()
        return ce


class LwFLoss(nn.Module):
    def __init__(self, w_kd=1.0, T=2.0):
        super(LwFLoss, self).__init__()
        self.w_kd = w_kd
        self.T = T
        self.loss = CrossEntropyLoss(exp=1.0/self.T)

    def forward(self, output_old, output_new):
        lwf_loss = self.w_kd * self.loss(output_old, output_new)
        return lwf_loss


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

