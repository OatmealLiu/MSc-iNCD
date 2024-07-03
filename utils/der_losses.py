import torch
import torch.nn as nn
import torch.nn.functional as F


class DERLoss(nn.Module):
    def __init__(self, alpha_der=0.3):
        super(DERLoss, self).__init__()
        self.alpha_der = alpha_der

    def forward(self, buf_output_new, buf_output_old):
        der_loss = self.alpha_der * F.mse_loss(buf_output_new, buf_output_old)
        return der_loss


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

