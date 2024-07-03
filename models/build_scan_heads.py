import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class HeadWnorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=2.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class HeadWOnorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=2.1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x