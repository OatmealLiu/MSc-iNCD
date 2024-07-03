import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.last_layer(x)
        return x

class LinearHeadFeatNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class LinearHeadCosNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    @torch.no_grad()
    def normalize_prototypes(self, s=1.0):
        # Cosine normalization
        w = self.last_layer.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.last_layer.weight.copy_(w)


class LinearHeadScaledCosNorm(nn.Module):
    def __init__(self, in_dim, out_dim, init_scale=1.0):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.scale_layer = nn.Parameter(torch.FloatTensor([init_scale]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        x = x * self.scale_layer
        return x

    @torch.no_grad()
    def normalize_prototypes(self):
        # Cosine normalization
        w = self.last_layer.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.last_layer.weight.copy_(w)

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
       return x * self.scale

class LinearHeadMultiManifoldCosNorm(nn.Module):
    def __init__(self, manifolds, out_dim):
        super().__init__()
        if len(manifolds) != 3:
            print("Pls use 3 manifolds")
            raise NotImplementedError

        # Manifold 1 e.g. 768
        self.dim_manifold1 = manifolds[0]
        self.last_layer1 = nn.Linear(self.dim_manifold1, out_dim, bias=False)
        # Manifold 2 e.g. 512
        self.dim_manifold2 = manifolds[1]
        self.last_layer2 = nn.Linear(self.dim_manifold2, out_dim, bias=False)
        # Manifold 3 e.g. 256
        self.dim_manifold3 = manifolds[2]
        self.last_layer3 = nn.Linear(self.dim_manifold3, out_dim, bias=False)

    def forward(self, x):
        # full-dim feature
        x = F.normalize(x, dim=-1, p=2)
        out1 = self.last_layer1(x)  # manifold1 e.g. 768
        out2 = self.last_layer2(x[:, :self.dim_manifold2])
        out3 = self.last_layer3(x[:, :self.dim_manifold3])
        return out1, out2, out3

    @torch.no_grad()
    def normalize_prototypes(self, s=1.0):
        # Apply Weights Normalization on to different manifolds linear heads
        # |- Manifold 1
        w1 = self.last_layer1.weight.data.clone()
        w1 = s * F.normalize(w1, dim=1, p=2)
        self.last_layer1.weight.copy_(w1)
        # |- Manifold 2
        w2 = self.last_layer2.weight.data.clone()
        w2 = s * F.normalize(w2, dim=1, p=2)
        self.last_layer2.weight.copy_(w2)
        # |- Manifold 3
        w3 = self.last_layer3.weight.data.clone()
        w3 = s * F.normalize(w3, dim=1, p=2)
        self.last_layer3.weight.copy_(w3)

class LinearHeadMultiManifoldCosNormRank(nn.Module):
    def __init__(self, manifolds, out_dim):
        super().__init__()
        if len(manifolds) != 3:
            print("Pls use 3 manifolds")
            raise NotImplementedError

        # Manifold 1 e.g. 768
        self.dim_manifold1 = manifolds[0]
        self.last_layer1 = nn.Linear(self.dim_manifold1, out_dim, bias=False)
        # Manifold 2 e.g. 512
        self.dim_manifold2 = manifolds[1]
        self.last_layer2 = nn.Linear(self.dim_manifold2, out_dim, bias=False)
        # Manifold 3 e.g. 256
        self.dim_manifold3 = manifolds[2]
        self.last_layer3 = nn.Linear(self.dim_manifold3, out_dim, bias=False)

    def forward(self, x):
        # full-dim feature
        x = F.normalize(x, dim=-1, p=2)
        out1 = self.last_layer1(x)  # manifold1 e.g. 768

        # sort feature
        idx = torch.argsort(x, dim=1, descending=True)

        # slice to manifold2
        out2 = self.last_layer2(torch.cat([x[i, idx[i]].view((1, x.size(1))) for i in range(x.size(0))], dim=0)[:, :self.dim_manifold2])  # manifold2
        # slice to manifold3
        out3 = self.last_layer3(torch.cat([x[i, idx[i]].view((1, x.size(1))) for i in range(x.size(0))], dim=0)[:, :self.dim_manifold3])  # manifold3

        return out1, out2, out3

    @torch.no_grad()
    def normalize_prototypes(self, s=1.0):
        # Apply Weights Normalization on to different manifolds linear heads
        # |- Manifold 1
        w1 = self.last_layer1.weight.data.clone()
        w1 = s * F.normalize(w1, dim=1, p=2)
        self.last_layer1.weight.copy_(w1)
        # |- Manifold 2
        w2 = self.last_layer2.weight.data.clone()
        w2 = s * F.normalize(w2, dim=1, p=2)
        self.last_layer2.weight.copy_(w2)
        # |- Manifold 3
        w3 = self.last_layer3.weight.data.clone()
        w3 = s * F.normalize(w3, dim=1, p=2)
        self.last_layer3.weight.copy_(w3)

