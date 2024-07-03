import torch
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .restune_vit import restune_block, vit_base


step_model_dict = {
    'encoder':  None,
    'head_mix': None,
    'head_res': None,
}

class LinearHeadCosNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.center = nn.Parameter(torch.Tensor(out_dim, out_dim))

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

    # @torch.no_grad()
    # def normalize_prototypes(self, s=1.0):
    #     # Cosine normalization
    #     w = self.last_layer.weight.data.clone()
    #     w = F.normalize(w, dim=1, p=2)
    #     self.last_layer.weight.copy_(w)

def build_restune_model(args):
    # Model data structure
    restune_models_dict = dict((f"step{s}", deepcopy(step_model_dict)) for s in range(args.num_steps))

    # init-step
    feat_dim = args.feat_dim
    path_model_state_dict = args.dino_pretrain_path

    # Base encoder
    vit16 = vit_base()
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
    vit16.load_state_dict(state_dict)
    # vit16 = vit16.to(args.device)
    #
    # for m in vit16.parameters():
    #     m.requires_grad = False

    # Template grown block
    template_block = restune_block()
    block_init_state_dict = template_block.state_dict()
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
    for k in block_init_state_dict.keys():
        if k == 'norm.weight' or k == 'norm.bias':
            block_init_state_dict[k] = deepcopy(state_dict[k])
        else:
            block_init_state_dict[k] = deepcopy(state_dict['blocks.11.'+k])

    # Init step-wise models LinearHeadCosNorm
    for s in range(args.num_steps):
        # initialize encoder
        if s == 0:
            restune_models_dict[f'step{s}']['encoder'] = vit16
            # print(f'--->>> {s}')
            # print(restune_models_dict[f'step{s}']['encoder'])
        else:
            new_block = restune_block()
            new_block_state_dict = deepcopy(block_init_state_dict)
            new_block.load_state_dict(new_block_state_dict)
            restune_models_dict[f'step{s}']['encoder'] = new_block
            # print(f'--->>> {s}')
            # print(restune_models_dict[f'step{s}']['encoder'])

        for m in restune_models_dict[f'step{s}']['encoder'].parameters():
            m.requires_grad = False

        # initialize head
        restune_models_dict[f'step{s}']['head_mix'] = LinearHeadCosNorm(in_dim=feat_dim,
                                                                        out_dim=args.num_novel_interval)

        restune_models_dict[f'step{s}']['head_res'] = LinearHeadCosNorm(in_dim=feat_dim,
                                                                        out_dim=args.num_novel_interval)

        restune_models_dict[f'step{s}']['encoder'] = restune_models_dict[f'step{s}']['encoder'].to(args.device)
        restune_models_dict[f'step{s}']['head_mix'] = restune_models_dict[f'step{s}']['head_mix'].to(args.device)
        restune_models_dict[f'step{s}']['head_res'] = restune_models_dict[f'step{s}']['head_res'] .to(args.device)

        # print(f'--->>> {s}')
        # print(restune_models_dict[f'step{s}']['encoder'])
        # print('\n')

    # print(restune_models_dict)
    return restune_models_dict
