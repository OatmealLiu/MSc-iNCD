import torch
from tqdm import tqdm
import numpy as np

class FeatureReplayer:
    def __init__(self, args, prev_pair_list, data_factory):
        self.num_kernels = args.current_novel_start
        self.num_interval = args.num_novel_interval

        self.nan_feat_idx_list = []
        self.nan_sig_idx_list = []

        # number of samples per kernel
        if args.dataset_name == 'cifar10':              # 6,000 imgs per class
            # 200 for S5, 80 for S2
            self.num_per_kernel = 200 #if args.num_steps >= 5 else 80
        elif args.dataset_name == 'cifar100':           # 600 imgs per class
            # 20 for S5, 8 for S2
            self.num_per_kernel = 20 if args.num_steps >= 5 else 8
        elif args.dataset_name == 'tinyimagenet':       # 500 imgs per class
            # 30 for S5, 12 for S2
            self.num_per_kernel = 30 if args.num_steps >= 5 else 12
        else:
            self.num_per_kernel = 30 if args.num_steps >= 5 else 12

        self.class_mean = torch.zeros(self.num_kernels, args.feat_dim).cuda()
        self.class_sig = torch.zeros(self.num_kernels, args.feat_dim).cuda()
        self.class_cov = 0
        self.generate_kernel(args, prev_pair_list, data_factory)

    def generate_kernel(self, args, prev_pair_list, data_factory):
        num_prev_steps = args.current_step

        all_feat = []
        all_labels = []

        print("Extract feature from previous samples and generate pseudo-labels")
        for step in range(num_prev_steps):
            this_prev_ulb_train_loader = data_factory.get_dataloader(
                split='train', aug=None, shuffle=True, target_list=range(step*args.num_novel_interval,
                                                                         (1+step)*args.num_novel_interval))

            for epoch in range(1):
                # switch model to eval mode
                prev_pair_list[step][0].eval()
                prev_pair_list[step][1].eval()

                for batch_idx, (x, _, idx) in enumerate(tqdm(this_prev_ulb_train_loader)):
                    x = x.to(args.device)
                    feat = prev_pair_list[step][0](x)
                    output = prev_pair_list[step][1](feat)
                    label = output.detach().max(1)[1] + step * args.num_novel_interval
                    all_feat.append(feat.detach().clone().to(args.device))
                    all_labels.append(label.detach().clone().to(args.device))

        # smooth the vector
        all_feat = torch.cat(all_feat, dim=0).to(args.device)
        all_labels = torch.cat(all_labels, dim=0).to(args.device)

        print("Calculate feature Mean-Var")
        for i in range(self.num_kernels):
            this_feat = all_feat[all_labels == i]
            this_mean = this_feat.mean(dim=0)
            this_var = this_feat.var(dim=0)

            # no samples are predicted as i-th class
            if torch.any(torch.isnan(this_mean)):
                self.nan_feat_idx_list.append(i)
            # only one sample is predicted as i-th class
            elif torch.any(torch.isnan(this_var)):
                self.nan_sig_idx_list.append(i)
                self.class_mean[i, :] = this_mean
                this_var = torch.zeros(this_var.shape).cuda()
                self.class_sig[i, :] = (this_var + 1e-5).sqrt()
            else: # everything is okay
                self.class_mean[i, :] = this_mean
                self.class_sig[i, :] = (this_var + 1e-5).sqrt()

        print("Finish kernel generation and initialization")
        self.class_mean = self.class_mean.cuda()
        self.class_sig = self.class_sig.cuda()
        self.class_cov = 0

        print(f"---> Feat NaN index list:")
        print(self.nan_feat_idx_list)
        print(f"---> Sig NaN index list:")
        print(self.nan_sig_idx_list)
        print(f"---> any NaN in class_mean: {torch.any(torch.isnan(self.class_mean))}")
        print(f"---> any NaN in class_sig: {torch.any(torch.isnan(self.class_sig))}")

    def replay_all(self):
        feats = []
        labels = []

        for k in range(self.num_kernels):
            if k in self.nan_feat_idx_list:
                # print(f"---> Do not have prototype for label {k}. Pass")
                continue
            k_dist = torch.distributions.Normal(self.class_mean[k], self.class_sig.mean(dim=0))
            k_feats = k_dist.sample((self.num_per_kernel,)).cuda()
            k_labels = torch.ones(k_feats.size(0)).cuda() * k

            feats.append(k_feats)
            labels.append(k_labels)

        feats = torch.cat(feats, dim=0).cuda()
        labels = torch.cat(labels, dim=0).long().cuda()
        return feats, labels
