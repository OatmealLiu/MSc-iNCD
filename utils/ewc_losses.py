import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm


class EWCLoss(nn.Module):
    def __init__(self, args):
        super(EWCLoss, self).__init__()
        # number of prevous tasks
        self.prev_num_tasks = args.num_steps - 1

        self.lr = args.lr
        self.wd = args.weight_decay
        self.momentum = args.momentum
        self.device = args.device

        self.w_ewc = args.w_ewc               # w_ewc sets how important the old task is compared to the new one
        self.alpha = args.alpha_ewc  # alpha define how old and new fisher is fused, by default it is a 50-50 fusion
        # self.num_samples = -1

        # model, mean, fisher, for prev-tasks
        self.older_params = None
        # self.theta_mean = None
        self.fisher = None

    def compute_fisher_matrix_diag(self, model, head, train_loader):
        """
        compute fisher information matrix of task_id
        """
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in model.named_parameters() if p.requires_grad}

        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (len(train_loader.dataset) // train_loader.batch_size)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

        model.train()
        head.train()

        print("---> EWC ---> compute approximated Fisher information matrix by the train dataset")
        for batch_idx, (x, label, _) in enumerate(tqdm(train_loader)):
            x, label = x.to(self.device), label.to(self.device)

            feat = model(x)
            output = head(feat)

            preds = output.argmax(1).flatten()

            loss = F.cross_entropy(output, preds)
            optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # print('----->[DEBUG] update EwC fisher')
                    fisher[n] += p.grad.pow(2) * output.size(0)

        # Apply mean across all samples
        n_samples = n_samples_batches * train_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    def compute_theta_mean(self, old_params):
        """
        make a deepcopy of the old weights (i.e. 𝜃∗A), in the loss equation
        we need this to calculate (𝜃 - 𝜃∗A)^2 because self.params will be changing
        upon every backward pass and parameter update by the optimizer
        """
        curr_theta_mean = {n: p.to(self.device) for n, p in deepcopy(old_params).items()}
        return curr_theta_mean

    def update(self, task_id, model, head, train_loader):
        self.older_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        # self.theta_mean = self.compute_theta_mean(self.older_params)
        curr_fisher = self.compute_fisher_matrix_diag(model, head, train_loader)

        if task_id == 0:
            """initial step"""
            self.fisher = curr_fisher
            # print(self.fisher)
        else:
            for n in self.fisher.keys():
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n])
                # print(self.fisher[n])

    def get_ewc_penalty(self, flying_model):
        """compute EWC loss between the current model w.r.t. all previous models"""
        loss_reg = torch.tensor(0.0).to(self.device)
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in flying_model.named_parameters():
            if n in self.fisher.keys():
                loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2

        return loss_reg

    def forward(self, flying_model):
        ewc_loss = self.get_ewc_penalty(flying_model)
        ewc_loss = self.w_ewc * ewc_loss
        # print(f"------------> EWC-Loss={ewc_loss}")
        return ewc_loss

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

