import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.lines as mline
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from tqdm import tqdm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class Pencil:
    def __init__(self, args):
        self.device = args.device
        self.num_steps = args.num_steps
        self.current_step = args.current_step
        self.current_novel_interval = args.num_novel_interval
        self.current_novel_start = args.current_novel_start
        self.current_novel_end = args.current_novel_end

        self.dataset_name = args.dataset_name
        self.num_classes = args.num_classes
        self.output_dir = args.plot_output_dir

        self.step_cmap = ['g','b','r','c']
        # if self.dataset_name == 'cifar10':
        #     self.step_cmap = ListedColormap(["darkorange", "gold"])
        # elif self.dataset_name == 'cifar100':
        #     self.step_cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagree"])
        # elif self.dataset_name == 'tinyimagenet':
        #     self.step_cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagree"])
        # else:
        #     N = args.num_steps
        #     vals = np.ones((N, 4))
        #     vals[:, 0] = np.linspace(90 / 256, 1, N)
        #     vals[:, 1] = np.linspace(60 / 256, 1, N)
        #     vals[:, 2] = np.linspace(30 / 256, 1, N)
        #     self.step_cmap = ListedColormap(vals)

    def plot_weights(self, weight, file_name='noname_weight.png'):
        l2_weight = torch.linalg.norm(weight, dim=1).cpu().numpy()

        step_weight_list = []
        step_x_range_list = []
        for step in range(1+self.current_step):
            # slice weights
            step_weight = l2_weight[step*self.current_novel_interval : (1+step)*self.current_novel_interval]
            step_weight_list.append(step_weight)

            # arange interval along x-axis
            x_range = np.arange(step*self.current_novel_interval, (1+step)*self.current_novel_interval)
            step_x_range_list.append(x_range)

        # plt.ylim(0, 5)
        plt.tick_params(labelsize=23)

        # step-wise plot
        for step in range(1+self.current_step):
            plt.bar(step_x_range_list[step], step_weight_list[step], label=f'Step {step}',
                    fc=self.step_cmap[step])

        plt.xticks([])
        plt.title(self.dataset_name+" Joint-head L2-norm Weights")
        plt.legend()
        plt.savefig(self.output_dir + f'/{file_name}')
        plt.close()

    def plot_logits(self, model, joint_head, target_loader, file_name='noname_weight.png'):
        model.eval()
        joint_head.eval()

        all_logits = []
        all_labels = []

        for batch_idx, (x, label, _) in enumerate(tqdm(target_loader)):
            x, label = x.to(self.device), label.to(self.device)

            feat = model(x)
            output = joint_head(feat)

            all_logits.append(output.detach().clone().cuda())
            all_labels.append(label.detach().clone().cuda())

        all_logits = torch.cat(all_logits, dim=0).cuda()
        all_labels = torch.cat(all_labels, dim=0).cuda()
        print(all_logits.shape)
        print(all_labels.shape)

        # colors = ['r','b','g','y','b','p']
        result = []
        for i in range(self.current_novel_end):
            this_mask = all_labels == i
            this_logits = all_logits[this_mask]
            if len(this_logits) == 0:
                this_average = torch.zeros(this_logits.shape[1]).cpu().numpy()
            else:
                this_average = this_logits.mean(dim=0).cpu().numpy()
            # result = np.append(result, this_average)
            result.append(this_average)
            print("Class {}: Avg.Logits = {}".format(i, this_average))

        # print(result.shape)
        print(result)

        step_x_range_list = []
        for step in range(1+self.current_step):
            # arange interval along x-axis
            x_range = np.arange(step*self.current_novel_interval, (1+step)*self.current_novel_interval)
            step_x_range_list.append(x_range)

        # 4096x2160
        fig = plt.figure(figsize=(19.6, 10.8), dpi=800)
        for i in range(self.current_novel_end):
            this_ax = fig.add_subplot(10, 10, 1 + i)

            # step-wise color
            for step in range(1 + self.current_step):
                this_ax.bar(step_x_range_list[step],
                            result[i][step*self.current_novel_interval : (1+step)*self.current_novel_interval],
                            label=f'Step {step}',
                            fc=self.step_cmap[step])

            this_ax.set_title("Class {}".format(i), fontsize=2)

            # this_ax.set_ylim(-15, 15)
            this_ax.tick_params()
            # plt.rcParams['font.size'] = 30
            this_ax.set_yticks([])
            this_ax.set_xticks([])

        plt.xticks([])
        plt.yticks([])
        plt.savefig(self.output_dir + f'/{file_name}')
        plt.close()


def plot_logits_2D(args, model, dataloader, dataloader_name, fig_dir):
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    model.eval()

    all_logits = []
    all_labels = []

    for batch_idx, (x, label, _) in enumerate(tqdm(dataloader)):
        x, label = x.to(args.device), label.to(args.device)
        output1, output2, _ = model(x)
        # print(output1.shape)
        if args.head == 'head1':
            if args.IL_version == 'SplitHead12' or args.IL_version == 'AutoNovel':
                output = torch.cat((output1, output2), dim=1)
                # print("{}.shape={}".format(args.IL_version, output.shape))
            else:
                output = output1
                # print("{}.shape={}".format(args.IL_version, output.shape))
        else:
            print("Please set args.head = head1")
            # if args.IL_version == 'JointHead1' or args.IL_version == 'JointHead1woPseudo':
            #     output = output1[:, -args.num_unlabeled_classes:]
            # else:
            #     output = output2
        all_logits.append(output.detach().clone().cuda())
        all_labels.append(label.detach().clone().cuda())

    all_logits = torch.cat(all_logits, dim=0).cuda()
    all_labels = torch.cat(all_labels, dim=0).cuda()
    print(all_logits.shape)
    print(all_labels.shape)

    # colors = ['r','b','g','y','b','p']
    result = []
    for i in range(num_classes):
        this_mask = all_labels == i
        this_logits = all_logits[this_mask]
        if len(this_logits) == 0:
            this_average = torch.zeros(this_logits.shape[1]).cpu().numpy()
        else:
            this_average = this_logits.mean(dim=0).cpu().numpy()
        # result = np.append(result, this_average)
        result.append(this_average)
        print("Class {}: Avg.Logits = {}".format(i, this_average))

    # print(result.shape)
    print(result)

    x_old = np.arange(args.num_labeled_classes)
    x_new = np.arange(args.num_labeled_classes, args.num_labeled_classes+args.num_unlabeled_classes)

    fig = plt.figure(figsize=(30, 15))
    for i in range(num_classes):
        this_ax = fig.add_subplot(2, 5, 1+i)
        this_ax.bar(x_old, result[i][:args.num_labeled_classes], fc='r')
        this_ax.bar(x_new, result[i][args.num_labeled_classes:], fc='b')
        if i < args.num_labeled_classes:
            this_ax.set_title("Old {}".format(i),   fontsize=20)
        else:
            this_ax.set_title("New {}".format(i), fontsize=20)
        this_ax.set_ylim(-15, 15)
        this_ax.tick_params(labelsize=20)
        # plt.rcParams['font.size'] = 30
        this_ax.set_xticks([])

    plt.xticks([])
    plt.savefig(fig_dir + '_Logits2D_' + args.dataset_name+'_'+ dataloader_name + '.pdf')
    plt.close()


def plot_confusion_matrix(args, model, dataloader, dataloader_name, fig_dir, ind=None,
                          cmap=cm.binary, grid_font_size=12):
    model.eval()
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(dataloader)):
        x, label = x.to(args.device), label.to(args.device)
        output1, output2, _ = model(x)
        if args.head == 'head1':
            if args.IL_version == 'SplitHead12' or 'AutoNovel':
                output = torch.cat((output1, output2), dim=1)
            else:
                output = output1
        else:
            if args.IL_version == 'JointHead1' or args.IL_version == 'JointHead1woPseudo':
                output = output1[:, -args.num_unlabeled_classes:]
            else:
                output = output2

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    if ind is not None:
        ind = ind[:args.num_unlabeled_classes, :]
        idx = np.argsort(ind[:, 1])
        id_map = ind[idx, 0]
        id_map += args.num_labeled_classes

        # targets_new = targets
        targets_new = np.copy(targets)
        for i in range(args.num_unlabeled_classes):
            targets_new[targets == i + args.num_labeled_classes] = id_map[i]
        targets = targets_new

        y_pred = torch.from_numpy(preds)
        y_true = torch.from_numpy(targets)
    else:
        y_pred = torch.from_numpy(preds)
        y_true = torch.from_numpy(targets)


    categories = [i for i in range(args.num_labeled_classes+args.num_unlabeled_classes)]
    tick_marks = np.array(range(len(categories))) + 0.5

    cm = confusion_matrix(y_true, y_pred, categories)
    print(cm)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ind_array = np.arange(len(categories))
    x, y = np.meshgrid(ind_array, ind_array)

    if args.dataset_name == 'cifar10':
        plt.figure(figsize=(12, 8), dpi=120)
    else:
        # plt.figure(figsize=(30, 30), dpi=250)
        plt.figure(figsize=(12, 8), dpi=120)

    if grid_font_size >= 0:
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=grid_font_size, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    if args.dataset_name == 'cifar10':
        plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tick_params(labelsize=20)
    plt.rcParams['font.size'] = 20

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title,fontsize=23)
    plt.colorbar()
    xlocations = np.array(range(len(categories)))
    if args.dataset_name == 'cifar10':
        plt.xticks(xlocations, categories, rotation=90)
        plt.yticks(xlocations, categories)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.ylabel('true classes', fontsize=20)
    plt.xlabel('predicted classes', fontsize=20)

    plt.savefig(fig_dir+'_CM_'+args.dataset_name+'_'+dataloader_name+'.pdf')
    plt.close()

def plot_confusion_matrix_tri(args, model, dataloader, dataloader_name, fig_dir, ind_new1=None, ind_new2=None,
                              cmap=cm.binary, grid_font_size=12):
    model.eval()
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(dataloader)):
        x, label = x.to(args.device), label.to(args.device)
        output1, output2, output3, _ = model(x, output='test')
        output = output1

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    # create id_map for new_1
    ind_new1 = ind_new1[:args.num_unlabeled_classes1, :]
    idx_new1 = np.argsort(ind_new1[:, 1])
    id_map_new1 = ind_new1[idx_new1, 0]
    id_map_new1 += args.num_labeled_classes

    ind_new2 = ind_new2[:args.num_unlabeled_classes2, :]
    idx_new2 = np.argsort(ind_new2[:, 1])
    id_map_new2 = ind_new2[idx_new2, 0]
    id_map_new2 += args.num_labeled_classes + args.num_unlabeled_classes1

    # targets_new = targets
    targets_new = np.copy(targets)
    for i in range(args.num_unlabeled_classes1):
        targets_new[targets == i + args.num_labeled_classes] = id_map_new1[i]

    for i in range(args.num_unlabeled_classes2):
        targets_new[targets == i + args.num_labeled_classes+args.num_unlabeled_classes1] = id_map_new2[i]

    targets = targets_new

    y_pred = torch.from_numpy(preds)
    y_true = torch.from_numpy(targets)

    categories = [i for i in range(args.num_labeled_classes+args.num_unlabeled_classes)]
    tick_marks = np.array(range(len(categories))) + 0.5

    cm = confusion_matrix(y_true, y_pred, categories)
    print(cm)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ind_array = np.arange(len(categories))
    x, y = np.meshgrid(ind_array, ind_array)

    if args.dataset_name == 'cifar10':
        plt.figure(figsize=(12, 8), dpi=120)
    else:
        # plt.figure(figsize=(30, 30), dpi=250)
        plt.figure(figsize=(12, 8), dpi=120)

    if grid_font_size >= 0:
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=grid_font_size, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    if args.dataset_name == 'cifar10':
        plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tick_params(labelsize=20)
    plt.rcParams['font.size'] = 20

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title,fontsize=23)
    plt.colorbar()
    xlocations = np.array(range(len(categories)))
    if args.dataset_name == 'cifar10':
        plt.xticks(xlocations, categories, rotation=90)
        plt.yticks(xlocations, categories)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.ylabel('true classes', fontsize=20)
    plt.xlabel('predicted classes', fontsize=20)

    plt.savefig(fig_dir+'_CM_'+args.dataset_name+'_'+dataloader_name+'.pdf')
    plt.close()