import argparse
import sys
import os, random
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from torch_geometric.data import ShaDowKHopSampler
from torch_geometric.loader import DataLoader

from logger import Logger, save_result
from data_utils import print_dataset_info
from dataset_clique import *
from eval import evaluate_node_task, evaluate_edge_task, evaluate_link_task, eval_acc, eval_rocauc, eval_f1, eval_rmse
from parse import parse_method, parser_add_main_args
from model import *
from advdifformer import *
# from ours import *

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_tr = DppinDataset(args.data_dir, osp.join('data/', f'DPPIN-{args.window}/'), mode='train', window=args.window, train_num=args.train_num, valid_num=1)
dataset_val = DppinDataset(args.data_dir, osp.join('data/', f'DPPIN-{args.window}/'), mode='valid', window=args.window, train_num=args.train_num, valid_num=1)
dataset_te = [DppinDataset(args.data_dir, osp.join('data/', f'DPPIN-{args.window}/'), mode=f'test{i}', window=args.window, train_num=args.train_num, valid_num=1) for i in range(11-args.train_num)]



# for data in dataset_val:
#     print(data)
# for data in dataset_te:
#     print(data)
train_loader = DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
test_loader = [DataLoader(i, batch_size=args.batch_size, shuffle=False) for i in dataset_te]





# if len(dataset.y.shape) == 1:
#     dataset.y = dataset.y.unsqueeze(1)
if args.task in ['node-cls', 'link-pre']:
    class_num = 2
elif args.task in ['node-reg', 'edge-reg']:
    class_num = 1
# class_num = max(dataset.y.max().item() + 1, dataset.y.shape[1])
feat_num = args.window

### Load method ###

if args.method == 'mixup':
    #criterion = LabelSmoothLoss(args.label_smooth_val, mode='classy_vision')
    criterion = nn.MSELoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')
elif args.task in ['node-cls', 'link-pre']:
    # criterion = LabelSmoothLoss(args.label_smooth_val, mode='classy_vision')
    # criterion = nn.MultiMarginLoss(margin=0.5)
    criterion = nn.BCEWithLogitsLoss()
elif args.task in ['node-reg', 'edge-reg']:
    criterion = nn.MSELoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')

if args.method in ('erm', 'reg', 'dropedge', 'irm', 'mixup', 'groupdro', 'coral', 'dann', 'eerm', 'srgnn'):
    model = Baseline(feat_num, class_num, args, device, train_env_num=args.train_num).to(device)
# elif args.method == 'ours':
#     model = GLIND(feat_num, class_num, args, device).to(device)
elif args.method == 'difformer':
    model = DIFFormer(feat_num, args.hidden_channels, class_num, num_layers=args.num_layers, num_heads=args.num_heads, kernel=args.kernel,
                 alpha=args.alpha, dropout=args.dropout, use_bn=args.use_bn, use_residual=args.use_residual, use_weight=args.use_weight).to(device)
elif args.method == 'advdifformer':
    model = AdvDIFFormer(feat_num, args.hidden_channels, class_num, beta=args.eerm_beta, theta=args.theta,
                 dropout=args.dropout, num_layers=args.num_layers, num_heads=args.num_heads, solver=args.solver, K_order=args.K_order,
                 use_bn=args.use_bn, use_residual=args.use_residual, task=args.task).to(device)
# if args.method != 'mixup':
#     if args.dataset in ('proteins', 'ppi', 'elliptic', 'twitch'):
#         criterion = nn.BCEWithLogitsLoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')
#     else:
#         criterion = nn.CrossEntropyLoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')
# else:
#     criterion = LabelSmoothLoss(args.label_smooth_val, mode='multilabel' if is_multilabel else 'classy_vision')

# if args.dataset in ('proteins', 'ppi', 'twitch'):
#     eval_func = eval_rocauc
# elif args.dataset in ('elliptic'):
#     eval_func = eval_f1
# else:
#     eval_func = eval_acc
if args.task in ['node-cls', 'link-pre']:
    eval_func = eval_rocauc
elif args.task in ['node-reg','edge-reg']:
    eval_func = eval_rmse
logger = Logger(args.runs, args)

# model.train()
# print('MODEL:', model)

tr_acc, val_acc = [], []

### Training loop ###
for run in range(args.runs):
    t0 = time.time()
    model.reset_parameters()
    if args.method == 'eerm':
        optimizer_gnn = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_pred = torch.optim.Adam(model.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_aug = torch.optim.Adam(model.gl.parameters(), lr=args.lr_a)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    a,b = [], []
    for epoch in range(args.epochs):
        t1 = time.time()
        model.train()
        if args.method == 'eerm':
            model.gl.reset_parameters()
            beta = 1 * args.eerm_beta * epoch / args.epochs + args.eerm_beta * (1- epoch / args.epochs)
            for batch in train_loader:
                batch.to(device)
                for m in range(args.T):
                    Var, Mean, Log_p = model.loss_compute(batch, criterion, args)
                    outer_loss = Var + beta * Mean
                    reward = Var.detach()
                    inner_loss = - reward * Log_p
                    if m == 0:
                        optimizer_gnn.zero_grad()
                        optimizer_pred.zero_grad()
                        outer_loss.backward()
                        optimizer_gnn.step()
                        optimizer_pred.step()
                    optimizer_aug.zero_grad()
                    inner_loss.backward()
                    optimizer_aug.step()
                loss = outer_loss
        else:
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                loss = model.loss_compute(batch, criterion, args)
                loss.backward()
                optimizer.step()
        t2=time.time()

        if args.task in ['node-cls', 'node-reg']:
            result = evaluate_node_task(model, train_loader, val_loader, test_loader, eval_func, args, device)
            t3=time.time()
        elif args.task == 'edge-reg':
            result = evaluate_edge_task(model, train_loader, val_loader, test_loader, eval_func, args, device)
            t3=time.time()
        elif args.task == 'link-pre':
            result = evaluate_link_task(model, train_loader, val_loader, test_loader, eval_func, args, device) 
            t3=time.time()
        logger.add_result(run, result)
        tr_acc.append(result[0])
        val_acc.append(result[2])
        a.append(t2-t1)
        b.append(t3-t2)
        print(t2-t1,t3-t2)
        if epoch % args.display_step == 0:
            if args.task in ['node-cls', 'link-pre']:
                m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.4f}%, Valid: {100 * result[1]:.4f}% '
                for i in range(len(result)-2):
                    m += f'Test OOD{i+1}: {100 * result[i+2]:.4f}% '
            elif args.task in ['node-reg','edge-reg']:
                m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {result[0]:.4f}, Valid: {result[1]:.4f} '
                for i in range(len(result)-2):
                    m += f'Test OOD{i+1}: {result[i+2]:.4f} '
            print(m)
    if args.task in ['node-cls', 'link-pre']:
        logger.print_statistics(run)
    else:
        logger.print_statistics_neg(run)
    t4 = time.time()
    print(sum(a)/10, sum(b)/10, (t4-t0)/10)


if args.task in ['node-cls', 'link-pre']:
    results = logger.print_statistics()
else:
    results = logger.print_statistics_neg()

### Save results ###
if args.save_result:
    save_result(args, results)