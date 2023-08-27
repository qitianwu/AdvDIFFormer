import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from torch_geometric.data import ShaDowKHopSampler

from logger import Logger, save_result
from data_utils import print_dataset_info
from dataset import *
from eval import evaluate_multi_graph, evaluate_single_graph, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args
from model import *
from ours import *
from ours2 import *

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
# multi-graph datasets, divide graphs into train/valid/test
if args.dataset == 'twitch':
    dataset = load_twitch_dataset(args.data_dir, args.method, train_num=3)
elif args.dataset == 'elliptic':
    dataset = load_elliptic_dataset(args.data_dir, args.method, train_num=5)
# single-graph datasets, divide nodes into train/valid/test
elif args.dataset == 'arxiv':
    dataset, dataset_val, dataset_te = load_arxiv_dataset(args.data_dir, args.method, train_num=3)
# synthetic datasets, add spurious node features
elif args.dataset in ('cora', 'citeseer', 'pubmed', 'photo', 'computer'):
    dataset = load_synthetic_dataset(args.data_dir, args.dataset, args.method, train_num=1)
else:
    raise ValueError('Invalid dataname')

if len(dataset.y.shape) == 1:
    dataset.y = dataset.y.unsqueeze(1)
class_num = max(dataset.y.max().item() + 1, dataset.y.shape[1])
feat_num = dataset.x.shape[1]

if args.dataset == 'twitch':
    print_dataset_info(dataset, args.dataset)
    dataset.x, dataset.y, dataset.edge_index, dataset.env, dataset.batch = \
        dataset.x.to(device), dataset.y.to(device), dataset.edge_index.to(device), dataset.env.to(
            device), dataset.batch.to(device)
elif args.dataset == 'arxiv':
    print_dataset_info(dataset, args.dataset, dataset_val, dataset_te)
    dataset.x, dataset.y, dataset.edge_index, dataset.env, dataset.batch = \
        dataset.x.to(device), dataset.y.to(device), dataset.edge_index.to(device), dataset.env.to(
            device), dataset.batch.to(device)
    dataset_val.x, dataset_val.y, dataset_val.edge_index, dataset_val.batch = \
        dataset_val.x.to(device), dataset_val.y.to(device), dataset_val.edge_index.to(device), dataset_val.batch.to(device)
    for d in dataset_te:
        d.x, d.y, d.edge_index, d.batch = \
        d.x.to(device), d.y.to(device), d.edge_index.to(device), d.batch.to(device)

### Load method ###
is_multilabel = args.dataset in ('proteins', 'ppi')

if args.method in ('erm', 'reg', 'dropedge', 'irm', 'mixup', 'groupdro', 'coral', 'dann', 'eerm', 'srgnn'):
    model = Baseline(feat_num, class_num, dataset, args, device, dataset.train_env_num, is_multilabel).to(device)
    if args.method == 'srgnn':
        model.srgnn_preprocess(dataset, num = min(dataset.train_idx.shape[0], 5000),beta=args.kmm_beta)
elif args.method == 'ours':
    model = DIFFormer(feat_num, args.hidden_channels, class_num, num_layers=args.num_layers, alpha=args.alpha, dropout=args.dropout,
                      num_heads=args.num_heads, kernel=args.kernel,
                      use_bn=args.use_bn, use_residual=args.use_residual,
                      use_weight=args.use_weight).to(device)
elif args.method == 'ours2':
    model = Ours(feat_num, args.hidden_channels, class_num, num_layers=args.num_layers, alpha=args.alpha, dropout=args.dropout,
                      num_heads=args.num_heads, kernel=args.kernel, K_order=args.K_order,
                      use_bn=args.use_bn, use_residual=args.use_residual).to(device)

if args.method != 'mixup':
    if args.dataset in ('proteins', 'ppi', 'elliptic', 'twitch'):
        criterion = nn.BCEWithLogitsLoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')
else:
    criterion = LabelSmoothLoss(args.label_smooth_val, mode='multilabel' if is_multilabel else 'classy_vision')

if args.dataset in ('proteins', 'ppi', 'twitch'):
    eval_func = eval_rocauc
elif args.dataset in ('elliptic'):
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

tr_acc, val_acc = [], []

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    if args.method == 'eerm':
        optimizer_gnn = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_pred = torch.optim.Adam(model.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_aug = torch.optim.Adam(model.gl.parameters(), lr=args.lr_a)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        if args.method == 'eerm':
            model.gl.reset_parameters()
            beta = 1 * args.beta * epoch / args.epochs + args.beta * (1- epoch / args.epochs)
            for m in range(args.T):
                Var, Mean, Log_p = model.loss_compute(dataset, criterion, args)
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
            optimizer.zero_grad()
            loss = model.loss_compute(dataset, criterion, args)
            loss.backward()
            optimizer.step()
        if args.dataset == 'arxiv':
            result = evaluate_single_graph(model, dataset, dataset_val, dataset_te, eval_func, args)
        elif args.dataset == 'twitch':
            result = evaluate_multi_graph(model, dataset, eval_func, args)
        logger.add_result(run, result)

        tr_acc.append(result[0])
        val_acc.append(result[2])

        if epoch % args.display_step == 0:
            m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}% '
            for i in range(len(result)-2):
                m += f'Test OOD{i+1}: {100 * result[i+2]:.2f}% '
            print(m)
    logger.print_statistics(run)


results = logger.print_statistics()

### Save results ###
if args.save_result:
    save_result(args, results)