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
from eval import *
from parse import *
from advdifformer import *


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
if args.dataset in ('synthetic'):
    dataset, dataset_val, dataset_te = load_synthetic_dataset(args.data_dir, syn_type=args.syn_type)
else:
    raise ValueError('Invalid dataname')

if args.dataset in ('synthetic'):
    class_num = 1
else:
    class_num = max(dataset.y.max().item() + 1, dataset.y.shape[1])
feat_num = dataset.x.shape[1]

print_dataset_info(dataset, args.dataset, dataset_val, dataset_te)

### Load method ###
if args.method == 'advdifformer':
    model = AdvDIFFormer(feat_num, args.hidden_channels, class_num, beta=args.beta, theta=args.theta,
             dropout=args.dropout, num_layers=args.num_layers, num_heads=args.num_heads, solver=args.solver, K_order=args.K_order,
             use_bn=args.use_bn, use_residual=args.use_residual).to(device)
else:
    raise ValueError('Invalid modelname')

if args.dataset in ('proteins', 'ppi', 'elliptic', 'twitch'):
    criterion = nn.BCEWithLogitsLoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')
elif args.dataset in ('synthetic'):
    criterion = nn.MSELoss(reduction='mean')
else:
    criterion = nn.CrossEntropyLoss(reduction='none' if args.method in ['irm', 'groupdro'] else 'mean')

if args.dataset in ('twitch'):
    eval_func = eval_rocauc
    logger = Logger(args.runs, max_as_opt=True)
elif args.dataset in ('synthetic'):
    eval_func = eval_mse
    logger = Logger(args.runs, max_as_opt=False)
else:
    eval_func = eval_acc
    logger = Logger(args.runs, max_as_opt=True)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    # dataset, dataset_val, dataset_te = load_synthetic_dataset(args.data_dir, run=run, syn_type=args.syn_type)
    dataset.x, dataset.y, dataset.edge_index, dataset.env, dataset.batch = \
        dataset.x.to(device), dataset.y.to(device), dataset.edge_index.to(device), dataset.env.to(
            device), dataset.batch.to(device)
    dataset_val.x, dataset_val.y, dataset_val.edge_index, dataset_val.batch = \
        dataset_val.x.to(device), dataset_val.y.to(device), dataset_val.edge_index.to(device), dataset_val.batch.to(
            device)
    for d in dataset_te:
        d.x, d.y, d.edge_index, d.batch = \
            d.x.to(device), d.y.to(device), d.edge_index.to(device), d.batch.to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss_compute(dataset, criterion, args)
        loss.backward()
        optimizer.step()
        result = evaluate_single_graph(model, dataset, dataset_val, dataset_te, eval_func, args)
        logger.add_result(run, result)

        if epoch % args.display_step == 0:
            m = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}% '
            for i in range(len(result)-2):
                m += f'Test OOD{i+1}: {100 * result[i+2]:.2f}% '
            print(m)
    logger.print_statistics(run)

results = logger.print_statistics()

# ### Save results ###
# if args.save_result:
#     save_result(args, results)