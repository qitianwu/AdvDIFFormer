from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit, StochasticBlockModelDataset
from torch_geometric.transforms import NormalizeFeatures, RadiusGraph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily, to_dense_adj, dense_to_sparse

from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv
from encoders import MLP, GCN
from difformer import DIFFormer

import pickle as pkl
import os
import csv
import json

def load_twitch(data_dir, lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"{data_dir}twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    # features = features[:, np.sum(features, axis=0) != 0] # remove zero cols. not need for cross graph task
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features

def load_twitch_dataset(data_dir, method, train_num=1):
    transform = T.NormalizeFeatures()
    sub_graphs = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']
    # x_list, edge_index_list, y_list, env_list = [], [], [], []
    # node_idx_list, batch_list = [], []
    # idx_shift = 0
    dataset_te = []
    for i, g in enumerate(sub_graphs):
        # torch_dataset = Twitch(root=f'{data_dir}Twitch',
        #                       name=g, transform=transform)
        # data = torch_dataset[0]
        A, label, features = load_twitch(data_dir, g)
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(label).reshape(-1, 1)
        if i == 0:
            dataset_tr = Data(x=x, edge_index=edge_index, y=y)
            dataset_tr.train_env_num = train_num
            dataset_tr.train_idx = torch.arange(x.shape[0])
            dataset_tr.batch = torch.zeros(x.shape[0], dtype=torch.long)
            dataset_tr.env = torch.zeros(x.shape[0])
        elif i == 1:
            dataset_val = Data(x=x, edge_index=edge_index, y=y)
            dataset_val.valid_idx = torch.arange(x.shape[0])
            dataset_val.batch = torch.zeros(x.shape[0], dtype=torch.long)
        else:
            dataset = Data(x=x, edge_index=edge_index, y=y)
            dataset.test_idx = torch.arange(x.shape[0])
            dataset.batch = torch.zeros(x.shape[0], dtype=torch.long)
            dataset_te.append(dataset)
        # x_list.append(x)
        # y_list.append(y)
        # edge_index_list.append(edge_index + idx_shift)
        # env_list.append(torch.ones(x.size(0)) * i)
        # node_idx_list.append(torch.arange(num_nodes) + idx_shift)
        # batch_list.append(torch.zeros(num_nodes) + i)
        # idx_shift += num_nodes
    # x = torch.cat(x_list, dim=0)
    # y = torch.cat(y_list, dim=0)
    # edge_index = torch.cat(edge_index_list, dim=1)
    # env = torch.cat(env_list, dim=0)
    # dataset = Data(x=x, edge_index=edge_index, y=y)
    # dataset.env = env
    # dataset.env_num = len(sub_graphs)
    # dataset.train_env_num = train_num
    # dataset.batch = torch.cat(batch_list, dim=0).long()

    # assert (train_num <= 5)
    #
    # ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    # idx = torch.randperm(ind_idx.size(0))
    # dataset.train_idx = node_idx_list[0]
    # dataset.valid_idx = node_idx_list[1]
    # dataset.test_idx = node_idx_list[2:]
    #
    # if method == 'eerm':
    #     A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
    #     train_edge_reindex = dense_to_sparse(A)[0]
    #     dataset.train_edge_reindex = train_edge_reindex

    return dataset_tr, dataset_val, dataset_te

def load_synthetic_dataset(data_dir, name, method, env_num=6, train_num=1, train_ratio=0.5, valid_ratio=0.25):
    transform = T.NormalizeFeatures()
    if name in ['cora', 'citeseer', 'pubmed']:
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                              name=name, transform=transform)
        preprocess_dir = os.path.join(data_dir, 'Planetoid', name)
    elif name == 'photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
        preprocess_dir = os.path.join(data_dir, 'Amazon', 'Photo')
    elif name == 'computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
        preprocess_dir = os.path.join(data_dir, 'Amazon', 'Computers')

    data = torch_dataset[0]

    edge_index = data.edge_index
    x = data.x
    d = x.shape[1]

    preprocess_dir = os.path.join(preprocess_dir, 'gen')
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)

    node_idx_list = [torch.arange(data.num_nodes) + i*data.num_nodes for i in range(env_num)]

    file_path = preprocess_dir + f'/homophily-{env_num}.pkl'
    print(file_path)
    if not os.path.exists(file_path):

        print("creating new synthetic data...")
        x_list, edge_index_list, y_list, env_list = [], [], [], []
        idx_shift = 0

        gap = 2 / (env_num + 1)
        p_ii = torch.arange(gap / 2, 2 + gap / 2, gap)
        p_ij = 2 - torch.arange(gap / 2, 2 + gap / 2, gap)

        with torch.no_grad():
            for i in range(env_num):
                x_list.append(x)
                y_list.append(data.y)
                edge_index_new = create_sbm_dataset(data, p_ii=p_ii[i], p_ij=p_ij[i])
                edge_index_list.append(edge_index_new + idx_shift)
                env_list.append(torch.ones(x.size(0)) * i)

                idx_shift += data.num_nodes

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        env = torch.cat(env_list, dim=0)
        dataset = Data(x=x, edge_index=edge_index, y=y)
        dataset.env = env

        with open(file_path, 'wb') as f:
            pkl.dump((dataset), f, pkl.HIGHEST_PROTOCOL)
    else:
        print("using existing synthetic data...")
        with open(file_path, 'rb') as f:
            dataset = pkl.load(f)

    assert (train_num <= env_num-1)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio): int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [node_idx_list[-1]] if train_num==env_num-1 else node_idx_list[train_num:]
    dataset.env_num = env_num
    dataset.train_env_num = train_num

    if method == 'eerm' or method == 'ours':
        A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
        train_edge_reindex = dense_to_sparse(A)[0]
        dataset.train_edge_reindex = train_edge_reindex

    return dataset

def load_arxiv_dataset(data_dir, method, train_num=3):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    year = torch.as_tensor(ogb_dataset.graph['node_year']).squeeze(1)

    year_bound = [[1960, 2014]]
    year_bound += [[2014, 2017]]
    year_bound += [[2017, 2018], [2018, 2019], [2019, 2020]]
    env = torch.zeros(label.shape[0])
    for n in range(year.shape[0]):
        ye = int(year[n])
        for i in range(1, len(year_bound)):
            if ye >= year_bound[i][0]:
                continue
            else:
                env[n] = i
                break

    tr_node_mask = (year <= year_bound[0][1])
    tr_edge_index, _ = subgraph(tr_node_mask, edge_index, relabel_nodes=True)
    tr_node_feat = node_feat[tr_node_mask]
    tr_label = label[tr_node_mask]
    dataset_tr = Data(x=tr_node_feat, edge_index=tr_edge_index, y=tr_label)
    dataset_tr.train_idx = torch.arange(tr_node_feat.size(0))
    dataset_tr.batch = torch.zeros(tr_node_feat.shape[0], dtype=torch.long)
    dataset_tr.env = env[tr_node_mask]
    dataset_tr.env_num = len(year_bound)
    dataset_tr.train_env_num = train_num

    val_node_mask = (year <= year_bound[1][1])
    val_edge_index, _ = subgraph(val_node_mask, edge_index, relabel_nodes=True)
    val_node_feat = node_feat[val_node_mask]
    val_label = label[val_node_mask]
    dataset_val = Data(x=val_node_feat, edge_index=val_edge_index, y=val_label)
    idx = torch.arange(val_node_feat.size(0))
    val_year = year[val_node_mask]
    dataset_val.valid_idx = idx[(val_year > year_bound[1][0])]
    dataset_val.batch = torch.zeros(val_node_feat.shape[0], dtype=torch.long)

    dataset_te = []
    for i in range(2, len(year_bound)):
        te_node_mask = (year <= year_bound[i][1])
        te_edge_index, _ = subgraph(te_node_mask, edge_index, relabel_nodes=True)
        te_node_feat = node_feat[te_node_mask]
        te_label = label[te_node_mask]
        dataset = Data(x=te_node_feat, edge_index=te_edge_index, y=te_label)
        idx = torch.arange(te_node_feat.size(0))
        te_year = year[te_node_mask]
        dataset.test_idx = idx[(te_year > year_bound[i][0])]
        dataset.batch = torch.zeros(te_node_feat.shape[0], dtype=torch.long)
        dataset_te.append(dataset)

    return dataset_tr, dataset_val, dataset_te


#
# def load_arxiv_dataset(data_dir, method, train_num=3, train_ratio=0.5, valid_ratio=0.25, inductive=True):
#     from ogb.nodeproppred import NodePropPredDataset
#
#     ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')
#
#     node_years = ogb_dataset.graph['node_year']
#
#     edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
#     node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
#     label = torch.as_tensor(ogb_dataset.labels)
#
#     year_bound = [2011, 2014, 2016, 2018, 2020]
#     env = torch.zeros(label.shape[0])
#     for n in range(node_years.shape[0]):
#         year = int(node_years[n])
#         for i in range(len(year_bound)-1):
#             if year >= year_bound[i+1]:
#                 continue
#             else:
#                 env[n] = i
#                 break
#
#     dataset = Data(x=node_feat, edge_index=edge_index, y=label)
#     dataset.env = env
#     dataset.env_num = len(year_bound)
#     dataset.train_env_num = train_num
#
#     # ind_mask = (node_years < year_bound[0]).squeeze(1)
#     idx = torch.arange(dataset.num_nodes)
#
#     # ind_idx = idx[ind_mask]
#     # idx_ = torch.randperm(ind_idx.size(0))
#     # train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
#     # valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
#     # test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
#     # dataset.train_idx = ind_idx[train_idx_ind]
#     # print('dataset',max(env[dataset.train_idx].long()))
#     # dataset.valid_idx = ind_idx[valid_idx_ind]
#     # dataset.test_in_idx = ind_idx[test_idx_ind]
#
#     # dataset.test_ood_idx = []
#     #
#     # for i in range(train_num, len(year_bound)-1):
#     #     ood_mask_i = ((node_years >= year_bound[i]) * (node_years < year_bound[i+1])).squeeze(1)
#     #     dataset.test_ood_idx.append(idx[ood_mask_i])
#
#     train_mask = (node_years <= year_bound[0]).squeeze(1)
#     dataset.train_idx = idx[train_mask]
#     valid_mask = ((node_years > year_bound[0]) * (node_years <= year_bound[1])).squeeze(1)
#     dataset.valid_idx = idx[valid_mask]
#     dataset.test_idx = []
#     for i in range(1, len(year_bound) - 1):
#         ood_mask_i = ((node_years > year_bound[i]) * (node_years <= year_bound[i + 1])).squeeze(1)
#         dataset.test_idx.append(idx[ood_mask_i])
#
#     dataset.batch = torch.zeros(node_feat.shape[0], dtype=torch.long)
#
#     if method == 'eerm':
#         A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
#         train_edge_reindex = dense_to_sparse(A)[0]
#         dataset.train_edge_reindex = train_edge_reindex
#
#     return dataset

# @torch.no_grad()
# def create_synthetic_dataset(data_dir, num_nodes, feat_dim, block_num, env_num, syn_type='edge'):
#
#     p_ii = 0.1 - torch.arange(0, 0.1, 0.1 / env_num)
#     p_ij = 0.01 + torch.arange(0, 0.1, 0.1 / env_num)
#     # interval = num_nodes // block_num
#     # block_sizes = [interval for _ in range(block_num-1)] + [num_nodes%block_num+interval]
#
#     latents = torch.zeros(num_nodes, 1).uniform_()
#     blocks = torch.zeros(num_nodes, dtype=torch.long)
#
#     cnt, interval = 0., 1.0 / block_num
#     for c in range(block_num):
#         mask = (latents[:, 0] >= cnt) * (latents[:, 0] < (cnt + interval))
#         blocks[mask] = c
#         cnt += interval
#
#     f_x = MLP(1, feat_dim, feat_dim, 2)
#     # f_y = GCN(1, feat_dim, 1, 1)
#     f_y = DIFFormer(1, feat_dim, 1, 1)
#
#     node_feat = f_x(latents)
#
#     edge_probs = torch.ones((block_num, block_num)) * p_ij[0]
#     edge_probs[torch.arange(block_num), torch.arange(block_num)] = p_ii[0]
#     row, col = torch.combinations(torch.arange(num_nodes), r=2).t()
#     mask = torch.bernoulli(edge_probs[blocks[row], blocks[col]]).to(torch.bool)
#     edge_index = torch.stack([row[mask], col[mask]], dim=0).to(torch.long)
#
#     node_label = f_y(latents, edge_index)
#
#     d = Data(x=node_feat, edge_index=edge_index, y=node_label)
#     d.env = torch.ones(num_nodes).to(torch.long) * 0
#     d.batch = torch.zeros(num_nodes).to(torch.long)
#     d.y = d.y.reshape(-1, 1)
#     dataset_list = [d]
#
#     def gen_dataset(env_id):
#
#         edge_probs = torch.ones((block_num, block_num)) * p_ij[env_id]
#         edge_probs[torch.arange(block_num), torch.arange(block_num)] = p_ii[env_id]
#         row, col = torch.combinations(torch.arange(num_nodes), r=2).t()
#         mask = torch.bernoulli(edge_probs[blocks[row], blocks[col]]).to(torch.bool)
#         edge_index = torch.stack([row[mask], col[mask]], dim=0).to(torch.long)
#         d = Data(x=node_feat, edge_index=edge_index, y=node_label)
#
#         # file_dir = data_dir + f'{env_id}'
#         # dataset = StochasticBlockModelDataset(file_dir, block_sizes, edge_probs, num_channels=feat_dim)
#         # d = dataset[0]
#
#         d.env = torch.ones(num_nodes).to(torch.long) * env_id
#         d.batch = torch.zeros(num_nodes).to(torch.long)
#         d.y = d.y.reshape(-1, 1)
#
#         return d
#
#     dataset_list += [gen_dataset(env_id=i) for i in range(1, env_num)]
#     return dataset_list


@torch.no_grad()
def create_synthetic_dataset(num_nodes, feat_dim, block_num, env_num, syn_type='edge'):
    if syn_type == 'homophily':
        p_iis = 0.1 * torch.ones(env_num)
        p_ijs = 0.01 + torch.arange(0, 0.05, 0.05 / env_num)
        block_nums = block_num * torch.ones(env_num, dtype=torch.long)
    elif syn_type == 'block':
        p_iis = 0.1 * torch.ones(env_num)
        p_ijs = 0.01 * torch.ones(env_num)
        block_nums = block_num + torch.arange(0, env_num, 1)
    elif syn_type == 'density':
        p_iis = 0.1 + torch.arange(0, 0.1, 0.1 / env_num)
        p_ijs = 0.01 + torch.arange(0, 0.1, 0.1 / env_num)
        block_nums = block_num * torch.ones(env_num, dtype=torch.long)
    else:
        raise NotImplementedError

    f_x = MLP(1, feat_dim, feat_dim, 2)
    f_y1 = DIFFormer(1, feat_dim, 1, 1, use_residual=True, use_act=True, use_bn=True, use_weight=True, use_graph=False)
    f_y2 = GCN(1, feat_dim, 1, 2)

    def gen_edge_index(block_n, p_ii, p_ij):
        blocks = torch.zeros(num_nodes, dtype=torch.long)
        cnt, interval = 0., 1.0 / block_n
        for c in range(block_n):
            mask = (latents[:, 0] >= cnt) * (latents[:, 0] < (cnt + interval))
            blocks[mask] = c
            cnt += interval

        edge_probs = torch.ones((block_n, block_n)) * p_ij
        edge_probs[torch.arange(block_n), torch.arange(block_n)] = p_ii
        row, col = torch.combinations(torch.arange(num_nodes), r=2).t()
        mask = torch.bernoulli(edge_probs[blocks[row], blocks[col]]).to(torch.bool)
        edge_index = torch.stack([row[mask], col[mask]], dim=0).to(torch.long)
        return edge_index

    latents = torch.zeros(num_nodes, 1).uniform_()
    node_feat = f_x(latents)
    edge_index = gen_edge_index(block_nums[0], p_iis[0], p_ijs[0])
    node_label = f_y1(latents, edge_index) + f_y2(latents, edge_index)

    d = Data(x=node_feat, edge_index=edge_index, y=node_label)
    d.env = torch.ones(num_nodes).to(torch.long) * 0
    d.batch = torch.zeros(num_nodes).to(torch.long)
    d.y = d.y.reshape(-1, 1)
    dataset_list = [d]

    for e in range(1, env_num):
        edge_index = gen_edge_index(block_nums[e], p_iis[e], p_ijs[e])
        d = Data(x=node_feat, edge_index=edge_index, y=node_label)
        d.env = torch.ones(num_nodes).to(torch.long) * e
        d.batch = torch.zeros(num_nodes).to(torch.long)
        d.y = d.y.reshape(-1, 1)
        dataset_list.append(d)

    return dataset_list

def load_synthetic_dataset(data_dir, run=0, num_nodes=1000, feat_dim=4, block_num=5, env_num=12, syn_type='edge'):
    data_dir = data_dir + f'synthetic/{syn_type}-{run}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        datasets = create_synthetic_dataset(num_nodes, feat_dim, block_num, env_num, syn_type)
        for i, d in enumerate(datasets):
            file_path = data_dir + f'/{i}.pkl'
            with open(file_path, 'wb') as f:
                pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
    else:
        print("using existing synthetic data...")
        datasets = []
        for i in range(env_num):
            file_path = data_dir + f'/{i}.pkl'
            with open(file_path, 'rb') as f:
                dataset = pkl.load(f)
            datasets.append(dataset)
    # datasets = create_synthetic_dataset(num_nodes, feat_dim, block_num, env_num, syn_type)
    dataset_tr, dataset_val = datasets[0], datasets[1]
    dataset_tr.train_idx = torch.arange(dataset_tr.x.size(0))
    dataset_tr.train_env_num = 1
    dataset_val.valid_idx = torch.arange(dataset_val.x.size(0))

    dataset_te = datasets[2:]
    for i, d in enumerate(dataset_te):
        d.test_idx = torch.arange(d.x.size(0))
        dataset_te[i] = d
    return dataset_tr, dataset_val, dataset_te