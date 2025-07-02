from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures, RadiusGraph
from torch_geometric.data import Data, Batch, Dataset, InMemoryDataset
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily, to_dense_adj, dense_to_sparse
from torch_sparse import coalesce
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv

import pickle as pkl
import os
import os.path as osp
import csv
import json

def link_sampling(x, edge_index, n):    # n means 1 positive for n negative edges
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    # Convert edge_index to numpy arrays for processing
    edge_index_np = edge_index.numpy()
    positive_edges = set(map(tuple, edge_index_np.T))

    # Generate negative samples
    negative_edges = set()
    while len(negative_edges) < num_edges * n:
        u = np.random.randint(num_nodes)
        v = np.random.randint(num_nodes)
        if u != v and (u, v) not in positive_edges and (v, u) not in negative_edges:
            negative_edges.add((u, v))

    # Convert negative samples to tensor
    negative_edges_np = np.array(list(negative_edges)).T
    edge_index_sampled = torch.tensor(np.hstack([edge_index_np, negative_edges_np]), dtype=torch.long)

    # Create labels: 1 for positive edges, 0 for negative edges
    link_pre_y = torch.cat([
        torch.ones(num_edges, dtype=torch.float),
        torch.zeros(len(negative_edges), dtype=torch.float)
    ])

    # print("sampled: ", edge_index_sampled.size())
    # print("pre: ", link_pre_y.size())

    return edge_index_sampled, link_pre_y.unsqueeze(1)

class DppinDataset(InMemoryDataset):
    def __init__(self, data_dir, root, mode='train', window=10, train_num=4, valid_num=1, transform=None, pre_transform=None):
        self.splits = ['train', 'valid', 'test0','test1','test2','test3','test4','test5','test6','test7']
        assert mode in self.splits
        self.data_dir = os.path.join(data_dir, 'dppin')
        self.mode = mode
        self.window = window
        self.train_num = train_num
        self.valid_num = valid_num
        # self.name_list = ['Babu','Breitkreutz','Gavin','Hazbun','Ho','Ito','Krogan(LCMS)','Krogan(MALDI)','Lambert','Tarassov','Uetz','Yu']
        # self.name_list = ['Ito', 'Ho', 'Breitkreutz','Babu', 'Gavin', 'Hazbun','Krogan(LCMS)','Krogan(MALDI)','Lambert','Tarassov','Uetz','Yu']
        # self.name_list = ['Gavin','Krogan(MALDI)', 'Breitkreutz', 'Ho','Krogan(LCMS)', 'Babu', 'Hazbun', 'Ito','Lambert','Tarassov','Uetz','Yu']
        self.name_list = ['Gavin', 'Ho', 'Hazbun', 'Ito','Krogan(LCMS)' , 'Breitkreutz', 'Babu','Krogan(MALDI)','Lambert','Tarassov','Uetz','Yu']
        super(DppinDataset, self).__init__(root, transform, pre_transform)
        idx = self.splits.index(mode)
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'ddpin_train.pt',  f'ddpin_valid.pt', f'ddpin_test0.pt', f'ddpin_test1.pt', f'ddpin_test2.pt', f'ddpin_test3.pt', f'ddpin_test4.pt', f'ddpin_test5.pt', f'ddpin_test6.pt', f'ddpin_test7.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        index=0
        if self.mode == 'train':
            name_list = self.name_list[:self.train_num]
            env_list = [i for i in range(self.train_num)]
        elif self.mode == 'valid':
            name_list = self.name_list[self.train_num:self.train_num+self.valid_num]
            #env_list = [self.train_num]
            env_list = [0]
        else:
            name_list = self.name_list[self.train_num+self.valid_num+int(self.mode[-1]):self.train_num+self.valid_num+int(self.mode[-1])+1]
            #env_list = [self.train_num+self.valid_num+int(self.mode[-1])]
            env_list = [0]
        for i, name in enumerate(name_list):
            env_num = env_list[i]
            folder=os.path.join(self.data_dir, f'DPPIN-{name}')
            src_list = [[] for _ in range(36)]
            tgt_list = [[] for _ in range(36)]
            weight_list = [[] for _ in range(36)]  # weight_list[i] stores weight for i-th graph
            
            graph_file_path = os.path.join(folder, 'Dynamic_PPIN.txt')
            df = pd.read_csv(graph_file_path, delimiter=',', header=None)

            for ind, row in df.iterrows():
                src = int(row[0])
                tgt = int(row[1])
                timestamp = int(row[2])
                weight = row[3]

                src_list[timestamp].append(src)
                tgt_list[timestamp].append(tgt)
                weight_list[timestamp].append(weight)

            edge_list = []  # edge_list[i] stores edge_index for i-th graph

            for i in range(len(src_list)):
                row, col = src_list[i], tgt_list[i]
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_list.append(edge_index)
            
            label_file = os.path.join(folder, 'Node_BLabels.txt')
            labels = pd.read_csv(label_file, sep=',')['Label_ID']
            labels = torch.tensor(labels.values,dtype=torch.long)
            mask = labels != -1

            env = torch.full_like(labels, env_num)

            for i in range(self.window,36):
                if i != 28 and i != 29:
                    continue
                feat_file = os.path.join(folder, 'Node_Features.txt')
                features = pd.read_csv(feat_file, sep=',', header=None)

                node_names = features[1]
                features = features.drop(columns=[0, 1])  # drop indeces and names
                past_features = torch.tensor(features.values[:,i-self.window:i], dtype=torch.float)

                edge_index_sampled, link_pre_y = link_sampling(past_features, edge_list[i-1], 3)

                data=Data(
                        x=past_features,
                        y=labels,
                        edge_index=edge_list[i-1],
                        edge_attr=torch.tensor(weight_list[i-1], dtype=torch.float).reshape(-1,1),
                        name=f'data_{name}_{i-self.window}', 
                        idx=index,
                        env=env,
                        edge_index_sampled=edge_index_sampled,
                        link_pre_y=link_pre_y
                    )
                index += 1
                data.node_rgs_y = torch.tensor(features.values[:,i], dtype=torch.float)

                # next_edge_index = coalesce(edge_list[i], None, past_features.size(0), past_features.size(0))
                # data.link_pre_y = next_edge_index 

                

                data.mask = mask
                data.link_cls_y = torch.tensor(weight_list[i], dtype=torch.float).reshape(-1,1)
                data_list.append(data)
        idx=self.splits.index(self.mode)
        torch.save(self.collate(data_list), self.processed_paths[idx])       


    # def len(self):
    #     return len(self.processed_file_names)

    # def get(self, idx):
    #     data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    #     return data