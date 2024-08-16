from torch_geometric.utils import degree
import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = torch.nn.Embedding(4, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCN(torch.nn.Module):
    def __init__(self, emb_dim, num_layer, drop_ratio, residual=True):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.atom_encoder = torch.nn.Embedding(16, emb_dim)
        self.edge_predictor = torch.nn.Linear(emb_dim * 2, 1)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.drop_fun = torch.nn.Dropout(drop_ratio)
        for i in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.convs.append(GCNConv(emb_dim))

    def forward(self, graph):
        x = self.atom_encoder(graph.x)
        for i in range(self.num_layer):
            conv_res = self.batch_norms[i](self.convs[i](
                x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr
            ))
            x = self.drop_fun(torch.relu(conv_res)) + \
                (x if self.residual else 0)

        row, col = graph.edge_index
        return self.edge_predictor(torch.cat([x[row], x[col]], dim=-1))
