from modules.utils import get_pool

from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch
from torch_geometric.nn import global_add_pool

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


from torch_geometric.utils import softmax as sp_softmax
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros


class MyGATConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, heads=1,
        negative_slope=0.2, dropout=0.1, add_self_loop=True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MyGATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.dropout_fun = torch.nn.Dropout(dropout)

        self.lin_src = self.lin_dst = Linear(
            in_channels, out_channels * heads,
            bias=False, weight_initializer='glorot'
        )
        self.add_self_loop = add_self_loop

        self.att_src = torch.nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_edge = torch.nn.Parameter(torch.zeros(1, heads, out_channels))

        self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels))
        self.lin_edge = BondEncoder(out_channels * heads)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        num_nodes = x.shape[0]
        if self.add_self_loop:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value='mean', num_nodes=num_nodes
            )

        H, C = self.heads, self.out_channels
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        edge_attr = self.lin_edge(edge_attr)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(
            edge_index, x=x, alpha=alpha, size=size,
            edge_attr=edge_attr.view(-1, H, C)
        )
        out = out.view(-1, H * C) + self.bias
        return out

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha_i + alpha_j + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.dropout_fun(sp_softmax(alpha, index, ptr, size_i))
        return alpha

    def message(self, x_j, alpha, edge_attr):
        return alpha.unsqueeze(-1) * (x_j + edge_attr)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.out_channels}, heads={self.heads})'
        )


class GATMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False, n_heads=1
    ):
        super(GATMolNode, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.n_heads = n_heads
        assert emb_dim % n_heads == 0, 'Dim is not evenly divided by n_heads'
        self.atom_encoder = AtomEncoder(emb_dim)
        if drop_ratio > 0 and drop_ratio < 1:
            self.dropout_layer = torch.nn.Dropout(drop_ratio)
        else:
            self.dropout_layer = torch.nn.Sequential()

        self.convs = torch.nn.ModuleList()
        self.BNs = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.convs.append(MyGATConv(
                out_channels=emb_dim // n_heads,
                in_channels=emb_dim, heads=n_heads
            ))
            self.BNs.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.BNs[layer](h)
            if layer == self.num_layer - 1:
                h = self.dropout_layer(h)
            else:
                h = self.dropout_layer(torch.relu(h))
            if self.residual:
                h += h_list[layer]
            h_list.append(h)
        if self.JK == 'last':
            node_repr = h_list[-1]
        elif self.JK == 'sum':
            node_repr = 0
            for layer in range(self.num_layer + 1):
                node_repr += h_list[layer]
        else:
            raise ValueError('JK should be "last" or "sum"')
        return node_repr


class VirtGATMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False, n_heads=1
    ):
        super(VirtGATMolNode, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.n_heads = n_heads
        self.atom_encoder = AtomEncoder(emb_dim)
        if drop_ratio > 0 and drop_ratio < 1:
            self.dropout_layer = torch.nn.Dropout(drop_ratio)
        else:
            self.dropout_layer = torch.nn.Sequential()

        self.convs = torch.nn.ModuleList()
        self.BNs = torch.nn.ModuleList()
        assert emb_dim % n_heads == 0, 'Dim is not evenly divided by n_heads'
        for _ in range(self.num_layer):
            self.convs.append(MyGATConv(
                out_channels=emb_dim // n_heads,
                in_channels=emb_dim, heads=n_heads
            ))
            self.BNs.append(torch.nn.BatchNorm1d(emb_dim))
        self.vir_mlp = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim * 2),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, emb_dim),
                torch.nn.BatchNorm1d(emb_dim),
                torch.nn.ReLU()
            ) for _ in range(self.num_layer - 1)
        ])
        self.virtual_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, batched_data):
        x, batch = batched_data.x, batched_data.batch
        edge_index, edge_attr = batched_data.edge_index, batched_data.edge_attr
        virt_emb = torch.zeros(batch[-1].item() + 1).to(edge_index)
        virt_emb = self.virtual_emb(virt_emb)
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virt_emb[batch]
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.BNs[layer](h)
            if layer == self.num_layer - 1:
                h = self.dropout_layer(h)
            else:
                h = self.dropout_layer(torch.relu(h))
            if self.residual:
                h += h_list[layer]
            h_list.append(h)
            if layer < self.num_layer - 1:
                virt_emb_temp = global_add_pool(
                    h_list[layer], batch) + virt_emb
                if self.residual:
                    virt_emb += self.dropout_layer(
                        self.vir_mlp[layer](virt_emb_temp))
                else:
                    virt_emb = self.dropout_layer(
                        self.vir_mlp[layer](virt_emb_temp))
        if self.JK == 'last':
            node_repr = h_list[-1]
        elif self.JK == 'sum':
            node_repr = 0
            for layer in range(self.num_layer + 1):
                node_repr += h_list[layer]
        else:
            raise ValueError('JK should be "last" or "sum"')
        return node_repr


class GATMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5, JK='last',
        residual=False, pooling='mean', n_heads=1
    ):
        super(GATMolGraph, self).__init__()
        self.pool_method = pooling
        if pooling not in ['attention', 'set2set']:
            self.pool = get_pool(pooling)
        elif pooling == 'attention':
            gate = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim * 2, 1)
            )
            self.pool = torch_geometric.nn.GlobalAttention(gate_nn=gate)
        else:
            self.pool = torch_geometric.nn.Set2Set(emb_dim, processing_steps=2)

        self.model = GATMolNode(
            emb_dim, num_layer, drop_ratio, JK, residual, n_heads)

    def forward(self, batched_data):
        x, batch = batched_data.x, batched_data.batch
        edge_index, edge_attr = batched_data.edge_index, batched_data.edge_attr
        node_feat = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_feat = self.pool(node_feat, batch)
        return graph_feat


class VirtGATMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5, JK='last',
        residual=False, pooling='mean', n_heads=1
    ):
        super(VirtSAGEMolGraph, self).__init__()
        self.pool_method = pooling
        if pooling not in ['attention', 'set2set']:
            self.pool = get_pool(pooling)
        elif pooling == 'attention':
            gate = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim * 2, 1)
            )
            self.pool = torch_geometric.nn.GlobalAttention(gate_nn=gate)
        else:
            self.pool = torch_geometric.nn.Set2Set(emb_dim, processing_steps=2)

        self.model = VirtGATMolNode(
            emb_dim, num_layer, drop_ratio, JK, residual, n_heads
        )

    def forward(self, batched_data):
        node_feat = self.model(batched_data)
        graph_feat = self.pool(node_feat, batched_data.batch)
        return graph_feat


class GATMol(torch.nn.Module):
    def __init__(
        self, emb_dim, num_tasks, num_layer, drop_ratio=0.5, JK='last',
        residual=False, pooling='mean', virtual=False, n_heads=1
    ):
        super(GATMol, self).__init__()
        if virtual:
            self.model = VirtGATMolGraph(
                emb_dim=emb_dim, num_layer=num_layer, drop_ratio=drop_ratio,
                JK=JK, residual=residual, pooling=pooling, n_heads=n_heads
            )
        else:
            self.model = GATMolGraph(
                emb_dim=emb_dim, num_layer=num_layer, drop_ratio=drop_ratio,
                JK=JK, residual=residual, pooling=pooling, n_heads=n_heads
            )
        if pooling == 'set2set':
            self.predictor = torch.nn.Linear(2 * emb_dim, num_tasks)
        else:
            self.predictor = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, batched_data):
        graph_feat = self.model(batched_data)
        return self.predictor(graph_feat)
