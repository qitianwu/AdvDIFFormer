from modules.utils import get_pool

from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_scatter

from torch_geometric.utils import softmax as sp_softmax
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros


def make_batch_mask(n_nodes, device='cpu'):
    max_node = n_nodes.max().item()
    mask = torch.zeros(len(n_nodes), max_node)
    for idx, nx in enumerate(n_nodes):
        mask[idx, :nx] = 1
    return mask.bool().to(device), max_node


def make_batch(n_nodes, device='cpu'):
    x = []
    for idx, ns in enumerate(n_nodes):
        x.extend([idx] * ns)
    return torch.LongTensor(x).to(device)


def to_pad(feat, mask, max_node, batch_size):
    model_dim = feat.shape[-1]
    new_feat = torch.zeros((batch_size, max_node, model_dim)).to(feat)
    new_feat[mask] = feat
    return new_feat


def full_attention_conv(
    qs, ks, vs, kernel, n_nodes, block_wise=False, output_attn=False
):
    if kernel == 'simple':
        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)

            # common vars
            device = qs.device
            node_mask, max_node = make_batch_mask(n_nodes, device)
            batch_size, batch = len(n_nodes), make_batch(n_nodes, device)

            # numerator

            q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, D]
            k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, D]
            v_pad = to_pad(vs, node_mask, max_node, batch_size)  # [B, M, D]
            qk_pad = torch.matmul(q_pad, k_pad.transpose(2, 1))  # [B, M, M]

            v_sum = torch.zeros((batch_size, vs.shape[-1])).to(device)
            v_idx = batch.unsqueeze(-1).repeat(1, vs.shape[-1])
            v_sum.scatter_add_(dim=0, index=v_idx, src=vs)

            numerator = torch.matmul(qk_pad, v_pad)[node_mask] + \
                torch.index_select(v_sum, dim=0, index=batch)
            # [N, D]

            denominator = qk_pad[node_mask].sum(dim=-1)  # [N]
            denominator += torch.index_select(
                n_nodes.float(), dim=0, index=batch
            )

            attn_output = numerator / denominator.unsqueeze(dim=-1)

        else:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lm,ld->md", ks, vs)
            attention_num = torch.einsum("nm,md->nd", qs, kvs)  # [N, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,ld->d", all_ones, vs)  # [D]
            # [N, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1)

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)
            ks_sum = torch.einsum("lm,l->m", ks, all_ones)
            attention_normalizer = torch.einsum("nm,m->n", qs, ks_sum)  # [N]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape)
            )  # [N, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) \
                    / attention_normalizer  # [N, L, H]
    else:
        raise NotImplementedError(f'Not Implemented for kernel {kernel}')

    if output_attn:
        attn = None
        return attn_output, attn
    else:
        return attn_output


class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''

    def __init__(
        self, in_channels, out_channels, num_heads=1,
        kernel='simple', use_weight=True
    ):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, x, n_nodes, block_wise=False, output_attn=False):
        # feature transformation
        query = self.Wq(x).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(x).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = x.reshape(-1, 1, self.out_channels)

        outputs, attns = [], []
        # compute full attentive aggregation
        for i in range(self.num_heads):
            qi, ki, vi = query[:, i, :], key[:, i, :], value[:, i, :]
            if output_attn:
                output, attn = full_attention_conv(
                    qi, ki, vi, self.kernel, n_nodes, block_wise, output_attn
                )  # [N, D]
                attns.append(attn)
            else:
                output = full_attention_conv(
                    qi, ki, vi, self.kernel, n_nodes, block_wise
                )  # [N, D]
            outputs.append(output)

        if self.num_heads > 1:
            outputs = torch.stack(outputs, dim=1)  # [N, H, D]
            final_output = outputs.mean(dim=1)
        else:
            final_output = outputs[0]

        if output_attn:
            attn = torch.cat(attns, dim=-1)  # [N, N, H]
            return final_output, attn
        else:
            return final_output


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.root_emb = torch.nn.Embedding(1, out_channels)
        self.bond_encoder = BondEncoder(emb_dim=out_channels)

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



class SAGEMolConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(SAGEMolConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.bond_encoder = BondEncoder(out_channels)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        edge_emb = self.bond_encoder(edge_attr)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        return torch.relu(x_j + edge_attr)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)



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



class DIFFormerMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, JK='last', num_heads=1, kernel='simple',
        alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True,
        gnn_type='gcn'
    ):
        super(DIFFormerMolNode, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.JK = JK
        self.residual = use_residual
        self.use_bn = use_bn
        self.use_weight = use_weight
        self.dropout = dropout
        self.alpha = alpha
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        self.gnn_convs = torch.nn.ModuleList()
        self.attn_convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            if gnn_type == 'gcn':
                self.gnn_convs.append(GCNConv(emb_dim, emb_dim))
            elif gnn_type == 'sage':
                self.gnn_convs.append(SAGEMolConv(emb_dim, emb_dim))
            elif gnn_type == 'gat':
                assert emb_dim % num_heads == 0, f'The dim {emb_dim} is not evenly divided by num_heads {num_heads}'
                self.gnn_convs.append(MyGATConv(emb_dim, emb_dim // num_heads, heads=num_heads))

            self.attn_convs.append(DIFFormerConv(
                emb_dim, emb_dim, num_heads=num_heads,
                kernel=kernel, use_weight=use_weight
            ))
            self.bns.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch, block_wise):

        n_nodes = torch_scatter.scatter(torch.ones_like(batch), batch)  # [B]

        layer_ = [self.atom_encoder(x)]

        for i in range(self.num_layer):
            x1 = self.gnn_convs[i](layer_[i], edge_index, edge_attr)
            x2 = self.attn_convs[i](layer_[i], n_nodes, block_wise)
            x = (x1 + x2) / 2.
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        if self.JK == 'last':
            node_repr = layer_[-1]
        elif self.JK == 'sum':
            node_repr = 0
            for layer in range(self.num_layer + 1):
                node_repr += layer_[layer]
        else:
            raise ValueError('JK should be "last" or "sum"')
        return node_repr


class DIFFormerMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, JK='last', pooling='mean',
        num_heads=1, kernel='simple', alpha=0.5, dropout=0.5,
        use_bn=True, use_residual=True, use_weight=True,
        gnn_type='gcn'
    ):
        super(DIFFormerMolGraph, self).__init__()
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

        self.model = DIFFormerMolNode(
            emb_dim, num_layer, JK, num_heads, kernel,
            alpha, dropout, use_bn, use_residual, use_weight, gnn_type
        )

    def forward(self, x, edge_index, edge_attr, batch, block_wise):
        node_feat = self.model(
            x=x, edge_index=edge_index,
            edge_attr=edge_attr, batch=batch, block_wise=block_wise
        )
        graph_feat = self.pool(node_feat, batch)
        return graph_feat



class DIFFormerMol(torch.nn.Module):
    def __init__(
        self, emb_dim, num_tasks, num_layer, JK='last', pooling='mean',
        num_heads=1, kernel='simple', alpha=0.5, gnn_type='gcn', 
        dropout=0.5, use_bn=True, use_residual=True, use_weight=True
    ):
        super(DIFFormerMol, self).__init__()
        self.model = DIFFormerMolGraph(
            emb_dim=emb_dim, num_layer=num_layer, JK=JK, pooling=pooling,
            num_heads=num_heads, kernel=kernel, alpha=alpha, dropout=dropout,
            use_bn=use_bn, use_residual=use_residual, use_weight=use_weight,
            gnn_type=gnn_type
        )
        if pooling == 'set2set':
            self.predictor = torch.nn.Linear(2 * emb_dim, num_tasks)
        else:
            self.predictor = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch, block_wise):
        graph_feat = self.model(x, edge_index, edge_attr, batch, block_wise)
        return self.predictor(graph_feat)
