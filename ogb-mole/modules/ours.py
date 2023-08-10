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

def to_block(inputs, n_nodes):
    '''
    input: (N, n_col), n_nodes: (B)
    '''
    feat_list = []
    cnt = 0
    for n in n_nodes:
        feat_list.append(inputs[cnt : cnt + n])
        cnt += n
    blocks = torch.block_diag(*feat_list)

    return blocks  # (N, n_col*B)


def unpack_block(inputs, n_col, n_nodes):
    '''
    input: (N, B*n_col), n_col: int, n_nodes: (B)
    '''
    feat_list = []
    cnt = 0
    start_col = 0
    for n in n_nodes:
        feat_list.append(inputs[cnt:cnt + n, start_col:start_col + n_col])
        cnt += n
        start_col += n_col

    return torch.cat(feat_list, dim=0)  # (N, n_col)


def batch_repeat(inputs, n_col, n_nodes):
    '''
    input: (B*n_col), n_col: int, n_nodes: (B)
    '''
    x_list = []
    cnt = 0
    for n in n_nodes:
        x = inputs[cnt:cnt + n_col].repeat((n, 1))  # (n, n_col)
        x_list.append(x)
        cnt += n_col

    return torch.cat(x_list, dim=0)

def full_attention_conv(qs, ks, vs, kernel, n_nodes, block_wise=False, output_attn=False):
    '''
    qs: query tensor [N, D]
    ks: key tensor [N, D]
    vs: value tensor [N, D]
    n_nodes: num of nodes per graph [B]

    return output [N, D]
    '''
    if kernel == 'simple':

        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)

            # numerator
            q_block = to_block(qs, n_nodes)  # (N, B*D)
            k_block_T = to_block(ks, n_nodes).T  # (B*D, N)
            v_block = to_block(vs, n_nodes)  # (N, B*D)
            kv_block = torch.matmul(k_block_T, v_block)  # (B*M, B*D)
            qkv_block = torch.matmul(q_block, kv_block)  # (N, B*D)
            qkv = unpack_block(qkv_block, qs.shape[1], n_nodes)  # (N, D)

            v_sum = v_block.sum(dim=0)  # (B*D,)
            v_sum = batch_repeat(v_sum, vs.shape[1], n_nodes)  # (N, D)
            numerator = qkv + v_sum  # (N, D)

            # denominator
            one_list = []
            for n in n_nodes:
                one = torch.ones((n, 1))
                one_list.append(one)
            one_block = torch.block_diag(*one_list).to(qs.device)
            k_sum_block = torch.matmul(k_block_T, one_block)  # (B*D, B)
            denom_block = torch.matmul(q_block, k_sum_block)  # (N, B)
            denominator = unpack_block(denom_block, 1, n_nodes)  # (N, 1)
            denominator += batch_repeat(n_nodes, 1, n_nodes)  # (N, 1)

            attn_output = numerator / denominator  # (N, D)

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
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1)  # [N, D]

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)
            ks_sum = torch.einsum("lm,l->m", ks, all_ones)
            attention_normalizer = torch.einsum("nm,m->n", qs, ks_sum)  # [N]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer  # [N, L, H]

    if output_attn:
        attn = None
        return attn_output, attn
    else:
        return attn_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads=1,
               kernel='simple',
               use_weight=True):
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
                output, attn = full_attention_conv(qi, ki, vi, self.kernel, n_nodes, block_wise, output_attn)  # [N, D]
                attns.append(attn)
            else:
                output = full_attention_conv(qi, ki, vi, self.kernel, n_nodes, block_wise) # [N, D]
            outputs.append(output)

        if self.num_heads > 1:
            outputs = torch.stack(outputs, dim=1) # [N, H, D]
            final_output = outputs.mean(dim=1)
        else:
            final_output = outputs[0]

        if output_attn:
            attn = torch.cat(attns, dim=-1) # [N, N, H]
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


class DIFFormerMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer,
        JK='last', num_heads=1, kernel='simple',
        alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True
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
            self.gnn_convs.append(GCNConv(emb_dim, emb_dim))
            self.attn_convs.append(DIFFormerConv(emb_dim, emb_dim, num_heads=num_heads, kernel=kernel, use_weight=use_weight))
            self.bns.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr, batch, block_wise):

        n_nodes = torch_scatter.scatter(torch.ones_like(batch), batch) # [B]

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

class VirtSAGEMolNode(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False
    ):
        super(VirtSAGEMolNode, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.atom_encoder = AtomEncoder(emb_dim)
        if drop_ratio > 0 and drop_ratio < 1:
            self.dropout_layer = torch.nn.Dropout(drop_ratio)
        else:
            self.dropout_layer = torch.nn.Sequential()

        self.convs = torch.nn.ModuleList()
        self.BNs = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.convs.append(SAGEMolConv(emb_dim, emb_dim))
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
                virt_emb_temp = global_add_pool(h_list[layer], batch) + virt_emb
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


class DIFFormerMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer,
        JK='last', pooling='mean',
            num_heads=1, kernel='simple',
            alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True
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

        self.model = DIFFormerMolNode(emb_dim, num_layer, JK, num_heads, kernel,
        alpha, dropout, use_bn, use_residual, use_weight)

    def forward(self, x, edge_index, edge_attr, batch, block_wise):
        node_feat = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, block_wise=block_wise)
        graph_feat = self.pool(node_feat, batch)
        return graph_feat


class VirtSAGEMolGraph(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio=0.5,
        JK='last', residual=False, pooling='mean'
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

        self.model = VirtSAGEMolNode(
            emb_dim, num_layer, drop_ratio, JK, residual
        )

    def forward(self, batched_data):
        node_feat = self.model(batched_data)
        graph_feat = self.pool(node_feat, batched_data.batch)
        return graph_feat


class DIFFormerMol(torch.nn.Module):
    def __init__(
        self, emb_dim, num_tasks, num_layer,
        JK='last', pooling='mean', virtual=False, num_heads=1, kernel='simple',
            alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True
    ):
        super(DIFFormerMol, self).__init__()
        if virtual:
            self.model = VirtDIFFormerMolGraph(
                emb_dim=emb_dim, num_layer=num_layer, drop_ratio=drop_ratio,
                JK=JK, residual=residual, pooling=pooling
            )
        else:
            self.model = DIFFormerMolGraph(
                emb_dim=emb_dim, num_layer=num_layer, JK=JK, pooling=pooling,
                num_heads=num_heads, kernel=kernel, alpha=alpha, dropout=dropout,
                use_bn=use_bn, use_residual=use_residual, use_weight=use_weight
            )
        if pooling == 'set2set':
            self.predictor = torch.nn.Linear(2 * emb_dim, num_tasks)
        else:
            self.predictor = torch.nn.Linear(emb_dim, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch, block_wise):
        graph_feat = self.model(x, edge_index, edge_attr, batch, block_wise)
        return self.predictor(graph_feat)
