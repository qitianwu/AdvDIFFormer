import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_scatter
from modules.utils import get_pool


def gnn_high_order_conv(x, edge_index, K, edge_attr):
    (N, Dim), device = x.shape, x.device
    deg_i = torch.zeros(N).to(device)
    deg_j = torch.zeros(N).to(device)

    (row, col), num_edge = edge_index, edge_index.shape[1]


    deg_i.index_add_(dim=0, index=row, source=torch.ones(num_edge).to(device))
    deg_j.index_add_(dim=0, index=col, source=torch.ones(num_edge).to(device))

    deg_inv_sqrt_i = torch.zeros(N).to(device)
    deg_inv_sqrt_i[deg_i > 0] = deg_i[deg_i > 0] ** (-0.5)

    deg_inv_sqrt_j = torch.zeros(N).to(device)
    deg_inv_sqrt_j[deg_j > 0] = deg_j[deg_j > 0] ** (-0.5)

    adj_t_val = torch.index_select(deg_inv_sqrt_i, dim=0, index=row)\
        * torch.index_select(deg_inv_sqrt_j, dim=0, index=col)

    message_edge = torch.zeros_like(x).to(x.device)
    edge_attr_src = adj_t_val.unsqueeze(-1) * edge_attr 
    message_edge.scatter_add_(dim=0, index=row.unsqueeze(-1).repeat(1, Dim), src=edge_attr_src)

    adj_t = torch.sparse_coo_tensor(edge_index, adj_t_val, size=(N, N))

    xs = [x]
    for _ in range(1, K + 1):
        xs += [torch.matmul(adj_t, xs[-1]) + message_edge]

    return torch.cat(xs, dim=1)

    

# def gnn_high_order_conv(x, edge_index, K, edge_attr):

#     N, Dim = x.shape
#     row, col = edge_index
#     adj_t = torch.zeros((N, N)).to(x.device)
#     adj_t[row, col] = 1
#     deg = adj_t.sum(dim=1)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

#     xs = [x]
#     for _ in range(1, K + 1):
#         message_edge = torch.zeros_like(x).to(x.device)
#         edge_src = adj_t[row, col].unsqueeze(-1) * edge_attr
#         edge_idx = col.unsqueeze(-1).repeat(1, Dim)
#         message_edge.scatter_add_(dim=0, index=edge_idx, src=edge_src)
#         xs += [(adj_t @ x) + message_edge]
#         adj_t = torch.matmul(adj_t, adj_t)

#     return torch.cat(xs, dim=1)  # [N, D * (1+K)]

def rewiring(edge_index, batch, ratio, edge_attr=None, type='delete'):
    edge_num, batch_size = edge_index.shape[1], batch.max() + 1

    if type == 'replace':
        num_nodes = torch_scatter.scatter(torch.ones_like(batch), batch) # [B]
        shift = torch.cumsum(num_nodes, dim=0) # [B]

        idx = torch.randint(0, edge_num, (int(ratio * edge_num),)) # [e]
        srx_idx_, end_idx_ = edge_index[:, idx]
        batchs = batch[end_idx_]  # [e]
        shifts = shift[batchs] # [e]

        srx_idx_new = torch.randint(0, edge_num, (idx.shape[0], )).to(batch.device) # [e]
        srx_idx_new = torch.remainder(srx_idx_new, shifts) # [e]
        edge_index[0, idx] = srx_idx_new
    else:
        srx_idx, end_idx = edge_index
        mask = torch.ones_like(srx_idx).bool()
        idx = torch.randint(0, edge_num, (int(ratio * edge_num),))  # [e]
        mask[idx] = False
        edge_index = torch.stack([srx_idx[mask], end_idx[mask]], dim=0) # [2, E - e]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    if edge_attr is not None:
        return edge_index, edge_attr
    else:
        return edge_index


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
    n_heads, model_dim = feat.shape[-2:]
    feat_shape = (batch_size, max_node, n_heads, model_dim)
    new_feat = torch.zeros(feat_shape).to(feat)
    new_feat[mask] = feat
    return new_feat


def full_attention_conv(
    qs, ks, vs, kernel, n_nodes=None, block_wise=False,
    output_attn=False
):
    '''
    qs: query tensor [N, H, D]
    ks: key tensor [N, H, D]
    vs: value tensor [N, H, D]
    n_nodes: num of nodes per graph [B]

    return output [N, H, D]
    '''
    if kernel == 'simple':
        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True)  # (N, H, D)
            ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True)  # (N, H, D)

            # numerator

            device = qs.device
            node_mask, max_node = make_batch_mask(n_nodes, device)
            batch_size, batch = len(n_nodes), make_batch(n_nodes, device)

            q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, H, D]
            k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, H, D]
            v_pad = to_pad(vs, node_mask, max_node, batch_size)  # [B, M, H, D]
            qk_pad = torch.einsum('abcd,aecd->abce', q_pad, k_pad)
            # [B, M, H, M]

            n_heads, v_dim = vs.shape[-2:]
            v_sum = torch.zeros((batch_size, n_heads, v_dim)).to(device)
            v_idx = batch.reshape(-1, 1, 1).repeat(1, n_heads, v_dim)
            v_sum.scatter_add_(dim=0, index=v_idx, src=vs)  # [B, H, D]

            numerator = torch.einsum('abcd,adce->abce', qk_pad, v_pad)
            numerator = numerator[node_mask] + \
                torch.index_select(v_sum, dim=0, index=batch)
            # [N, H, D]

            denominator = qk_pad[node_mask].sum(dim=-1)  # [N, H]
            denominator += torch.index_select(
                n_nodes.float(), dim=0, index=batch
            ).unsqueeze(dim=-1)

            attn_output = numerator / denominator.unsqueeze(dim=-1) # [N, H, D]

        else:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True)  # (N, H, D)
            ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True)  # (N, H, D)
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
            attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
            # [N, H, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)  # [N]
            ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
            attention_normalizer = torch.einsum(
                "nhm,hm->nh", qs, ks_sum
            )  # [N, H]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape)
            )  # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, H, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) /\
                    attention_normalizer  # [N, L, H]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

class GloAttnConv(nn.Module):
    '''
    one global attention layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               kernel='simple'):
        super(GloAttnConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()

    def forward(self, x, n_nodes=None, block_wise=False, output_attn=False):
        # feature transformation
        query = self.Wq(x).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.out_channels)
        value = x.unsqueeze(1).repeat(1, self.num_heads, 1)

        if output_attn:
            outputs, attns = full_attention_conv(query, key, value, self.kernel, n_nodes, block_wise, output_attn)  # [N, H, D]
        else:
            outputs = full_attention_conv(query, key, value, self.kernel, n_nodes, block_wise) # [N, H, D]

        outputs = outputs.reshape(-1, self.num_heads * self.out_channels)

        if output_attn:
            return outputs, attns
        else:
            return outputs


class Ours(nn.Module):
    def __init__(
        self, emb_dim, num_layer, num_tasks, num_heads=1, kernel='simple', K_order=3, 
        alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, pooling='mean'
    ):
        super(Ours, self).__init__()
        self.attn_convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = torch.nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layer):
            self.attn_convs.append(
                GloAttnConv(emb_dim, emb_dim, num_heads=num_heads, kernel=kernel))
            self.fcs.append(nn.Linear(emb_dim * (K_order+num_heads+1), emb_dim))
            self.bns.append(nn.LayerNorm(emb_dim))
            self.bond_encoder.append(BondEncoder(emb_dim))


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

        if pooling == 'set2set':
            self.fcs.append(torch.nn.Linear(emb_dim * 2, num_tasks))
        else:
            self.fcs.append(torch.nn.Linear(emb_dim, num_tasks))

        self.num_layers = num_layer

        self.K_order = K_order
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.residual = use_residual
        self.use_bn = use_bn
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.attn_convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, block_wise=False):

        if block_wise:
            n_nodes = torch_scatter.scatter(torch.ones_like(batch), batch) # [B]
        else:
            n_nodes = None

        x = self.atom_encoder(x)

        # store as residual link
        layer_ = [x]

        for i in range(self.num_layers):
            x1 = self.attn_convs[i](x, n_nodes, block_wise)
            x2 = gnn_high_order_conv(x, edge_index, self.K_order, self.bond_encoder[i](edge_attr))
            x = torch.cat([x1, x2], dim=-1) # [N, D * (1+K+H)]
            x = self.fcs[i](x)
            if self.residual:
                x += layer_[i]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.dropout_fun(torch.relu(x))
            layer_.append(x)

        x = self.pool(x, batch)
        return self.fcs[-1](x)



