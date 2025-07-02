import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch_scatter

def to_block(inputs, n_nodes):
    '''
    input: (N, H, n_col), n_nodes: (B)
    '''
    blocks = []
    for h in range(inputs.size(1)):
        feat_list = []
        cnt = 0
        for n in n_nodes:
            feat_list.append(inputs[cnt : cnt + n, h])
            cnt += n
        blocks_h = torch.block_diag(*feat_list) # (N, n_col*B)
        blocks.append(blocks_h)
    blocks = torch.stack(blocks, dim=1) # (N, H, n_col*B)
    return blocks

def unpack_block(inputs, n_col, n_nodes):
    '''
    input: (N, H, B*n_col), n_col: int, n_nodes: (B)
    '''
    unblocks = []
    for h in range(inputs.size(1)):
        feat_list = []
        cnt = 0
        start_col = 0
        for n in n_nodes:
            feat_list.append(inputs[cnt:cnt + n, h, start_col:start_col + n_col])
            cnt += n
            start_col += n_col
        unblocks_h = torch.cat(feat_list, dim=0) # (N, n_col)
        unblocks.append(unblocks_h)
    unblocks = torch.stack(unblocks, dim=1) # (N, H, n_col)
    return unblocks

def batch_repeat(inputs, n_col, n_nodes):
    '''
    input: (H, B*n_col), n_col: int, n_nodes: (B)
    '''
    x_list = []
    cnt = 0
    for n in n_nodes:
        x = inputs[:, cnt:cnt + n_col].repeat(n, 1, 1)  # (n, H, n_col)
        x_list.append(x)
        cnt += n_col
    return torch.cat(x_list, dim=0) # [N, H, n_col]

def full_attention_conv(qs, ks, vs, kernel, n_nodes=None, block_wise=False, output_attn=False):
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
            q_block = to_block(qs, n_nodes)  # (N, H, B*D)
            k_block = to_block(ks, n_nodes)  # (N, H, B*D)
            v_block = to_block(vs, n_nodes)  # (N, H, B*D)
            kvs = torch.einsum("lhm,lhd->hmd", k_block, v_block) # [H, B*D, B*D]
            attention_num = torch.einsum("nhm,hmd->nhd", q_block, kvs)  # (N, H, B*D)
            attention_num = unpack_block(attention_num, qs.shape[2], n_nodes)  # (N, H, D)

            vs_sum = v_block.sum(dim=0)  # (H, B*D)
            vs_sum = batch_repeat(vs_sum, vs.shape[2], n_nodes)  # (N, H, D)
            attention_num += vs_sum  # (N, H, D)

            # denominator
            all_ones = torch.ones([ks.shape[0], qs.shape[1]]).to(ks.device).unsqueeze(2)  # [N, H, 1]
            one_block = to_block(all_ones, n_nodes) # [N, H, B]
            ks_sum = torch.einsum("lhm,lhb->hmb", k_block, one_block) # [H, B*D, B]
            attention_normalizer = torch.einsum("nhm,hmb->nhb", q_block, ks_sum)  # [N, H, B]

            attention_normalizer = unpack_block(attention_normalizer, 1, n_nodes)  # (N, H, 1)
            attention_normalizer += batch_repeat(n_nodes.repeat(qs.shape[1], 1), 1, n_nodes)  # (N, 1)

            attn_output = attention_num / attention_normalizer  # (N, D)
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
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device) # [N]
            ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
            attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, H, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer  # [N, L, H]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
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

    def forward(self, x, n_nodes=None, block_wise=False, output_attn=False):
        # feature transformation
        query = self.Wq(x).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(x).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = x.reshape(-1, 1, self.out_channels)

        if output_attn:
            outputs, attns = full_attention_conv(query, key, value, self.kernel, n_nodes, block_wise, output_attn)  # [N, H, D]
        else:
            outputs = full_attention_conv(query, key, value, self.kernel, n_nodes, block_wise) # [N, H, D]

        final_output = outputs.mean(dim=1)

        if output_attn:
            return final_output, attns
        else:
            return final_output

class DIFFormer(nn.Module):
    '''
    DIFFormer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_layers=2, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0., use_bn=True, use_residual=True, use_weight=True,device=None):
        super(DIFFormer, self).__init__()

        self.attn_convs = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers):
            self.attn_convs.append(
                DIFFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel=kernel, use_weight=use_weight))
            self.gnn_convs.append(
                GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.attn_convs:
            conv.reset_parameters()
        for conv in self.gnn_convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None, block_wise=False, neg_edge=None):

        if block_wise:
            n_nodes = torch_scatter.scatter(torch.ones_like(batch), batch) # [B]
        else:
            n_nodes = None

        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i in range(self.num_layers):
            # graph convolution with DIFFormer layer
            x1 = self.attn_convs[i](x, n_nodes, block_wise)
            x2 = self.gnn_convs[i](x, edge_index)
            # x = torch.cat([x1, x2], dim=-1)
            x = (x1 + x2) / 2.
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        # output MLP layer
        x_out = self.fcs[-1](x)
        return x_out