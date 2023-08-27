import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops, add_self_loops, degree, add_remaining_self_loops, negative_sampling
from torch_geometric.nn import GCNConv
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor, matmul
import torch_scatter

def gnn_high_order_conv(x, edge_index, K, edge_weight=None):
    N = x.shape[0]
    row, col = edge_index
    if edge_weight is None:
        value = torch.ones(edge_index.shape[1], dtype=torch.float).to(x.device)
    else:
        value = edge_weight
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    xs = [x]
    for _ in range(1, K + 1):
        xs += [adj_t @ xs[-1]]

    return torch.cat(xs, dim=1)  # [N, D * (1+K)]

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
    '''
    Ours model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, kernel='simple',
                    K_order=3, alpha=0.5, dropout=0.5, use_bn=True, use_residual=True):
        super(Ours, self).__init__()

        self.attn_convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers):
            self.attn_convs.append(
                GloAttnConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel=kernel))
            self.fcs.append(nn.Linear(hidden_channels * (K_order+num_heads+1), hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.num_layers = num_layers
        self.K_order = K_order

    def reset_parameters(self):
        for conv in self.attn_convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, batch=None, edge_weight=None, block_wise=False):

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
            x1 = self.attn_convs[i](x, n_nodes, block_wise)
            x2 = gnn_high_order_conv(x, edge_index, self.K_order, edge_weight)
            x = torch.cat([x1, x2], dim=-1) # [N, D * (1+K+H)]
            x = self.fcs[1+i](x)
            if self.residual:
                x += layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        # output MLP layer
        x_out = self.fcs[-1](x)
        return x_out

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic', 'ppi', 'proteins'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def print_weight(self):
        for i, con in enumerate(self.convs):
            weights = con.weights[:, :64, :].detach().cpu().numpy()
            np.save(f'weights{i}.npy', weights)

    # rewiring as augmentation
    def loss_compute(self, d, criterion, args):
        logits = self.forward(d.x, d.edge_index, d.batch, block_wise=args.use_block)[d.train_idx]
        y = d.y[d.train_idx]
        sup_loss = self.sup_loss_calc(y, logits, criterion, args)
        if args.use_reg:
            reg_loss_ = []
            for i in range(args.num_aug_branch):

                edge_index_i = rewiring(d.edge_index.clone(), d.batch, args.modify_ratio, type=args.rewiring_type)

                logits_i = self.forward(d.x, edge_index_i, d.batch, block_wise=args.use_block)[d.train_idx]
                reg_loss_i = self.sup_loss_calc(y, logits_i, criterion, args)
                # reg_loss_.append(reg_loss_i)
                reg_loss_.append(torch.relu(reg_loss_i - sup_loss) ** 2)
            # loss_reg = torch.mean(torch.stack(reg_loss_))
            reg_loss = torch.mean(torch.stack(reg_loss_))
            print(sup_loss.data, reg_loss.data)
            loss = sup_loss + args.reg_weight * reg_loss
        else:
            loss = sup_loss
        return loss