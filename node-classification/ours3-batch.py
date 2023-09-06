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

def attn_conv(qs, ks, vs, return_attn=False):
    '''
    qs: [B, N, H, D]
    ks: [B, N, H, D]
    vs: [B, N, H, D]
    return attn: return propagation results (False) or attention matrix (True)
    '''
    N = qs.shape[0]

    if return_attn:
        qks = torch.einsum("bnhd,blhd->bnlh", qs, ks)  # [B, N, N, H]
        attn_num = qks + torch.ones_like(qks)  # (B, N, N, H)
        attn_den = attn_num.sum(dim=2, keepdim=True)  # [B, N, 1, H]

        return attn_num / attn_den  # [B, N, N, H]
    else:
        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs) # [B, H, D, D]
        qkvs = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, N, H, D]
        all_ones = torch.ones([vs.shape[1]]).to(vs.device) # [N]
        vs_sum = torch.einsum("l,blhd->bhd", all_ones, vs)  # [B, H, D]
        attn_conv_num = qkvs + vs_sum.unsqueeze(1).repeat(1, vs.shape[1], 1, 1)  # [B, N, H, D]

        # denominator
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones) # [B, H, D]
        attn_conv_den = torch.einsum("bnhm,bhm->bnh", qs, ks_sum).unsqueeze(-1)  # [B, N, H, 1]
        attn_conv_den += torch.ones_like(attn_conv_den) * N

        return attn_conv_num / attn_conv_den  # [B, N, H, D]

def gcn_conv(xs, adj_t):
    '''
    xs: [N_total, H, D]
    '''
    x_ = []
    for h in range(xs.shape[1]):
        x_h = xs[:, h] # [N, D]
        x_.append(adj_t @ x_h)
    return torch.stack(x_, dim=1) # [N, H, D]

def norm_adj(num_nodes, edge_index, edge_weight=None):
    '''
    compute A_tilde for a batch of graphs stored as a diagonal-block A [N_total, N_total]
    '''
    row, col = edge_index
    if edge_weight is None:
        value = torch.ones(edge_index.shape[1], dtype=torch.float).to(x.device)
    else:
        value = edge_weight
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))

    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t

class GloAttnConv(nn.Module):
    '''
    one global attention layer
    implement a solver for approximating the closed-form solution of diffusion PDE
    when solver is series: use series expansion for approximation
    when solver is inverse: use Chebyshev technique for approximation
    '''
    def __init__(self, in_channels, out_channels, num_heads=1, beta=0.5, solver='series', K_order=4, theta=0.):
        super(GloAttnConv, self).__init__()
        self.Wk = nn.Linear(in_channels, in_channels * num_heads)
        self.Wq = nn.Linear(in_channels, in_channels * num_heads)
        self.Wo = nn.Linear(in_channels * num_heads, out_channels)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.K_order = K_order
        self.beta = beta
        self.num_heads = num_heads
        self.solver = solver
        self.theta = theta

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        '''
        input a batch of graphs stored as a diagonal-block form
        x: [N_total, D]
        edge_index: [2, E_total]
        edge_weight: [2, E_total]
        batch: [N_total] indicates each node belonging to the graph id in the batch
        '''
        query = self.Wq(x).reshape(-1, self.num_heads, self.in_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.in_channels)
        qs = query / torch.norm(query, p=2, dim=2, keepdim=True)  # (N_total, H, D)
        ks = key / torch.norm(key, p=2, dim=2, keepdim=True)  # (N_total, H, D)

        adj_t = norm_adj(x.shape[0], edge_index, edge_weight)  # [N_total, N_total] sparse matrix
        xs = x.unsqueeze(1).repeat(1, self.num_heads, 1)  # [N_total, H, D]
        qs_batch = block_to_batch(qs, batch)  # [B, N, H, D], N for max_num_nodes
        ks_batch = block_to_batch(ks, batch)  # [B, N, H, D]

        if self.solver == 'series':
            xs_batch = block_to_batch(xs, batch)  # [B, N, H, D]
            for i in range(self.K_order):
                gcn_i = gcn_conv(xs, adj_t) # [N_total, H, D]
                attn_i = attn_conv(qs_batch, ks_batch, xs_batch)
                attn_i = batch_to_block(attn_i, batch) # [N_total, H, D]
                xs += self.beta * gcn_i + (1 - self.beta) * attn_i # [N_total, H, D]
            x = xs.reshape(-1, self.num_heads * self.in_channels) # [N_total, H*D]
        elif self.solver == 'inverse':
            S = attn_comp(qs_batch, ks_batch) # [B, N, N, H]
            A_batch = block_to_batch(adj_t.to_dense(), batch)  # [B, N, N]
            x_batch = block_to_batch(x, batch)  # [B, N, D]
            x_ = []
            for h in range(self.num_heads):
                A_h = (1 - self.beta) * S[:, :, :, h] + self.beta * A_batch  # [B, N, N]
                L_h = (1 + self.theta) * torch.eye(xs_batch.shape[1]).unsqueeze(0).to(A_h.device) - A_h # [B, N, N]
                x_h = torch.linalg.solve(L_h, x_batch)  # [B, N, D]
                x_h = batch_to_block(x_h, batch) # [N_total, D]
                x_ += [x_h]
            x = torch.cat(x_, dim=1)  # [N_total, H*D]
        else:
            raise NotImplementedError

        x_out = self.Wo(x) / self.num_heads # [N_total, D]

        return x_out

class PCFormer(nn.Module):
    '''
    PCFormer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, num_heads=1, solver='series',
                    K_order=3, beta=0.5, theta=0., dropout=0.5, use_bn=True, use_residual=True):
        super(PCFormer, self).__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GloAttnConv(hidden_channels, hidden_channels, K_order=K_order, num_heads=num_heads, beta=beta, theta=theta, solver=solver))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.num_layers = num_layers
        self.K_order = K_order

    def reset_parameters(self):
        for conv in self.convs:
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

        # store each head
        layer_.append(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight, n_nodes, block_wise)
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
        elif args.dataset in ('synthetic'):
            loss = criterion(pred, y)
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

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
                reg_loss_.append(reg_loss_i)
                # reg_loss_.append(torch.relu(reg_loss_i - sup_loss) ** 2)
            # loss_reg = torch.mean(torch.stack(reg_loss_))
            reg_loss = torch.mean(torch.stack(reg_loss_))
            print(sup_loss.data, reg_loss.data)
            loss = sup_loss + args.reg_weight * reg_loss
        else:
            loss = sup_loss
        return loss