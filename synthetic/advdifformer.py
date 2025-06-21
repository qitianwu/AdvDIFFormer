import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
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


def gcn_conv(x, adj_t):
    '''
    x: [N, H, D]
    '''
    x_ = []
    for h in range(x.shape[1]):
        x_h = x[:, h] # [N, D]
        x_.append(adj_t @ x_h)
    return torch.stack(x_, dim=1) # [N, H, D]

def attn_conv(qs, ks, vs, n_nodes=None, block_wise=False):
    '''
    qs: [N, H, D]
    ks: [N, H, D]
    vs: [N, H, D]
    '''
    N = qs.shape[0]

    if block_wise:
        # numerator
        q_block = to_block(qs, n_nodes)  # (N, H, B*D)
        k_block = to_block(ks, n_nodes)  # (N, H, B*D)
        v_block = to_block(vs, n_nodes)  # (N, H, B*D)
        kvs = torch.einsum("lhm,lhd->hmd", k_block, v_block)  # [H, B*D, B*D]
        qkvs = torch.einsum("nhm,hmd->nhd", q_block, kvs)  # (N, H, B*D)
        qkvs = unpack_block(qkvs, qs.shape[2], n_nodes)  # (N, H, D)

        vs_sum = v_block.sum(dim=0)  # (H, B*D)
        vs_sum = batch_repeat(vs_sum, vs.shape[2], n_nodes)  # (N, H, D)
        attn_conv_num = qkvs + vs_sum  # (N, H, D)

        # denominator
        all_ones = torch.ones([ks.shape[0], qs.shape[1]]).to(ks.device).unsqueeze(2)  # [N, H, 1]
        one_block = to_block(all_ones, n_nodes)  # [N, H, B]
        ks_sum = torch.einsum("lhm,lhb->hmb", k_block, one_block)  # [H, B*D, B]
        qks_sum = torch.einsum("nhm,hmb->nhb", q_block, ks_sum)  # [N, H, B]

        qks_sum = unpack_block(qks_sum, 1, n_nodes)  # (N, H, 1)
        attn_conv_den = qks_sum + batch_repeat(n_nodes.repeat(qs.shape[1], 1), 1, n_nodes)  # (N, H, 1)

    else:
        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        qkvs = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
        attn_conv_num = qkvs + vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)  # [N]
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        qks_sum = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        qks_sum = torch.unsqueeze(qks_sum, len(qks_sum.shape))  # [N, H, 1]
        attn_conv_den = qks_sum + torch.ones_like(qks_sum) * N

    return attn_conv_num / attn_conv_den  # [N, H, D]

def norm_adj_comp(num_nodes, edge_index, edge_weight=None):
    row, col = edge_index
    if edge_weight is None:
        value = torch.ones(edge_index.shape[1], dtype=torch.float).to(edge_index.device)
    else:
        value = edge_weight
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))

    deg = adj_t.sum(dim=1).to(torch.float)
    # deg_inv_sqrt = deg.pow(-0.5)
    # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    deg_inv_sqrt = deg.pow(-1.0)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t

def Laplacian_comp(qs, ks, A, beta, theta, n_nodes=None, block_wise=False):
    '''
    qs: [N, H, D]
    ks: [N, H, D]
    '''
    if block_wise:
        q_block = to_block(qs, n_nodes)  # (N, H, B*D)
        k_block = to_block(ks, n_nodes)  # (N, H, B*D)
        qks = torch.einsum("nhd,lhd->nlh", q_block, k_block)  # [N, N, H]
        ones_block = torch.zeros_like(qks)
        cnt = 0
        for n in n_nodes:
            ones_block[cnt:cnt + n, cnt:cnt + n, :] = 1.0
            cnt += n
        attn_num = qks + ones_block  # (N, N, H)
        attn_den = attn_num.sum(dim=1, keepdim=True) # [N, 1, H]

    else:
        qks = torch.einsum("nhd,lhd->nlh", qs, ks)  # [N, N, H]
        attn_num = qks + torch.ones_like(qks)  # (N, N, H)
        attn_den = attn_num.sum(dim=1, keepdim=True)  # [N, 1, H]

    S = attn_num / attn_den  # [N, N, H]

    A_prime = S + beta * A.unsqueeze(-1).repeat(1, 1, qs.shape[1]) # [N, N, H]
    L = (1 + theta) * torch.eye(qs.shape[0]).to(qs.device).unsqueeze(-1).repeat(1, 1, qs.shape[1]) - A_prime

    return L

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
        if solver == 'inverse':
            self.Wo = nn.Linear(in_channels * num_heads, out_channels)
        else:
            self.Wo = nn.Linear(in_channels * num_heads * (K_order+1), out_channels)

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

    def forward(self, x, edge_index, edge_weight=None, n_nodes=None, block_wise=False):
        query = self.Wq(x).reshape(-1, self.num_heads, self.in_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.in_channels)
        qs = query / torch.norm(query, p=2, dim=2, keepdim=True)  # (N, H, D)
        ks = key / torch.norm(key, p=2, dim=2, keepdim=True)  # (N, H, D)

        adj_t = norm_adj_comp(x.shape[0], edge_index, edge_weight)  # [N, N] sparse matrix
        if self.solver == 'series':
            x = x.unsqueeze(1).repeat(1, self.num_heads, 1)  # [N, H, D]
            x_ = [x]
            for i in range(self.K_order):
                gcn_i = gcn_conv(x_[-1], adj_t)
                attn_i = attn_conv(qs, ks, x_[-1], n_nodes, block_wise)
                x_i = self.beta * gcn_i + attn_i
                x_.append(x_i)
            # x = torch.stack(x_, dim=-1).sum(-1) # [N, H, D, K+1] -> [N, H, D]
            x = torch.concat(x_, dim=-1)  # [N, H, D*(K+1)]
            x = x.reshape(-1, self.num_heads * self.in_channels * (self.K_order+1)) # [N, H*D*(K+1)]
            x_out = self.Wo(x) / self.num_heads  # [N, D]
        elif self.solver == 'inverse':
            L = Laplacian_comp(qs, ks, adj_t.to_dense(), self.beta, self.theta, n_nodes, block_wise) # [N, N, H]
            x_ = []
            for h in range(self.num_heads):
                L_h = L[:, :, h] # [N, N]
                x_h = torch.linalg.solve(L_h, x) # [N, D]
                x_ += [x_h]
            x = torch.cat(x_, dim=1) # [N, H*D]
            x_out = self.Wo(x) / self.num_heads  # [N, D]
        else:
            raise NotImplementedError

        return x_out

class AdvDIFFormer(nn.Module):
    '''
    AdvDIFFormer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, num_heads=1, solver='series',
                    K_order=3, beta=0.5, theta=0., dropout=0.5, use_bn=True, use_residual=True):
        super(AdvDIFFormer, self).__init__()

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

    def loss_compute(self, d, criterion, args):
        logits = self.forward(d.x, d.edge_index, d.batch, block_wise=args.use_block)[d.train_idx]
        y = d.y[d.train_idx]
        sup_loss = self.sup_loss_calc(y, logits, criterion, args)
        loss = sup_loss
        return loss