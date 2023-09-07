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
        num_nodes = torch_scatter.scatter(torch.ones_like(batch), batch)  # [B]
        shift = torch.cumsum(num_nodes, dim=0)  # [B]

        idx = torch.randint(0, edge_num, (int(ratio * edge_num),))  # [e]
        srx_idx_, end_idx_ = edge_index[:, idx]
        batchs = batch[end_idx_]  # [e]
        shifts = shift[batchs]  # [e]

        srx_idx_new = torch.randint(
            0, edge_num, (idx.shape[0], )).to(batch.device)  # [e]
        srx_idx_new = torch.remainder(srx_idx_new, shifts)  # [e]
        edge_index[0, idx] = srx_idx_new
    else:
        srx_idx, end_idx = edge_index
        mask = torch.ones_like(srx_idx).bool()
        idx = torch.randint(0, edge_num, (int(ratio * edge_num),))  # [e]
        mask[idx] = False
        edge_index = torch.stack(
            [srx_idx[mask], end_idx[mask]], dim=0)  # [2, E - e]
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


def gcn_conv(x, edge_index, edge_weight=None):
    # x: [N_tot, H, D]
    num_edge, device = edge_index.shape[1], x.device
    (row, col), N, D = edge_index, x.shape[0], x.shape[-1]
    values = edge_weight if edge_weight is not None\
        else torch.ones(num_edge).to(device)

    deg = torch.zeros(N).to(device)
    deg.scatter_add_(dim=0, index=row, src=values)

    deg_inv_sqrt = torch.zeros_like(deg)

    deg_inv_sqrt[deg > 0] = deg[deg > 0] ** (-0.5)

    adj_t_value = deg_inv_sqrt[row] * deg_inv_sqrt[col] * values

    new_feat = torch.zeros_like(x)

    new_feat.scatter_add_(
        dim=0, index=row.reshape(-1, 1, 1).repeat(1, H, D),
        src=x[col] * adj_t_value.reshape(-1, 1, 1)
    )
    return new_feat


# def gcn_conv(x, edge_index, edge_weight=None):
#     N, H = x.shape[0], x.shape[1]
#     row, col = edge_index
#     if edge_weight is None:
#         value = torch.ones(edge_index.shape[1], dtype=torch.float).to(x.device)
#     else:
#         value = edge_weight
#     adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

#     deg = adj_t.sum(dim=1).to(torch.float)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

#     x_ = []
#     for h in range(H):
#         x_h = x[:, h]  # [N, D]
#         x_.append(adj_t @ x_h)
#     return torch.stack(x_, dim=1)  # [N, H, D]


def fast_attn_conv(qs, ks, vs):
    '''
    qs: query tensor [N, H, D]
    ks: key tensor [N, H, D]
    vs: value tensor [N, H, D]
    n_nodes: num of nodes per graph [B]

    return output [N, H, D]
    '''

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
    numerator = numerator[node_mask] + v_sum[batch] # [N, H, D]

    denominator = qk_pad[node_mask].sum(dim=-1)  # [N, H]
    denominator += torch.index_select(
        n_nodes.float(), dim=0, index=batch
    ).unsqueeze(dim=-1)

    attn_output = numerator / denominator.unsqueeze(dim=-1) # [N, H, D]
    return attn_output


def norm_adj_comp(x, edge_index, edge_weight=None):
    N, H = x.shape[0], x.shape[1]
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
    return adj_t.to_dense()


def attn_comp(qs, ks, n_nodes=None, block_wise=False):
    device = qs.device
    node_mask, max_node = make_batch_mask(n_nodes, device)
    batch_size, batch = len(n_nodes), make_batch(n_nodes, device)
    q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, H, D]
    k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, H, D]
    qk_pad = torch.einsum('abcd,aecd->abce', q_pad, k_pad)
    # [B, M, H, M]
    

    useful_block = []
    for idx, np in enumerate(n_nodes):
        useful_block.append(qk_pad[idx, :np, :, :np] + 1)


    attention_normalizer = attention_num.sum(dim=1, keepdim=True)  # [N, 1, H]

    return attention_num / attention_normalizer


class GloAttnConv(nn.Module):
    '''
    one global attention layer
    '''

    def __init__(
        self, in_channels, out_channels, K_order=4, num_heads=1,
        beta=0.5, theta=0., solver='series'
    ):
        super(GloAttnConv, self).__init__()
        self.Wk = nn.Linear(in_channels, in_channels * num_heads)
        self.Wq = nn.Linear(in_channels, in_channels * num_heads)
        # self.Wo = nn.ModuleList()
        # for h in range(num_heads):
        #     self.Wo.append(nn.Linear(in_channels, out_channels))
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
        # for fc in self.Wo:
        #     fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, n_nodes=None):
        query = self.Wq(x).reshape(-1, self.num_heads, self.in_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.in_channels)

        # for h in range(self.num_heads):
        # qs, ks = query[:, h], key[:, h]
        qs = query / torch.norm(query, p=2, dim=2, keepdim=True)  # (N, H, D)
        ks = key / torch.norm(key, p=2, dim=2, keepdim=True)  # (N, H, D)

        if self.solver == 'series':
            x = x.unsqueeze(1).repeat(1, self.num_heads, 1)  # [N, H, D]
            x_ = [x]
            for i in range(self.K_order):
                gcn_i = gcn_conv(x_[-1], adj_t)
                attn_i = fast_attn_conv(qs, ks, x_[-1], n_nodes)
                x_i = self.beta * gcn_i + (1 - self.beta) * attn_i
                x_.append(x_i)
            x = torch.stack(x_, dim=-1).sum(-1) # [N, H, D, K+1] -> [N, H, D]
            x = x.reshape(-1, self.num_heads * self.in_channels) # [N, H*D]
        elif self.solver == 'inverse':
            S = attn_comp(qs, ks, n_nodes, block_wise)  # [N, N, H]
            A_tilde = norm_adj_comp(x, edge_index, edge_weight)  # [N, N]
            x_ = []
            for h in range(self.num_heads):
                A_h = (1 - self.beta) * S[:, :, h] + \
                    self.beta * A_tilde  # [N, N]
                L_h = (1 + self.theta) * \
                    torch.eye(x.shape[0]).to(A_h.device) - A_h
                x_h = torch.linalg.solve(L_h, x)  # [N, D]
                x_ += [x_h]
            x = torch.cat(x_, dim=1)  # [N, H*D]
        else:
            raise NotImplementedError

        x_out = self.Wo(x) / self.num_heads  # [N, D]

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
            n_nodes = torch_scatter.scatter(
                torch.ones_like(batch), batch)  # [B]
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

    def print_weight(self):
        for i, con in enumerate(self.convs):
            weights = con.weights[:, :64, :].detach().cpu().numpy()
            np.save(f'weights{i}.npy', weights)

    # rewiring as augmentation
    def loss_compute(self, d, criterion, args):
        logits = self.forward(d.x, d.edge_index, d.batch,
                              block_wise=args.use_block)[d.train_idx]
        y = d.y[d.train_idx]
        sup_loss = self.sup_loss_calc(y, logits, criterion, args)
        if args.use_reg:
            reg_loss_ = []
            for i in range(args.num_aug_branch):
                edge_index_i = rewiring(d.edge_index.clone(
                ), d.batch, args.modify_ratio, type=args.rewiring_type)

                logits_i = self.forward(d.x, edge_index_i, d.batch, block_wise=args.use_block)[
                    d.train_idx]
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
