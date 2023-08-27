import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import (
    erdos_renyi_graph, remove_self_loops, add_self_loops,
    degree, add_remaining_self_loops, negative_sampling
)
from torch_geometric.nn import GCNConv
from data_utils import (
    sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
)
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
            0, edge_num, (idx.shape[0], )
        ).to(batch.device)  # [e]
        srx_idx_new = torch.remainder(srx_idx_new, shifts)  # [e]
        edge_index[0, idx] = srx_idx_new
    else:
        srx_idx, end_idx = edge_index
        mask = torch.ones_like(srx_idx).bool()
        idx = torch.randint(0, edge_num, (int(ratio * edge_num),))  # [e]
        mask[idx] = False
        edge_index = torch.stack(
            [srx_idx[mask], end_idx[mask]], dim=0
        )  # [2, E - e]
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
        attn = None
        return attn_output, attn
    else:
        return attn_output


class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''

    def __init__(
        self, in_channels,
        out_channels,
        num_heads,
        kernel='simple',
        use_weight=True
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

    def forward(self, x, n_nodes=None, block_wise=False, output_attn=False):
        # feature transformation
        query = self.Wq(x).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(x).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = x.reshape(-1, 1, self.out_channels)

        if output_attn:
            outputs, attns = full_attention_conv(
                query, key, value, self.kernel, n_nodes,
                block_wise, output_attn
            )  # [N, H, D]
        else:
            outputs = full_attention_conv(
                query, key, value, self.kernel, n_nodes, block_wise
            )  # [N, H, D]

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

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=2,
        num_heads=1, kernel='simple', alpha=0.5, dropout=0.5, use_bn=True,
        use_residual=True, use_weight=True
    ):
        super(DIFFormer, self).__init__()

        self.attn_convs = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels * 2))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * 2))
        for i in range(num_layers):
            self.attn_convs.append(DIFFormerConv(
                hidden_channels * 2, hidden_channels, num_heads=num_heads,
                kernel=kernel, use_weight=use_weight
            ))
            self.gnn_convs.append(
                GCNConv(hidden_channels * 2, hidden_channels)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * 2))

        self.fcs.append(nn.Linear(hidden_channels * 2, out_channels))

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

    def forward(
        self, x, edge_index, batch,
        edge_weight=None, block_wise=False
    ):

        if block_wise:
            n_nodes = torch_scatter.scatter(
                torch.ones_like(batch), batch
            )  # [B]
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
            x = torch.cat([x1, x2], dim=-1)
            # x = (x1 + x2) / 2.
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
        logits = self.forward(d.x, d.edge_index, d.batch,
                              block_wise=args.use_block)[d.train_idx]
        y = d.y[d.train_idx]
        sup_loss = self.sup_loss_calc(y, logits, criterion, args)
        if args.use_reg:
            reg_loss_ = []
            # num_r = int(d.edge_index.shape[1] * args.ratio_rewiring)
            for i in range(args.num_aug_branch):
                edge_index_i = rewiring(
                    d.edge_index.clone(), d.batch, args.modify_ratio,
                    type=args.rewiring_type
                )

                logits_i = self.forward(
                    d.x, edge_index_i, d.batch, block_wise=args.use_block
                )[d.train_idx]
                reg_loss_i = self.sup_loss_calc(y, logits_i, criterion, args)
                reg_loss_.append(reg_loss_i)
            var = torch.var(torch.stack(reg_loss_))
            print(sup_loss, var)
            # reg_loss = torch.stack(reg_loss_).max()
            loss = sup_loss + args.reg_weight * var
        else:
            loss = sup_loss
        return loss
