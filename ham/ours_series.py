import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gnn_high_order_conv(x, edge_index, K, edge_attr):
    (N, Dim), device = x.shape, x.device
    deg = torch.zeros(N).to(device)

    (row, col), num_edge = edge_index, edge_index.shape[1]

    deg.index_add_(
        dim=0, index=row,
        source=torch.ones(num_edge).to(device)
    )

    deg_inv_sqrt = torch.zeros(N).to(device)
    deg_inv_sqrt[deg > 0] = deg[deg > 0] ** (-0.5)

    adj_t_val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    message_edge = torch.zeros_like(x).to(x.device)
    edge_attr_src = adj_t_val.unsqueeze(-1) * edge_attr
    message_edge.index_add_(dim=0, source=edge_attr_src, index=row)

    adj_t = torch.sparse_coo_tensor(edge_index, adj_t_val, size=(N, N))

    xs = [x]
    for _ in range(1, K + 1):
        xs += [torch.matmul(adj_t, xs[-1]) + message_edge]

    return torch.cat(xs, dim=1)


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


def full_attention_conv(qs, ks, vs, n_nodes):
    '''
    qs: query tensor [N, H, D]
    ks: key tensor [N, H, D]
    vs: value tensor [N, H, D]
    n_nodes: num of nodes per graph [B]

    return output [N, H, D]
    '''
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

    kv_pad = torch.einsum('abcd,abce->adce', k_pad, v_pad)
    # [B, D, H, D]

    (n_heads, v_dim), k_dim = vs.shape[-2:], ks.shape[-1]
    v_sum = torch.zeros((batch_size, n_heads, v_dim)).to(device)
    v_sum.index_add_(source=vs, index=batch, dim=0)  # [B, H, D]

    numerator = torch.einsum('abcd,adce->abce', q_pad, kv_pad)
    numerator = numerator[node_mask] + v_sum[batch]

    k_sum = torch.zeros((batch_size, n_heads, k_dim)).to(device)
    k_sum.index_add_(dim=0, index=batch, source=ks)  # [B, H, D]

    denominator = torch.einsum('abcd,acd->abc', q_pad, k_sum)
    denominator = denominator[node_mask] + torch.index_select(
        n_nodes.float(), dim=0, index=batch
    ).unsqueeze(dim=-1)

    attn_output = numerator / denominator.unsqueeze(dim=-1)  # [N, H, D]

    return attn_output


class GloAttnConv(torch.nn.Module):
    '''
    one global attention layer
    '''

    def __init__(self, in_channels, out_channels, num_heads):
        super(GloAttnConv, self).__init__()
        self.Wk = torch.nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = torch.nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()

    def forward(self, x, n_nodes):
        # feature transformation
        query = self.Wq(x).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(x).reshape(-1, self.num_heads, self.out_channels)
        value = x.unsqueeze(1).repeat(1, self.num_heads, 1)

        outputs = full_attention_conv(query, key, value, n_nodes)  # [N, H, D]
        outputs = outputs.reshape(-1, self.num_heads * self.out_channels)

        return outputs


class OursSeries(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, num_heads=1,  K_order=3,
        alpha=0.5, dropout=0.5, use_bn=True, use_residual=True,
    ):
        super(OursSeries, self).__init__()
        self.attn_convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.atom_encoder = torch.nn.Embedding(16, emb_dim)
        self.bond_encoder = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layer):
            self.attn_convs.append(GloAttnConv(emb_dim, emb_dim, num_heads))
            self.fcs.append(torch.nn.Linear(
                emb_dim * (K_order + num_heads + 1), emb_dim
            ))
            self.bns.append(torch.nn.LayerNorm(emb_dim))
            self.bond_encoder.append(torch.nn.Embedding(4, emb_dim))

        self.num_layers = num_layer
        self.K_order = K_order
        self.dropout_fun = torch.nn.Dropout(dropout)
        self.residual = use_residual
        self.use_bn = use_bn
        self.edge_predictor = torch.nn.Linear(emb_dim * 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.attn_convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, graph):
        x, edge_index, = graph.x, graph.edge_index,
        edge_attr, batch = graph.edge_attr, graph.batch

        n_nodes = torch.zeros(batch.max().item() + 1).long().to(x.device)
        n_nodes.index_add_(index=batch, dim=0, source=torch.ones_like(batch))

        x = self.atom_encoder(x)

        # store as residual link
        layer_ = [x]

        for i in range(self.num_layers):
            x1 = self.attn_convs[i](x, n_nodes)
            x2 = gnn_high_order_conv(
                x=x, edge_index=edge_index, K=self.K_order,
                edge_attr=self.bond_encoder[i](edge_attr)
            )
            x = torch.cat([x1, x2], dim=-1)  # [N, D * (1+K+H)]
            x = self.fcs[i](x)
            if self.residual:
                x += layer_[i]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.dropout_fun(torch.relu(x))
            layer_.append(x)

        row, col = edge_index
        return self.edge_predictor(torch.cat([x[row], x[col]], dim=-1))
