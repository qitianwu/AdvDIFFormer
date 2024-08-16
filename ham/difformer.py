from GCN import GCNConv
import torch.nn as nn
import torch

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


def full_attention_conv(qs, ks, vs, n_nodes):

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
    kv_pad = torch.matmul(k_pad.transpose(2, 1), v_pad)  # [B, D, D]

    v_sum = torch.zeros((batch_size, vs.shape[-1])).to(device)
    v_sum.index_add_(source=vs, index=batch, dim=0)  # [B, D]

    k_sum = torch.zeros((batch_size, ks.shape[-1])).to(device)
    k_sum.index_add_(source=ks, index=batch, dim=0) # [B, D]

    numerator = torch.matmul(q_pad, kv_pad)[node_mask] + \
        torch.index_select(v_sum, dim=0, index=batch)
    # [N, D]

    denominator = torch.einsum('abc,ac->ab', q_pad, k_sum) # [B, M]
    denominator = denominator[node_mask] + torch.index_select(
        n_nodes.float(), dim=0, index=batch
    ) # [N]

    attn_output = numerator / denominator.unsqueeze(dim=-1)

    return attn_output


class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''

    def __init__(
        self, in_channels, out_channels, num_heads=1, use_weight=True
    ):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, x, n_nodes):
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
            output = full_attention_conv(qi, ki, vi, n_nodes)  # [N, D]
            outputs.append(output)

        if self.num_heads > 1:
            outputs = torch.stack(outputs, dim=1)  # [N, H, D]
            final_output = outputs.mean(dim=1)
        else:
            final_output = outputs[0]

        return final_output


class DIFFormer(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, num_heads=1, alpha=0.5, dropout=0.5,
        use_bn=True, use_residual=True, use_weight=True,
    ):
        super(DIFFormer, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.residual = use_residual
        self.use_bn = use_bn
        self.use_weight = use_weight
        self.dropout = dropout
        self.alpha = alpha
        self.atom_encoder = torch.nn.Embedding(16, emb_dim)

        self.gnn_convs = torch.nn.ModuleList()
        self.attn_convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.num_layer):
            self.gnn_convs.append(GCNConv(emb_dim))
            self.attn_convs.append(DIFFormerConv(
                emb_dim, emb_dim, num_heads=num_heads,
                use_weight=use_weight
            ))
            self.bns.append(nn.BatchNorm1d(emb_dim))
        self.edge_predictor = torch.nn.Linear(emb_dim * 2, 1)
        self.drop_fun = torch.nn.Dropout(dropout)

    def forward(self, graph):
        x, edge_index, = graph.x, graph.edge_index,
        edge_attr, batch = graph.edge_attr, graph.batch

        n_nodes = torch.zeros(batch.max().item() + 1).long().to(x.device)
        n_nodes.index_add_(index=batch, dim=0, source=torch.ones_like(batch))

        layer_ = [self.atom_encoder(x)]

        for i in range(self.num_layer):
            x1 = self.gnn_convs[i](layer_[i], edge_index, edge_attr)
            x2 = self.attn_convs[i](layer_[i], n_nodes)
            x = (x1 + x2) / 2.
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i](x)
            x = torch.relu(x)
            x = self.drop_fun(x)
            layer_.append(x)
        node_repr = layer_[-1]
        row, col = edge_index
        return self.edge_predictor(
            torch.cat([node_repr[row], node_repr[col]], dim=-1)
        )
