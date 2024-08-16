import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from performer_pytorch import SelfAttention

from torch_geometric.utils import degree


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = torch.nn.Embedding(4, emb_dim)

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

def make_batch_mask(n_nodes, device='cpu'):
    max_node = n_nodes.max().item()
    mask = torch.zeros(len(n_nodes), max_node)
    for idx, nx in enumerate(n_nodes):
        mask[idx, :nx] = 1
    return mask.bool().to(device), max_node


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, in_channels,
                 num_heads,
                 dropout=0.0,
                 attn_dropout=0.0, use_bn=True):
        super().__init__()

        self.dim_h = in_channels
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.batch_norm = use_bn

        # Local message-passing model.
        self.local_model=GCNConv(in_channels)

        # Global attention transformer-style model.
        self.self_attn = SelfAttention(
            dim=in_channels, heads=num_heads,
            dropout=self.attn_dropout, causal=False)

        # Normalization for MPNN and Self-Attention representations.
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(in_channels)
            self.norm1_attn = nn.BatchNorm1d(in_channels)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(in_channels, in_channels * 2)
        self.ff_linear2 = nn.Linear(in_channels * 2, in_channels)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(in_channels)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        self.device=None

    def reset_parameters(self):
        for child in self.children():
            # print(child.__class__.__name__)
            classname=child.__class__.__name__
            if classname not in ['SelfAttention','Dropout']:
                child.reset_parameters()
        
        if self.device is None:
            param=next(iter(self.local_model.parameters()))
            self.device=param.device

        self.self_attn=SelfAttention(
            dim=self.dim_h, heads=self.num_heads,
            dropout=self.attn_dropout, causal=False).to(self.device)

    def forward(self, x, edge_index, edge_attr, n_nodes):

        batch_mask, max_node = make_batch_mask(n_nodes, x.device)

        h_in1 = x  # for first residual connection, x has shape (n, in_channels)

        h_out_list = []
        # Local MPNN with edge attributes.
        h_local=self.local_model(x,edge_index, edge_attr)
        h_local=h_in1+h_local # Residual connection.

        if self.batch_norm:
            h_local=self.norm1_local(h_local)
        h_out_list.append(h_local)

        # h_attn=self.self_attn(x.unsqueeze(0)) # (1, n, in_channels)
        # h_attn=h_attn.squeeze(0) # (n, in_channels)
        
        h_attn = torch.zeros(len(n_nodes), max_node, x.shape[-1]).to(x.device)
        h_attn[batch_mask] = x
        h_attn = self.self_attn(h_attn, mask=batch_mask)[batch_mask]


        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.batch_norm:
            h = self.norm2(h)

        return h

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
class GPSModel(nn.Module):
    def __init__(
        self, hidden_channels, num_layers, num_heads, 
        dropout, attn_dropout, use_bn
    ):
        super().__init__()

        self.pre_mp=torch.nn.Embedding(16, hidden_channels)
        self.dropout=dropout
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPSLayer(
                hidden_channels,
                num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_bn=use_bn,
            ))

        self.post_mp = nn.Linear(hidden_channels * 2, 1)
        self.reset_parameters()

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index=data.edge_index
        n_nodes = torch.zeros(batch.max().item() + 1).long().to(x.device)
        n_nodes.index_add_(index=batch, dim=0, source=torch.ones_like(batch))

        x=self.pre_mp(x)
        x=F.relu(x)
        x=F.dropout(x,self.dropout,training=self.training)
        for layer in self.layers:
            x=layer(x,edge_index, data.edge_attr, n_nodes)
        
        row, col = edge_index
        return self.post_mp(torch.cat([x[row], x[col]], dim=-1))
        
    
    def reset_parameters(self):
        self.pre_mp.reset_parameters()
        self.post_mp.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()



