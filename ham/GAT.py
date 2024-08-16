import torch.nn.functional as F
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.utils import softmax as sp_softmax
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.inits import glorot, zeros


class MyGATConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, heads=1,
        negative_slope=0.2, dropout=0.1, add_self_loop=True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(MyGATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.dropout_fun = torch.nn.Dropout(dropout)

        self.lin_src = self.lin_dst = Linear(
            in_channels, out_channels * heads,
            bias=False, weight_initializer='glorot'
        )
        self.add_self_loop = add_self_loop

        self.att_src = torch.nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_edge = torch.nn.Parameter(torch.zeros(1, heads, out_channels))

        self.bias = torch.nn.Parameter(torch.zeros(heads * out_channels))
        self.lin_edge = torch.nn.Embedding(4, out_channels * heads)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        num_nodes = x.shape[0]
        edge_attr = self.lin_edge(edge_attr)

        if self.add_self_loop:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value='mean', num_nodes=num_nodes
            )

        H, C = self.heads, self.out_channels
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(
            edge_index, x=x, alpha=alpha, size=size,
            edge_attr=edge_attr.view(-1, H, C)
        )
        out = out.view(-1, H * C) + self.bias
        return out

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha_i + alpha_j + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.dropout_fun(sp_softmax(alpha, index, ptr, size_i))
        return alpha

    def message(self, x_j, alpha, edge_attr):
        return alpha.unsqueeze(-1) * (x_j + edge_attr)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.out_channels}, heads={self.heads})'
        )


class GAT(torch.nn.Module):
    def __init__(
        self, emb_dim, num_layer, drop_ratio, heads,
        residual=True, negative_slope=0.2
    ):
        super(GAT, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.atom_encoder = torch.nn.Embedding(16, emb_dim)
        self.edge_predictor = torch.nn.Linear(emb_dim * 2, 1)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.drop_fun = torch.nn.Dropout(drop_ratio)
        for i in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.convs.append(MyGATConv(
                in_channels=emb_dim, out_channels=emb_dim // heads,
                heads=heads, negative_slope=negative_slope,
                drop_ratio=drop_ratio
            ))

    def forward(self, graph):
        x = self.atom_encoder(graph.x)
        for i in range(self.num_layer):
            conv_res = self.batch_norms[i](self.convs[i](
                x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr
            ))
            x = self.drop_fun(torch.relu(conv_res)) + \
                (x if self.residual else 0)

        row, col = graph.edge_index
        return self.edge_predictor(torch.cat([x[row], x[col]], dim=-1))
