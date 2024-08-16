from GCN import GCN
from GAT import GAT
from ours_series import OursSeries
from difformer import DIFFormer
from graphgps import GPSModel
import torch
from tqdm import tqdm
import argparse
from dataset import atom_to_idx, bond_to_idx
import torch_geometric
import json
import os
from sklearn.metrics import adjusted_mutual_info_score


def make_graph_sample(graph):
    num_nodes = len(graph['nodes'])
    node_feat = torch.zeros(num_nodes).long()
    for nd in graph['nodes']:
        node_feat[nd['id']] = atom_to_idx[nd['element']]
    edge_feat, edge_idx = [], []
    for eg in graph['edges']:
        edge_idx.append((eg['source'], eg['target']))
        edge_feat.append(bond_to_idx[eg['bondtype']])

        edge_idx.append((eg['target'], eg['source']))
        edge_feat.append(bond_to_idx[eg['bondtype']])

    edge_idx = torch.LongTensor(edge_idx).transpose(0, 1)
    edge_feat = torch.LongTensor(edge_feat)
    batch = torch.zeros(num_nodes).long()
    ptr = torch.LongTensor([0, 1])

    return torch_geometric.data.Data(**{
        'x': node_feat, 'edge_attr': edge_feat,
        'edge_index': edge_idx, 'num_nodes': num_nodes,
        'batch': batch, "ptr": ptr
    })


def dfs(x, graph, block, vis):
    block.append(x)
    vis.add(x)
    for neighbor in graph[x]:
        if neighbor not in vis:
            dfs(neighbor, graph, block, vis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--K_order', type=int, default=3)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    if args.gnn == 'gcn':
        model = GCN(args.dim, args.num_layer, 0.0).to(device)
    elif args.gnn == 'gat':
        model = GAT(args.dim, args.num_layer, 0.0, args.heads).to(device)
    elif args.gnn == 'ours-series':
        model = OursSeries(
            emb_dim=args.dim, num_layer=args.num_layer, num_heads=args.heads,
            K_order=args.K_order, alpha=args.alpha, dropout=0,
            use_bn=True, use_residual=True,
        ).to(device)
    elif args.gnn == 'difformer':
        model = DIFFormer(
            emb_dim=args.dim, num_layer=args.num_layer, num_heads=args.heads,
            alpha=args.alpha, dropout=0, use_bn=True,
            use_residual=True, use_weight=True,
        ).to(device)
    elif args.gnn == 'graphgps':
        model = GPSModel(
            hidden_channels=args.dim, num_layers=args.num_layer,
            num_heads=args.heads, dropout=0, attn_dropout=0, use_bn=True
        ).to(device)

    model_weight = torch.load(args.model_dir, map_location=device)
    model.load_state_dict(model_weight)

    model = model.eval()
    AMIS = []

    out_dir = os.path.join(args.output_dir, args.gnn)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fx in tqdm(os.listdir(args.data_dir)):
        if not fx.endswith('.json'):
            continue
        with open(os.path.join(args.data_dir, fx)) as Fin:
            INFO = json.load(Fin)

        graph = make_graph_sample(INFO).to(device)

        with torch.no_grad():
            result = torch.sigmoid(model(graph))

        px,  vis, c_nodes = {}, set(), []
        for idx in range(graph.edge_index.shape[1]):
            i = graph.edge_index[0, idx].item()
            j = graph.edge_index[1, idx].item()
            px[(i, j)] = result[idx].item()

        g_es = {i: [] for i in range(graph.num_nodes)}
        for idx in range(graph.edge_index.shape[1]):
            i = graph.edge_index[0, idx].item()
            j = graph.edge_index[1, idx].item()
            if px[(i, j)] + px[(i, j)] < 1:
                g_es[i].append(j)

        for i in range(graph.num_nodes):
            if i not in vis:
                block = []
                dfs(i, g_es, block, vis)
                c_nodes.append(block)

        gt_labels = -torch.ones(graph.num_nodes).long()
        pred_labels = -torch.ones(graph.num_nodes).long()

        for idx, x in enumerate(INFO['cgnodes']):
            gt_labels[x] = idx

        for idx, x in enumerate(c_nodes):
            pred_labels[x] = idx

        assert torch.all(gt_labels != -1).item(), 'Invalid GT'
        assert torch.all(pred_labels != -1).item(), 'Invalid pred'

        gt_labels = gt_labels.tolist()
        pred_labels = pred_labels.tolist()

        ami = adjusted_mutual_info_score(gt_labels, pred_labels)

        # print(ami)
        # print(INFO['cgnodes'])
        # print(c_nodes)
        # print(gt_labels)
        # print(pred_labels)
        AMIS.append(ami)

        INFO['pred'], INFO['AMI'] = c_nodes, ami
        with open(os.path.join(out_dir, fx), 'w') as Fout:
            json.dump(INFO, Fout, indent=4)

    print('[AVG AMI]', sum(AMIS) / len(AMIS))
